"""CPU/GPU smoke test: shapes, flow conventions, DSER++ mask equivalence, forward/backward per stage.

Run from the repository root:
    python tools/smoke_test.py            # CPU, random data, no pretrained weights needed
    python tools/smoke_test.py --device cuda
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model.loss import MaskedSSIMLoss  # noqa: E402
from model.ourNet import MyNet  # noqa: E402
from model.texture import hcs_patch_masks  # noqa: E402
from model.warp import backwarp, reverse_voxel  # noqa: E402
from util.event import EventSequence  # noqa: E402
from util.utils_func import mask_reverse, pyramid_patch_mask  # noqa: E402
from util.voxelization import to_voxel_grid  # noqa: E402
import train_bsergb as tb  # noqa: E402


class L1Perceptual(nn.Module):
    """Stand-in for LPIPS so the test needs no downloads."""

    def forward(self, a, b):
        return (a - b).abs().mean(dim=(1, 2, 3), keepdim=True)


def make_args(stage, bins=8):
    return argparse.Namespace(
        bins=bins, stage=stage, beta=0.1, mask_patch_size=32, hcs_beta_mode="sum",
        ref_mode="composite", rec_use_frames=False, detach_rec_for_flow=False,
        raft_iters=3, refine_iters=1, texture_window=3, no_freeze_raft_bn=False,
        freeze_rec=False, freeze_raft=False, raft_lr_mult=0.25, rec_lr_mult=1.0, lr=1e-4,
        lambda_rec=1.0, lambda_syn=1.0, lambda_ref=1.0, lambda_ps=1.0,
        lambda_warp=0.5, lambda_smooth=0.01, warp_gamma=0.8, lambda_tc=0.1,
    )


def test_backwarp_direction(device):
    img0 = torch.rand(1, 3, 64, 64, device=device)
    shift = 3
    gt = torch.roll(img0, shifts=shift, dims=-1)        # content moved +3 px in x from 0 to t
    flow_t0 = torch.zeros(1, 2, 64, 64, device=device)
    flow_t0[:, 0] = -shift                               # F_{t->0}: x_t + F = x_0
    warped = backwarp(img0, flow_t0)
    err = (warped - gt)[..., shift:].abs().max().item()
    assert err < 1e-5, f"backwarp convention broken (err={err})"
    print("[ok] backwarp: backwarp(I0, F_t->0) reproduces I_t")


def test_reverse_voxel(bins=8):
    rng = np.random.default_rng(0)
    n = 2000
    feats = np.stack([rng.uniform(0, 63, n), rng.uniform(0, 47, n), np.sort(rng.uniform(0, 1, n)),
                      rng.choice([-1.0, 1.0], n)], axis=1).astype(np.float32)
    seq = EventSequence(feats.copy(), 48, 64)
    forward = to_voxel_grid(seq, bins)
    reversed_seq = EventSequence(feats.copy(), 48, 64).reverse()
    expected = to_voxel_grid(reversed_seq, bins)
    err = (reverse_voxel(forward.unsqueeze(0))[0] - expected).abs().max().item()
    assert err < 1e-3, f"reverse_voxel mismatch {err}"
    print("[ok] reverse_voxel equals voxelising the reversed event stream")


def test_hcs_equivalence(device):
    torch.manual_seed(0)
    img0, img1, coarse = (torch.rand(2, 3, 128, 128, device=device) for _ in range(3))
    mask = (torch.rand(2, 1, 128, 128, device=device) > 0.995).float()
    for beta in (0.1, 1010.0, 1024.0, 1040.0):  # patch sums are ~1024 for random images
        m0, m1, union = hcs_patch_masks(img0, img1, coarse, mask, 32, beta, "sum")
        ref0 = pyramid_patch_mask(img0, coarse, mask, patch_size=32, beta=beta)
        ref1 = pyramid_patch_mask(img1, coarse, mask, patch_size=32, beta=beta)
        for i in range(3):
            r0, r1 = mask_reverse(ref0[i], ref1[i])
            assert torch.equal(m0[i], r0) and torch.equal(m1[i], r1), f"HCS mismatch at scale {i}, beta {beta}"
            assert torch.equal(union[i], torch.clamp(r0 + r1, max=1)), "HCS union mismatch"
    print("[ok] hcs_patch_masks == DSER++ pyramid_patch_mask + mask_reverse")


def random_batch(batch, bins, height, width, device):
    imgs = torch.rand(batch, 9, height, width, device=device)
    voxels = torch.randn(batch, 3 * bins, height, width, device=device) * 0.3
    mask = (torch.rand(batch, 1, height, width, device=device) > 0.7).float()
    t = torch.rand(batch, device=device) * 0.8 + 0.1
    return imgs, voxels, mask, t


def random_window_batch(batch, num_targets, bins, height, width, device):
    """Batch in the dataset's multi-target layout (B windows, K targets each)."""
    return {
        "endpoints": torch.rand(batch, 6, height, width, device=device),
        "gts": torch.rand(batch, num_targets, 3, height, width, device=device),
        "voxels": torch.randn(batch, num_targets, 3 * bins, height, width, device=device) * 0.3,
        "masks": (torch.rand(batch, num_targets, 1, height, width, device=device) > 0.7).float(),
        "t": torch.arange(1, num_targets + 1, device=device).float().div(num_targets + 1).repeat(batch, 1),
    }


def test_stage(stage, device, num_targets=3, size=128):
    torch.manual_seed(0)
    args = make_args(stage)
    model = MyNet(args).to(device)
    tb.configure_trainable(model, args)
    groups = tb.build_param_groups(model, args)
    model.train()
    batch = random_window_batch(1, num_targets, args.bins, size, size, device)
    imgs, gt, voxels, mask, t, batch_size, k = tb.flatten_targets(batch, device)
    assert imgs.shape == (num_targets, 9, size, size) and k == num_targets
    assert torch.equal(imgs[:, 3:6], gt) and torch.equal(imgs[-1, :3], batch["endpoints"][0, :3])
    outputs = model(imgs, voxels, mask, args.bins, t)
    losses = tb.compute_losses(outputs, imgs, gt, mask, L1Perceptual(), MaskedSSIMLoss(11).to(device), args)
    losses["tc"] = tb.compute_temporal_loss(outputs, gt, mask, batch_size, k, args)
    losses["total"] = losses["total"] + args.lambda_tc * losses["tc"]
    assert num_targets < 2 or float(losses["tc"]) > 0, "temporal loss should be active for K >= 2"
    losses["total"].backward()
    trainable = [p for g in groups for p in g["params"]]
    with_grad = sum(1 for p in trainable if p.grad is not None)
    for p in trainable:
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), "non-finite gradient"
    n = num_targets
    if stage == "rec":
        assert len(outputs) == 2 and outputs[0].shape == (n, 3, size, size)   # K gray reconstructions
    else:
        _, rec, coarse, preds, sel, aux = outputs
        assert coarse.shape == (n, 3, size, size)                              # K predicted frames
        assert [p.shape[-1] for p in preds] == [size, size // 2, size // 4]
        assert sel.shape == (n, 1, size, size)
        assert aux["flow_t0"].shape == (n, 2, size, size)
    msg = ", ".join(f"{name}={float(value):.4f}" for name, value in losses.items())
    print(f"[ok] stage={stage}, K={n}: outputs for {n} targets, params with grad {with_grad}/{len(trainable)} | {msg}")


def _write_synthetic_bsergb(root, num_frames=6, height=64, width=96, events_per_interval=3000, seed=0):
    """Tiny scene in BS-ERGB layout (images/*.png, events/*.npz with raw sensor coordinates)."""
    import cv2
    rng = np.random.default_rng(seed)
    scene = os.path.join(root, "scene_000")
    os.makedirs(os.path.join(scene, "images"))
    os.makedirs(os.path.join(scene, "events"))
    for i in range(num_frames):
        cv2.imwrite(os.path.join(scene, "images", f"{i:06d}.png"),
                    rng.integers(0, 255, (height, width, 3), dtype=np.uint8))
    for i in range(num_frames - 1):
        px = rng.integers(0, width, events_per_interval)
        py = rng.integers(0, height, events_per_interval)
        # inverse of the coordinate scaling in util/event.py load_events(bsergb=True)
        raw_x = (px + 0.5) * 62000.0 / (2 * 970)
        raw_y = (py + 0.5) * 20160.0 / 630
        ts = np.sort(rng.uniform(i, i + 1, events_per_interval))
        np.savez(os.path.join(scene, "events", f"{i:06d}.npz"), x=raw_x, y=raw_y, timestamp=ts,
                 polarity=rng.integers(0, 2, events_per_interval).astype(np.float32))
    return root


def test_dataset_multi_target(bins=8):
    import tempfile
    from torch.utils.data import DataLoader
    from datasets import SkipGroupedBatchSampler, build_bsergb_train_dataset
    with tempfile.TemporaryDirectory() as root:
        _write_synthetic_bsergb(root)
        dataset = build_bsergb_train_dataset(root, bins=bins, skips=(1, 3), crop_size=64, target_mode="all")
        # 6 frames: skip=1 -> 4 windows, skip=3 -> 2 windows
        assert len(dataset) == 6, len(dataset)
        sample = next(dataset[i] for i in range(len(dataset)) if dataset.samples[i][3] == 3)
        assert sample["gts"].shape == (3, 3, 64, 64) and sample["voxels"].shape == (3, 3 * bins, 64, 64)
        assert sample["masks"].shape == (3, 1, 64, 64)
        assert torch.allclose(sample["t"], torch.tensor([0.25, 0.5, 0.75]))
        assert sample["target_indices"].tolist() == [sample["start_index"] + k for k in (1, 2, 3)]
        # different targets -> different event splits
        assert not torch.equal(sample["voxels"][0], sample["voxels"][2])
        sampler = SkipGroupedBatchSampler(dataset, batch_size=2, generator=torch.Generator().manual_seed(0))
        loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
        seen = 0
        for batch in loader:
            k = batch["gts"].shape[1]
            assert all(int(s) == k for s in batch["skip"]), "mixed K inside a batch"
            seen += batch["gts"].shape[0]
        assert seen == len(dataset)
        single = build_bsergb_train_dataset(root, bins=bins, skips=(3,), crop_size=64, target_mode="random")
        assert single[0]["gts"].shape[0] == 1
    print("[ok] dataset: skip=3 window returns 3 targets; batches never mix K")


def test_inference_padding():
    from model.ourModel import Model
    args = make_args("full")
    wrapper = Model(args, inference_only=True)
    net_device = next(wrapper.my_net.parameters()).device
    imgs, voxels, mask, t = random_batch(1, args.bins, 100, 130, net_device)
    pred = wrapper.inference(imgs, voxels, mask, t)
    assert pred.shape == (1, 3, 100, 130), pred.shape
    print("[ok] Model.inference pads/unpads arbitrary sizes")


def test_eval_metrics_and_script():
    import tempfile
    import eval_bsergb as ev
    zeros = torch.zeros(1, 3, 32, 32)
    assert abs(float(ev.psnr(zeros + 0.1, zeros)) - 20.0) < 1e-3, "PSNR formula"
    x = torch.rand(2, 3, 32, 32)
    assert torch.allclose(ev.ssim(x, x), torch.ones(2), atol=1e-6), "SSIM(x, x) must be 1"
    assert float(ev.ssim(x, torch.rand(2, 3, 32, 32)).max()) < 0.5, "SSIM of unrelated images"
    # OF-AF metrics: uint8 truncation + skimage-equivalent PSNR / SSIM
    x8 = ev.to_uint8_truncate(x)
    assert torch.allclose(ev.ssim_skimage(x8, x8), torch.ones(2), atol=1e-6)
    y = (x + 0.05 * torch.randn_like(x)).clamp(0, 1)
    try:
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
        a = x8[0].permute(1, 2, 0).numpy()
        b = ev.to_uint8_truncate(y)[0].permute(1, 2, 0).numpy()
        ref_psnr = peak_signal_noise_ratio(a, b, data_range=255)
        ref_ssim = structural_similarity(a, b, data_range=255, channel_axis=2)
        mine_psnr, mine_ssim, _ = ev.compute_metrics(y[:1], x[:1], "ofaf")
        assert abs(float(mine_psnr) - ref_psnr) < 1e-3 and abs(float(mine_ssim) - ref_ssim) < 1e-5, \
            (float(mine_psnr), ref_psnr, float(mine_ssim), ref_ssim)
        print("[ok] OF-AF PSNR/SSIM match scikit-image exactly")
    except ImportError:
        print("[skip] scikit-image not installed: OF-AF metric cross-check skipped (pip install scikit-image)")
    with tempfile.TemporaryDirectory() as root:
        data_root = os.path.join(root, "test")
        _write_synthetic_bsergb(data_root)                         # 6 frames of 64x96
        args = make_args("full")
        model = MyNet(args)
        ckpt = os.path.join(root, "last.pth")
        torch.save({"net": model.state_dict(), "args": vars(args), "epoch": 0}, ckpt)
        out_dir = os.path.join(root, "eval")
        summary = ev.evaluate(ev.parse_args([
            "--data-root", data_root, "--checkpoint", ckpt, "--device", "cpu", "--workers", "0",
            "--skips", "1", "3", "--output-dir", out_dir, "--save-images"]))
        # OF-AF protocol (default), 6 frames: skip=1 -> starts 0,1,2,3 -> frames 1..4 (4 frames);
        # skip=3 -> start 0 -> frames 1..3 (3 frames)
        assert summary["protocol"] == "ofaf"
        assert summary["skips"]["1"]["frames"] == 4 and summary["skips"]["3"]["frames"] == 3, summary["skips"]
        assert set(summary["skips"]["3"]["by_t"]) == {"0.25", "0.5", "0.75"}
        assert os.path.isfile(os.path.join(out_dir, "per_frame.csv"))
        assert os.path.isfile(os.path.join(out_dir, "images", "skip3", "scene_000", "000002.png"))
        flow_summary = ev.evaluate(ev.parse_args([
            "--data-root", data_root, "--checkpoint", ckpt, "--device", "cpu", "--workers", "0",
            "--skips", "1", "--stage", "flow", "--protocol", "nonoverlap", "--output-dir", out_dir + "_flow"]))
        # non-overlapping windows: skip=1 -> (0,2),(2,4) = 2 frames
        assert flow_summary["stage"] == "flow" and flow_summary["skips"]["1"]["frames"] == 2
    print("[ok] eval_bsergb: metrics, OF-AF / non-overlap window protocols, CSV/PNG outputs, flow-stage scoring")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    device = torch.device(parser.parse_args().device)
    test_backwarp_direction(device)
    test_reverse_voxel()
    test_hcs_equivalence(device)
    test_dataset_multi_target()
    for stage_name in ("rec", "flow", "full"):
        test_stage(stage_name, device, num_targets=3)   # skip=3 window
    test_stage("full", device, num_targets=1)           # skip=1 window
    test_inference_padding()
    test_eval_metrics_and_script()
    print("All smoke tests passed.")
