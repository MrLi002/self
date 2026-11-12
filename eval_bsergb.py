"""Evaluate a checkpoint on the BS-ERGB test split (PSNR / SSIM, optional LPIPS).

Two protocols (``--protocol``):

``ofaf`` (default) reproduces the evaluation code of OF-AF (CVPR 2026, github.com/Sudadaaaa/OF-AF,
bsergb_eval.py / datasets/bsergb.py ``BSERGB_Test``):
  * windows start every ``skip`` frames: (0, skip+1), (skip, 2skip+1), ... so every frame from 1 on
    is scored exactly once (window end points are frames that are targets of the neighbouring window);
  * windows containing a frame of OF-AF's ``skip_inds`` list are dropped (only basket_09 is affected);
  * prediction and GT are converted to uint8 by truncation, ``(x * 255).astype(np.uint8)``;
  * PSNR = skimage ``peak_signal_noise_ratio(gt, pred, data_range=255)``;
    SSIM = skimage ``structural_similarity(gt, pred, data_range=255, channel_axis=2)`` with its
    defaults (7x7 uniform window, sample covariance) -- re-implemented here in torch (identical maths);
  * headline = mean over all scored frames of a skip.
  Not reproduced: FloLPIPS. OF-AF also feeds [0, 1] images to LPIPS without normalize=True; this
  script always uses the standard [-1, 1] LPIPS input, so LPIPS values are not comparable with OF-AF logs.

``nonoverlap``: non-overlapping windows (0, N+1), (N+1, 2N+2), ..., 8-bit rounding, SSIM with an
  11x11 Gaussian window (sigma 1.5). ``all``: every window (stride 1), same metrics as ``nonoverlap``.

Both protocols evaluate at full resolution (zero-padded to a multiple of 64, cropped back).

Examples:
  python eval_bsergb.py --data-root /path/BSERGB/1_TEST --checkpoint train/bsergb_full/last.pth
  python eval_bsergb.py --data-root ... --checkpoint train/bsergb_flow/last.pth   # coarse frame of stage 2
  python eval_bsergb.py --data-root ... --checkpoint ... --lpips --save-images
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from common.size_adapter import ImgAndEventSizeAdapter
from datasets.bsergb import (_event_channels_or_zeros, _list_files, _load_event_chunks,
                             _sequence_from_chunks, _voxel_or_zeros)
from model.ourNet import MyNet
from util.utils_func import process_mask

# OF-AF BSERGB_Test.skip_inds (0-based positions in the sorted image list). Only basket_09 is one of
# the 26 test scenes; the may29_* entries are kept verbatim but never match.
OFAF_SKIP_INDS = {
    "basket_09": [31, 32, 33, 34],
    "may29_rooftop_handheld_02": [17, 70],
    "may29_rooftop_handheld_03": [306],
    "may29_rooftop_handheld_05": [21],
}
PROTOCOLS = ("ofaf", "nonoverlap", "all")

# defaults for checkpoints that do not store their training arguments
MODEL_DEFAULTS = dict(
    bins=8, stage="full", beta=0.1, mask_patch_size=32, hcs_beta_mode="sum", ref_mode="composite",
    rec_use_frames=False, detach_rec_for_flow=False, raft_iters=12, refine_iters=1, texture_window=3,
    no_freeze_raft_bn=False,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate on the BS-ERGB test split")
    parser.add_argument("--data-root", required=True, help="BS-ERGB test root with one folder per scene (e.g. 1_TEST)")
    parser.add_argument("--checkpoint", required=True, help="checkpoint saved by train_bsergb.py")
    parser.add_argument("--skips", nargs="+", type=int, default=[1, 3])
    parser.add_argument("--protocol", default="ofaf", choices=PROTOCOLS,
                        help="ofaf: same windows / exclusions / metrics as OF-AF's bsergb_eval.py (default)")
    parser.add_argument("--scenes", nargs="*", default=None, help="optional scene allow-list")
    parser.add_argument("--exclude-json", default=None,
                        help="extra JSON {scene: [frame indices]}; windows containing these frames are skipped "
                             "(added to OF-AF's list under --protocol ofaf)")
    parser.add_argument("--max-windows-per-scene", type=int, default=None, help="for quick checks")
    parser.add_argument("--stage", default=None, choices=("flow", "full"),
                        help="override the stage stored in the checkpoint (flow -> score the coarse frame)")
    parser.add_argument("--raft-iters", type=int, default=None, help="override RAFT iterations at test time")
    parser.add_argument("--chunk", type=int, default=1, help="targets per forward pass (raise if memory allows)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true", help="fp16 autocast for inference")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers for event loading")
    parser.add_argument("--no-quantize", action="store_true", help="score float predictions (no 8-bit rounding)")
    parser.add_argument("--lpips", action="store_true", help="also report LPIPS (alex)")
    parser.add_argument("--save-images", action="store_true", help="write predicted frames as PNG")
    parser.add_argument("--output-dir", default=None, help="default: <checkpoint dir>/eval_bsergb")
    return parser.parse_args(argv)


# ----------------------------------------------------------------------------- metrics
def quantize(image: torch.Tensor) -> torch.Tensor:
    return torch.round(image.clamp(0.0, 1.0) * 255.0) / 255.0


def psnr(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Per-image PSNR (B,) for images in [0, 1]."""
    mse = ((pred - gt) ** 2).flatten(1).mean(1)
    return 10.0 * torch.log10(1.0 / mse.clamp_min(1e-10))


def _gaussian_window(channels: int, size: int = 11, sigma: float = 1.5, device=None, dtype=torch.float32):
    coords = torch.arange(size, device=device, dtype=dtype) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).expand(channels, 1, size, size).contiguous()


def ssim(pred: torch.Tensor, gt: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Per-image SSIM (B,): Gaussian window, 'valid' filtering, averaged over channels and pixels."""
    channels = pred.shape[1]
    pred, gt = pred.double(), gt.double()
    window = _gaussian_window(channels, window_size, sigma, pred.device, pred.dtype)
    filt = lambda x: F.conv2d(x, window, groups=channels)  # noqa: E731
    mu1, mu2 = filt(pred), filt(gt)
    sigma1 = filt(pred * pred) - mu1 ** 2
    sigma2 = filt(gt * gt) - mu2 ** 2
    sigma12 = filt(pred * gt) - mu1 * mu2
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2))
    return ssim_map.flatten(1).mean(1).float()


def to_uint8_truncate(image: torch.Tensor) -> torch.Tensor:
    """OF-AF: (x * 255).astype(np.uint8) on float32 -> truncation toward zero."""
    return (image.float().clamp(0.0, 1.0) * 255.0).to(torch.uint8)


def psnr_skimage(pred8: torch.Tensor, gt8: torch.Tensor) -> torch.Tensor:
    """skimage.metrics.peak_signal_noise_ratio(gt, pred, data_range=255) per image, uint8 inputs."""
    mse = ((pred8.double() - gt8.double()) ** 2).flatten(1).mean(1)
    return (10.0 * torch.log10(255.0 ** 2 / mse.clamp_min(1e-10))).float()


def ssim_skimage(pred8: torch.Tensor, gt8: torch.Tensor, win_size: int = 7) -> torch.Tensor:
    """skimage.metrics.structural_similarity(gt, pred, data_range=255, channel_axis=2) per image.

    skimage defaults: uniform win_size x win_size filter, use_sample_covariance=True, K1=0.01, K2=0.03;
    the SSIM map is cropped by (win_size-1)//2 on each side (== 'valid' filtering) and averaged,
    then averaged over channels.
    """
    x, y = pred8.double(), gt8.double()
    channels = x.shape[1]
    kernel = torch.full((channels, 1, win_size, win_size), 1.0 / win_size ** 2, dtype=x.dtype, device=x.device)
    filt = lambda img: F.conv2d(img, kernel, groups=channels)  # noqa: E731
    ux, uy = filt(x), filt(y)
    uxx, uyy, uxy = filt(x * x), filt(y * y), filt(x * y)
    cov_norm = win_size ** 2 / (win_size ** 2 - 1.0)
    vx, vy, vxy = cov_norm * (uxx - ux * ux), cov_norm * (uyy - uy * uy), cov_norm * (uxy - ux * uy)
    c1, c2 = (0.01 * 255.0) ** 2, (0.03 * 255.0) ** 2
    s_map = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux ** 2 + uy ** 2 + c1) * (vx + vy + c2))
    return s_map.flatten(2).mean(2).mean(1).float()


def compute_metrics(pred: torch.Tensor, gt: torch.Tensor, protocol: str, quantize_output: bool = True):
    """Returns (psnr, ssim, scored_pred) per image for a batch (B, 3, H, W) in [0, 1]."""
    if protocol == "ofaf":
        if not quantize_output:
            return psnr(pred, gt), ssim_skimage(pred * 255.0, gt * 255.0), pred
        pred8, gt8 = to_uint8_truncate(pred), to_uint8_truncate(gt)
        return psnr_skimage(pred8, gt8), ssim_skimage(pred8, gt8), pred8.float() / 255.0
    scored = quantize(pred) if quantize_output else pred
    return psnr(scored, gt), ssim(scored, gt), scored


# ----------------------------------------------------------------------------- data
def _first(batch):
    """collate_fn: one window per item, no stacking (top-level so worker processes can pickle it)."""
    return batch[0]


class BSERGBEvalWindows(Dataset):
    """Full-resolution test windows; each item holds all K = skip targets of one window."""

    def __init__(self, data_root, skips, bins, protocol="ofaf", scenes=None, excluded=None,
                 max_windows_per_scene=None):
        self.root = Path(data_root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"BS-ERGB test root does not exist: {self.root}")
        self.bins = int(bins)
        names = list(scenes) if scenes else sorted(p.name for p in self.root.iterdir() if p.is_dir())
        merged = {k: set(v) for k, v in (OFAF_SKIP_INDS.items() if protocol == "ofaf" else [])}
        for key, value in (excluded or {}).items():
            merged.setdefault(key, set()).update(value)
        excluded = merged
        self.dropped_windows = 0
        self.scenes, self.items = [], []
        for name in names:
            images = _list_files(self.root / name / "images", (".png", ".jpg", ".jpeg"))
            events = _list_files(self.root / name / "events", (".npz",))
            if len(images) < 3 or len(events) < len(images) - 1:
                print(f"[warn] skip scene {name}: {len(images)} images, {len(events)} event files")
                continue
            scene_id = len(self.scenes)
            self.scenes.append({"name": name, "images": images, "events": events})
            for skip in skips:
                offset = skip + 1
                # ofaf: range(0, n - skip - 1, skip)  (OF-AF BSERGB_Test.test)
                step = {"ofaf": skip, "nonoverlap": offset, "all": 1}[protocol]
                count = 0
                for start in range(0, len(images) - offset, step):
                    end = start + offset
                    if any(f in excluded.get(name, ()) for f in range(start, end + 1)):
                        self.dropped_windows += 1
                        continue
                    if max_windows_per_scene is not None and count >= max_windows_per_scene:
                        break
                    self.items.append((scene_id, start, end, skip))
                    count += 1
        if not self.items:
            raise RuntimeError("No evaluation windows found; check --data-root / --skips / --scenes")

    def __len__(self):
        return len(self.items)

    @staticmethod
    def _read(path):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {path}")
        return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0

    def __getitem__(self, index):
        scene_id, start, end, skip = self.items[index]
        scene = self.scenes[scene_id]
        img0, img1 = self._read(scene["images"][start]), self._read(scene["images"][end])
        height, width = img0.shape[-2:]
        targets = list(range(start + 1, end))
        gts = torch.stack([self._read(scene["images"][k]) for k in targets])
        chunks = _load_event_chunks(scene["events"][start:end], (0, 0, height, width))
        voxels, masks = [], []
        for target in targets:
            split = target - start
            ev_0t = _sequence_from_chunks(chunks[:split], height, width)
            ev_t1 = _sequence_from_chunks(chunks[split:], height, width)
            counts = torch.cat((_event_channels_or_zeros(ev_0t, height, width),
                                _event_channels_or_zeros(ev_t1, height, width)), 0)
            masks.append(process_mask(counts).float())
            ev_1t = None if ev_t1 is None else ev_t1.copy().reverse()
            voxels.append(torch.cat((_voxel_or_zeros(ev_0t, self.bins, height, width),
                                     _voxel_or_zeros(ev_t1, self.bins, height, width),
                                     _voxel_or_zeros(ev_1t, self.bins, height, width)), 0).float())
        return {
            "img0": img0, "img1": img1, "gts": gts,
            "voxels": torch.stack(voxels), "masks": torch.stack(masks),
            "t": torch.tensor([(k - start) / float(end - start) for k in targets]),
            "scene": scene["name"], "start": start, "end": end, "skip": skip, "targets": targets,
        }


# ----------------------------------------------------------------------------- model
def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_model(checkpoint_path: str, device: torch.device, stage: Optional[str] = None,
               raft_iters: Optional[int] = None):
    checkpoint = _torch_load(checkpoint_path)
    saved_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    config = dict(MODEL_DEFAULTS)
    config.update({k: v for k, v in saved_args.items() if k in MODEL_DEFAULTS})
    if stage is not None:
        config["stage"] = stage
    if raft_iters is not None:
        config["raft_iters"] = raft_iters
    if config["stage"] == "rec":
        raise ValueError("This checkpoint is from the rec stage (no RGB prediction). "
                         "Evaluate a flow/full checkpoint, or pass --stage flow/full explicitly.")
    model_args = argparse.Namespace(**config)

    state = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("net", "model", "state_dict"):
            if isinstance(checkpoint.get(key), dict):
                state = checkpoint[key]
                break
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items() if torch.is_tensor(v)}
    model = MyNet(model_args)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, model_args, checkpoint.get("epoch") if isinstance(checkpoint, dict) else None


@torch.no_grad()
def predict_window(model, sample, device, bins, chunk=1, amp=False):
    """Predict all K targets of one window. Returns (K, 3, H, W) in [0, 1] on ``device``."""
    num_targets = sample["voxels"].shape[0]
    img0 = sample["img0"].to(device)[None].expand(num_targets, -1, -1, -1)
    img1 = sample["img1"].to(device)[None].expand(num_targets, -1, -1, -1)
    imgs = torch.cat((img0, torch.zeros_like(img0), img1), dim=1)          # GT slot is never read
    voxels = sample["voxels"].to(device)
    masks = sample["masks"].to(device)
    times = sample["t"].to(device)
    adapter = ImgAndEventSizeAdapter(minimum_size=64)
    imgs, voxels, masks = adapter.pad(imgs, voxels, masks)
    outputs = []
    use_amp = amp and device.type == "cuda"
    for begin in range(0, num_targets, max(1, chunk)):
        sl = slice(begin, begin + max(1, chunk))
        with torch.amp.autocast("cuda", enabled=use_amp):
            result = model(imgs[sl], voxels[sl], masks[sl], bins, times[sl])
        outputs.append(adapter.unpad(result[3][0]).float().clamp(0.0, 1.0))
    return torch.cat(outputs, dim=0)


# ----------------------------------------------------------------------------- main loop
def evaluate(args: argparse.Namespace) -> Dict[str, object]:
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    model, model_args, epoch = load_model(args.checkpoint, device, args.stage, args.raft_iters)
    excluded = json.loads(Path(args.exclude_json).read_text()) if args.exclude_json else None
    dataset = BSERGBEvalWindows(args.data_root, args.skips, model_args.bins, args.protocol,
                                args.scenes, excluded, args.max_windows_per_scene)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers,
                        collate_fn=_first)
    out_dir = Path(args.output_dir or Path(args.checkpoint).resolve().parent / "eval_bsergb")
    out_dir.mkdir(parents=True, exist_ok=True)

    lpips_metric = None
    if args.lpips:
        import lpips
        lpips_metric = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    print(f"Checkpoint: {args.checkpoint} (stage={model_args.stage}, epoch={epoch}) | device={device} | "
          f"windows={len(dataset)} (dropped {dataset.dropped_windows}) | scenes={len(dataset.scenes)} | "
          f"skips={args.skips} | protocol={args.protocol}")
    rows: List[Dict[str, object]] = []
    gpu_time, start_time = 0.0, time.time()
    for index, sample in enumerate(loader):
        t0 = time.time()
        preds = predict_window(model, sample, device, model_args.bins, args.chunk, args.amp)
        if device.type == "cuda":
            torch.cuda.synchronize()
        gpu_time += time.time() - t0
        gts = sample["gts"].to(device)
        frame_psnr, frame_ssim, scored = compute_metrics(preds, gts, args.protocol, not args.no_quantize)
        frame_lpips = None
        if lpips_metric is not None:
            frame_lpips = lpips_metric(scored[:, [2, 1, 0]] * 2 - 1, gts[:, [2, 1, 0]] * 2 - 1).flatten()
        for k, target in enumerate(sample["targets"]):
            row = {"scene": sample["scene"], "skip": sample["skip"], "start": sample["start"],
                   "end": sample["end"], "target": target, "t": round(float(sample["t"][k]), 4),
                   "psnr": float(frame_psnr[k]), "ssim": float(frame_ssim[k])}
            if frame_lpips is not None:
                row["lpips"] = float(frame_lpips[k])
            rows.append(row)
            if args.save_images:
                path = out_dir / "images" / f"skip{sample['skip']}" / sample["scene"] / f"{target:06d}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                image = (preds[k].clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
                cv2.imwrite(str(path), image)
        if (index + 1) % 20 == 0 or index + 1 == len(dataset):
            done = [r for r in rows if r["skip"] == sample["skip"]]
            print(f"  [{index + 1}/{len(dataset)}] skip={sample['skip']} running PSNR "
                  f"{np.mean([r['psnr'] for r in done]):.3f} SSIM {np.mean([r['ssim'] for r in done]):.4f}")

    summary = summarize(rows, lpips_metric is not None)
    summary.update({"checkpoint": str(args.checkpoint), "stage": model_args.stage, "epoch": epoch,
                    "protocol": args.protocol, "quantized": not args.no_quantize,
                    "seconds_per_frame_model": gpu_time / max(1, len(rows)),
                    "seconds_total": time.time() - start_time})
    write_outputs(out_dir, rows, summary)
    print_summary(summary)
    print(f"Per-frame results: {out_dir / 'per_frame.csv'} | summary: {out_dir / 'summary.json'}")
    return summary


def summarize(rows, with_lpips):
    metrics = ["psnr", "ssim"] + (["lpips"] if with_lpips else [])

    def mean_of(subset):
        return {m: float(np.mean([r[m] for r in subset])) for m in metrics}

    by_skip = defaultdict(list)
    for row in rows:
        by_skip[row["skip"]].append(row)
    summary = {"metrics": metrics, "skips": {}}
    for skip, subset in sorted(by_skip.items()):
        by_t, by_scene = defaultdict(list), defaultdict(list)
        for row in subset:
            by_t[row["t"]].append(row)
            by_scene[row["scene"]].append(row)
        scene_means = [mean_of(v) for v in by_scene.values()]
        summary["skips"][str(skip)] = {
            "frames": len(subset),
            "windows": len({(r["scene"], r["start"]) for r in subset}),
            "frame_mean": mean_of(subset),                                   # main number
            "scene_mean": {m: float(np.mean([s[m] for s in scene_means])) for m in metrics},
            "by_t": {str(t): dict(mean_of(v), frames=len(v)) for t, v in sorted(by_t.items())},
            "by_scene": {name: dict(mean_of(v), frames=len(v)) for name, v in sorted(by_scene.items())},
        }
    return summary


def write_outputs(out_dir, rows, summary):
    with open(out_dir / "per_frame.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(out_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def print_summary(summary):
    metrics = summary["metrics"]
    print("\n================ BS-ERGB results ================")
    print(f"stage={summary['stage']} epoch={summary['epoch']} quantized={summary['quantized']} "
          f"protocol={summary['protocol']}")
    header = f"{'skip':>4} {'frames':>7} " + " ".join(f"{m.upper():>8}" for m in metrics)
    print(header)
    for skip, info in summary["skips"].items():
        values = " ".join(f"{info['frame_mean'][m]:8.4f}" for m in metrics)
        print(f"{skip:>4} {info['frames']:>7} {values}")
        for t, info_t in info["by_t"].items():
            values_t = " ".join(f"{info_t[m]:8.4f}" for m in metrics)
            print(f"{'':>4} {'t=' + t:>7} {values_t}")
    print(f"model time per frame: {summary['seconds_per_frame_model']:.3f} s "
          f"(total {summary['seconds_total'] / 60:.1f} min)")


if __name__ == "__main__":
    evaluate(parse_args())
