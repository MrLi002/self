"""Train the event-reference guided flow interpolation network on BS-ERGB.

Recommended three-stage schedule (each stage can start from the previous checkpoint):
  1) --stage rec   : train only the E2V network (EventRecNet) with the masked gray GT;
  2) --stage flow  : load stage-1 weights + official RAFT weights, train coarse flow,
                     event refinement and selective fusion (add --freeze-rec);
  3) --stage full  : train everything end-to-end incl. texture mapping and refinement.
"""

from __future__ import annotations

import argparse
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import TARGET_MODES, SkipGroupedBatchSampler, build_bsergb_train_dataset
from model.loss import MaskedSSIMLoss
from model.ourNet import STAGES, MyNet
from model.warp import backwarp, edge_aware_smoothness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or fine-tune DSER++ on BS-ERGB"
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="BS-ERGB training root containing one directory per scene",
    )
    parser.add_argument("--output-dir", default="train/bsergb", help="Checkpoint directory")
    parser.add_argument("--resume", default=None, help="Resume a full training checkpoint")
    parser.add_argument(
        "--pretrained",
        default=None,
        help="Load network weights only before starting a new optimizer",
    )
    parser.add_argument(
        "--allow-partial-pretrained",
        action="store_true",
        help="Allow missing or unexpected keys when loading --pretrained",
    )
    parser.add_argument("--scenes", nargs="*", default=None, help="Optional scene allow-list")

    parser.add_argument("--bins", default=8, type=int)
    parser.add_argument("--skips", nargs="+", default=[1, 3], type=int)
    parser.add_argument("--crop-size", default=256, type=int)
    parser.add_argument("--beta", default=0.1, type=float)
    parser.add_argument("--mask-patch-size", default=32, type=int)
    parser.add_argument("--no-augmentation", action="store_true")

    parser.add_argument("--epochs", default=40, type=int)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--min-lr", default=1e-6, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--grad-clip", default=1.0, type=float)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--log-interval", default=20, type=int)
    parser.add_argument("--save-every", default=1, type=int)
    parser.add_argument("--no-save", action="store_true",
                        help="Do not write checkpoints (e.g. for memory / speed tests)")
    parser.add_argument(
        "--max-steps-per-epoch",
        default=None,
        type=int,
        help="Optional limit useful for smoke tests",
    )

    parser.add_argument("--lambda-rec", default=1.0, type=float)
    parser.add_argument("--lambda-syn", default=1.0, type=float)
    parser.add_argument("--lambda-ref", default=1.0, type=float)
    parser.add_argument("--lambda-ps", default=1.0, type=float)
    parser.add_argument("--target-mode", default="all", choices=TARGET_MODES,
                        help="'all': a skip-N window predicts all N intermediate frames (K=N); "
                             "'random': one random target per window (K=1, old behaviour)")
    parser.add_argument("--target-weighting", default="window", choices=("window", "frame"),
                        help="'window': every window has the same weight (losses averaged over its K "
                             "targets); 'frame': every target frame has the same weight")
    parser.add_argument("--lambda-tc", default=0.0, type=float,
                        help="Temporal consistency between adjacent targets of a window: "
                             "|(P_k+1 - P_k) - (G_k+1 - G_k)| (needs K >= 2)")
    parser.add_argument("--lambda-warp", default=0.5, type=float,
                        help="Photometric loss on the coarse (RAFT) and refined flows")
    parser.add_argument("--lambda-smooth", default=0.0, type=float,
                        help="Edge-aware smoothness of the refined flows (0 disables)")
    parser.add_argument("--warp-gamma", default=0.8, type=float,
                        help="RAFT-style weight decay over the coarse-flow iterations")

    # ---- network / stage options -------------------------------------------------
    parser.add_argument("--stage", default="full", choices=STAGES)
    parser.add_argument("--raft-ckpt", default=None,
                        help="Official RAFT weights (e.g. raft-things.pth) loaded into the coarse flow net")
    parser.add_argument("--raft-iters", default=12, type=int)
    parser.add_argument("--refine-iters", default=1, type=int)
    parser.add_argument("--texture-window", default=3, type=int)
    parser.add_argument("--ref-mode", default="composite", choices=("composite", "rec"),
                        help="Reference for the cost volume: E2V inside the event mask, key-frame mean outside")
    parser.add_argument("--rec-use-frames", action="store_true",
                        help="Also feed gray I0/I1 to the E2V net (not compatible with DSER++ EventRecNet weights)")
    parser.add_argument("--detach-rec-for-flow", action="store_true",
                        help="Stop flow/synthesis gradients from reaching the E2V net")
    parser.add_argument("--hcs-beta-mode", default="sum", choices=("sum", "mean"),
                        help="'sum' reproduces DSER++ (patch sum >= beta); 'mean' thresholds the patch mean")
    parser.add_argument("--no-freeze-raft-bn", action="store_true")
    parser.add_argument("--freeze-rec", action="store_true", help="Do not update the E2V network")
    parser.add_argument("--freeze-raft", action="store_true", help="Do not update RAFT")
    parser.add_argument("--raft-lr-mult", default=0.25, type=float)
    parser.add_argument("--rec-lr-mult", default=1.0, type=float)
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision")
    parser.add_argument(
        "--data-parallel",
        action="store_true",
        help="Use every visible CUDA device through DataParallel",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Training device, for example cuda, cuda:0, or cpu",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _extract_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must contain a state dictionary")
    for key in ("net", "model", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            checkpoint = value
            break
    if not isinstance(checkpoint, dict):
        raise TypeError("Could not find a network state dictionary in checkpoint")
    state_dict: Dict[str, torch.Tensor] = {}
    for key, value in checkpoint.items():
        if not isinstance(value, torch.Tensor):
            continue
        if key.startswith("module."):
            key = key[7:]
        key = key.replace(
            ".transformer_block.to_patch_embedding2.0.",
            ".transformer_block.to_patch_embedding.1.",
        )
        state_dict[key] = value
    return state_dict


def _torch_load(path: str) -> object:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_network_weights(model: torch.nn.Module, path: str, strict: bool,) -> None:
    checkpoint = _torch_load(path)
    state_dict = _extract_state_dict(checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=strict)
    if not strict:
        print(f"Missing keys: {list(incompatible.missing_keys)}")
        print(f"Unexpected keys: {list(incompatible.unexpected_keys)}")


def resume_training(model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: CosineAnnealingLR, scaler, path: str) -> Tuple[int, int]:
    checkpoint = _torch_load(path)
    if not isinstance(checkpoint, dict):
        raise TypeError("Resume checkpoint must be a dictionary")
    model.load_state_dict(_extract_state_dict(checkpoint), strict=True)
    if "optimizer" not in checkpoint or "scheduler" not in checkpoint:
        raise KeyError("Resume checkpoint must contain optimizer and scheduler states")
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    return int(checkpoint.get("epoch", -1)) + 1, int(checkpoint.get("global_step", 0))


def bgr_to_gray(image: torch.Tensor) -> torch.Tensor:
    weights = image.new_tensor((0.114, 0.587, 0.299)).view(1, 3, 1, 1)
    return (image * weights).sum(dim=1, keepdim=True)


def lpips_distance(metric: torch.nn.Module, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Images are OpenCV BGR; LPIPS was trained on RGB, so swap the channels first.
    return metric(pred[:, [2, 1, 0]] * 2.0 - 1.0, target[:, [2, 1, 0]] * 2.0 - 1.0).mean()


def multiscale_l1(predictions: Iterable[torch.Tensor], gt: torch.Tensor) -> torch.Tensor:
    weights = (1.0, 0.5, 0.1)
    loss = gt.new_zeros(())
    for weight, prediction in zip(weights, predictions):
        target = F.interpolate(gt, size=prediction.shape[-2:], mode="bilinear", align_corners=False)
        loss = loss + weight * F.l1_loss(prediction, target)
    return loss


def min_photometric(warp0: torch.Tensor, warp1: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Occlusion-tolerant photometric loss: every pixel only needs one good source frame."""
    err0 = (warp0 - gt).abs().mean(1, keepdim=True)
    err1 = (warp1 - gt).abs().mean(1, keepdim=True)
    return torch.minimum(err0, err1).mean() + 0.1 * (err0.mean() + err1.mean())


def sequence_warp_loss(img0, img1, gt, flows_t0, flows_t1, gamma: float) -> torch.Tensor:
    """RAFT-style weighted photometric loss over all coarse-flow iterations."""
    count = len(flows_t0)
    loss = gt.new_zeros(())
    for index, (flow_t0, flow_t1) in enumerate(zip(flows_t0, flows_t1)):
        weight = gamma ** (count - index - 1)
        loss = loss + weight * min_photometric(backwarp(img0, flow_t0), backwarp(img1, flow_t1), gt)
    return loss


def compute_losses(outputs: Tuple[object, ...], imgs: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor,
                   perceptual: torch.nn.Module, structural_loss: MaskedSSIMLoss,
                   args: argparse.Namespace) -> Dict[str, torch.Tensor]:
    zero = gt.new_zeros(())
    masked_reference = outputs[1]
    gray_gt = (bgr_to_gray(gt) * mask).repeat(1, 3, 1, 1)
    loss_rec = F.l1_loss(masked_reference, gray_gt) + lpips_distance(perceptual, masked_reference, gray_gt)
    losses = {"rec": loss_rec, "syn": zero, "warp": zero, "smooth": zero, "ref": zero, "ps": zero}
    total = args.lambda_rec * loss_rec

    if args.stage != "rec":
        _, _, coarse, predictions, selected_mask, aux = outputs
        img0, img1 = imgs[:, :3], imgs[:, 6:9]
        losses["syn"] = F.l1_loss(coarse, gt) + lpips_distance(perceptual, coarse, gt)
        losses["warp"] = (sequence_warp_loss(img0, img1, gt, aux["flows_t0"], aux["flows_t1"], args.warp_gamma)
                          + min_photometric(aux["warp0"], aux["warp1"], gt))
        total = total + args.lambda_syn * losses["syn"] + args.lambda_warp * losses["warp"]
        if args.lambda_smooth > 0:
            losses["smooth"] = (edge_aware_smoothness(aux["flow_t0"], gt)
                                + edge_aware_smoothness(aux["flow_t1"], gt))
            total = total + args.lambda_smooth * losses["smooth"]
        if args.stage == "full":
            final_prediction = predictions[0]
            losses["ref"] = multiscale_l1(predictions, gt) + lpips_distance(perceptual, final_prediction, gt)
            losses["ps"] = structural_loss(final_prediction, gt, selected_mask)
            total = total + args.lambda_ref * losses["ref"] + args.lambda_ps * losses["ps"]

    losses["total"] = total
    return losses


def flatten_targets(batch: Dict[str, object], device: torch.device):
    """(B, K, ...) window batch -> (B*K, ...) per-target batch for MyNet.

    imgs keeps the original 9-channel layout [I_0, GT_t, I_1] so MyNet / compute_losses
    are unchanged; targets of one window stay contiguous and in ascending time order.
    """
    endpoints = batch["endpoints"].to(device, non_blocking=True)   # (B, 6, H, W)
    gts = batch["gts"].to(device, non_blocking=True)               # (B, K, 3, H, W)
    voxels = batch["voxels"].to(device, non_blocking=True)         # (B, K, 3*bins, H, W)
    masks = batch["masks"].to(device, non_blocking=True)           # (B, K, 1, H, W)
    times = batch["t"].to(device, non_blocking=True)               # (B, K)
    batch_size, num_targets, _, height, width = gts.shape
    img0 = endpoints[:, None, :3].expand(batch_size, num_targets, 3, height, width)
    img1 = endpoints[:, None, 3:6].expand(batch_size, num_targets, 3, height, width)
    imgs = torch.cat((img0, gts, img1), dim=2).reshape(batch_size * num_targets, 9, height, width)
    return (imgs,
            gts.reshape(batch_size * num_targets, 3, height, width),
            voxels.reshape(batch_size * num_targets, voxels.shape[2], height, width),
            masks.reshape(batch_size * num_targets, 1, height, width),
            times.reshape(batch_size * num_targets),
            batch_size, num_targets)


def temporal_consistency_loss(pred: torch.Tensor, target: torch.Tensor, batch_size: int, num_targets: int,
                              mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """|(P_{k+1} - P_k) - (G_{k+1} - G_k)| over adjacent targets of the same window.

    Flow-free, so it is valid for any motion; it penalises errors that change from one
    target time to the next (flicker). ``mask`` restricts it to pixels valid in both frames.
    """
    if num_targets < 2:
        return pred.new_zeros(())
    shape = (batch_size, num_targets) + tuple(pred.shape[1:])
    pred = pred.reshape(shape)
    target = target.reshape(shape)
    diff = ((pred[:, 1:] - pred[:, :-1]) - (target[:, 1:] - target[:, :-1])).abs()
    if mask is None:
        return diff.mean()
    mask = mask.reshape((batch_size, num_targets) + tuple(mask.shape[1:]))
    both = (mask[:, 1:] * mask[:, :-1]).expand_as(diff)
    return (diff * both).sum() / both.sum().clamp_min(1.0)


def compute_temporal_loss(outputs: Tuple[object, ...], gt: torch.Tensor, mask: torch.Tensor,
                          batch_size: int, num_targets: int, args: argparse.Namespace) -> torch.Tensor:
    if args.stage == "rec":
        gray_gt = (bgr_to_gray(gt) * mask).repeat(1, 3, 1, 1)
        return temporal_consistency_loss(outputs[1], gray_gt, batch_size, num_targets, mask)
    if args.stage == "flow":
        return temporal_consistency_loss(outputs[2], gt, batch_size, num_targets)
    return temporal_consistency_loss(outputs[3][0], gt, batch_size, num_targets)


def configure_trainable(model: MyNet, args: argparse.Namespace) -> None:
    """Freeze the sub-networks that the current stage does not train."""
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    frozen = []
    if args.stage == "rec":
        frozen += ["coarse_flow", "refiner", "fusion", "texture", "decoder"]
    elif args.stage == "flow":
        frozen += ["texture", "decoder"]
    if args.freeze_rec:
        frozen.append("event_rec")
    if args.freeze_raft:
        frozen.append("coarse_flow")
    for name in set(frozen):
        for parameter in getattr(model, name).parameters():
            parameter.requires_grad_(False)
    if args.stage == "rec" and args.freeze_rec:
        raise ValueError("--freeze-rec leaves nothing to train in the rec stage")


def build_param_groups(model: MyNet, args: argparse.Namespace):
    groups = {"other": [], "raft": [], "rec": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("coarse_flow."):
            groups["raft"].append(parameter)
        elif name.startswith("event_rec."):
            groups["rec"].append(parameter)
        else:
            groups["other"].append(parameter)
    multipliers = {"other": 1.0, "raft": args.raft_lr_mult, "rec": args.rec_lr_mult}
    param_groups = [
        {"params": params, "lr": args.lr * multipliers[key], "name": key}
        for key, params in groups.items() if params
    ]
    if not param_groups:
        raise RuntimeError("No trainable parameters for this stage")
    return param_groups


def gpu_memory_gib(device: torch.device) -> Tuple[float, float]:
    """Peak allocated / reserved CUDA memory since the last reset, in GiB (0 on CPU)."""
    if device.type != "cuda":
        return 0.0, 0.0
    return (torch.cuda.max_memory_allocated(device) / 2 ** 30,
            torch.cuda.max_memory_reserved(device) / 2 ** 30)


def save_checkpoint(
    output_dir: Path,
    filename: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    scaler,
    epoch: int,
    global_step: int,
    args: argparse.Namespace,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    network = model.module if isinstance(model, torch.nn.DataParallel) else model
    checkpoint = {
        "net": network.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "args": vars(args),
    }
    destination = output_dir / filename
    temporary = output_dir / f".{filename}.tmp"
    torch.save(checkpoint, temporary)
    os.replace(temporary, destination)
    return destination


def train(args: argparse.Namespace) -> None:
    if args.resume and args.pretrained:
        raise ValueError("--resume and --pretrained are mutually exclusive")
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")
    if args.workers < 0:
        raise ValueError("workers cannot be negative")
    if args.log_interval <= 0 or args.save_every <= 0:
        raise ValueError("log-interval and save-every must be positive")
    if args.max_steps_per_epoch is not None and args.max_steps_per_epoch <= 0:
        raise ValueError("max-steps-per-epoch must be positive")
    if args.crop_size <= 0 or args.mask_patch_size <= 0:
        raise ValueError("crop-size and mask-patch-size must be positive")
    if args.crop_size % 32 != 0:
        raise ValueError("crop-size must be divisible by 32")
    if args.crop_size % args.mask_patch_size != 0:
        raise ValueError("crop-size must be divisible by mask-patch-size")
    if args.mask_patch_size % 4 != 0:
        raise ValueError("mask-patch-size must be divisible by 4")
    if args.resume and args.raft_ckpt:
        raise ValueError("--raft-ckpt is ignored when resuming; remove it")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")
    if args.amp and device.type != "cuda":
        raise ValueError("--amp requires a CUDA device")

    try:
        import lpips
    except ImportError as error:
        raise ImportError(
            "Training requires lpips; install the project dependencies with "
            "`pip install -r requirements.txt`."
        ) from error

    seed_everything(args.seed)
    dataset = build_bsergb_train_dataset(
        data_root=args.data_root,
        bins=args.bins,
        skips=args.skips,
        crop_size=args.crop_size,
        scenes=args.scenes,
        augment=not args.no_augmentation,
        target_mode=args.target_mode,
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    # Batches only contain windows with the same number of targets K (skip=1 -> 1, skip=3 -> 3).
    batch_sampler = SkipGroupedBatchSampler(dataset, args.batch_size, drop_last=False, generator=generator)
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    mean_targets = dataset.mean_targets
    print(f"Target mode: {args.target_mode} | windows per K: "
          f"{ {k: len(v) for k, v in sorted(batch_sampler.groups.items())} } | mean K: {mean_targets:.2f}")
    if len(dataloader) == 0:
        raise RuntimeError("The BS-ERGB DataLoader is empty")

    model = MyNet(args).to(device)
    if not args.resume:
        if args.pretrained:
            load_network_weights(model, args.pretrained, strict=not args.allow_partial_pretrained)
            print(f"Loaded pretrained weights from {args.pretrained}")
        if args.raft_ckpt:
            # Loaded after --pretrained so that a stage-1 checkpoint does not overwrite RAFT.
            model.coarse_flow.load_pretrained(args.raft_ckpt)
            print(f"Loaded RAFT weights from {args.raft_ckpt}")
    configure_trainable(model, args)
    optimizer = AdamW(build_param_groups(model, args), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(dataloader)
    if args.max_steps_per_epoch is not None:
        steps_per_epoch = min(steps_per_epoch, args.max_steps_per_epoch)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs * steps_per_epoch),
        eta_min=args.min_lr,
    )
    # torch.amp.* replaces the deprecated torch.cuda.amp.* (same behaviour, no FutureWarning)
    scaler = (torch.amp.GradScaler("cuda", enabled=args.amp) if hasattr(torch.amp, "GradScaler")
              else torch.cuda.amp.GradScaler(enabled=args.amp))

    start_epoch = 0
    global_step = 0
    if args.resume:
        start_epoch, global_step = resume_training(model, optimizer, scheduler, scaler, args.resume)
        print(f"Resumed {args.resume} at epoch {start_epoch}")

    if args.data_parallel:
        if device.type != "cuda" or torch.cuda.device_count() < 2:
            raise RuntimeError("--data-parallel requires at least two visible CUDA devices")
        model = torch.nn.DataParallel(model)

    perceptual = lpips.LPIPS(net="alex").to(device).eval()
    for parameter in perceptual.parameters():
        parameter.requires_grad_(False)
    structural_loss = MaskedSSIMLoss(window_size=11).to(device)

    print(
        f"Stage: {args.stage} | training samples: {len(dataset)} | batches: {len(dataloader)} | "
        f"parameters: {sum(p.numel() for p in model.parameters()):,} | "
        f"trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )
    output_dir = Path(args.output_dir)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running = {"total": 0.0, "rec": 0.0, "syn": 0.0, "warp": 0.0, "smooth": 0.0, "ref": 0.0, "ps": 0.0,
                   "tc": 0.0}
        processed_frames = 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epoch_start_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        progress = tqdm(dataloader, desc=f"{epoch_start_time} Epoch {epoch + 1}/{args.epochs}")
        processed_steps = 0

        for step, batch in enumerate(progress):
            if args.max_steps_per_epoch is not None and step >= args.max_steps_per_epoch:
                break
            # (B, K, ...) -> (B*K, ...): skip=3 windows run all 3 targets in one forward pass
            imgs, gt, voxels, mask, t, batch_size, num_targets = flatten_targets(batch, device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp):
                outputs = model(imgs, voxels, mask, args.bins, t)
                # per-target losses are means over all B*K targets, i.e. averaged per window
                losses = compute_losses(outputs, imgs, gt, mask, perceptual, structural_loss, args)
                losses["tc"] = compute_temporal_loss(outputs, gt, mask, batch_size, num_targets, args)
                losses["total"] = losses["total"] + args.lambda_tc * losses["tc"]
                if args.target_weighting == "frame":
                    losses["total"] = losses["total"] * (num_targets / mean_targets)

            scaler.scale(losses["total"]).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1
            processed_steps += 1
            processed_frames += batch_size * num_targets
            for name in running:
                running[name] += float(losses[name].detach())

            if global_step % args.log_interval == 0:
                shown = ("total", "rec", "syn", "warp", "ref", "ps", "tc")
                postfix = {name: f"{running[name] / processed_steps:.4f}" for name in shown}
                postfix["lr"] = f"{optimizer.param_groups[0]['lr']:.2e}"
                if device.type == "cuda":
                    postfix["mem"] = f"{torch.cuda.max_memory_allocated(device) / 2 ** 30:.2f}G"
                if device.type == "cuda":
                    postfix["mem"] = f"{gpu_memory_gib(device)[0]:.2f}G"
                progress.set_postfix(postfix)

        if processed_steps == 0:
            raise RuntimeError("No training steps were processed")
        means = {name: value / processed_steps for name, value in running.items()}
        print(f"Epoch {epoch + 1} ({processed_steps} windows, {processed_frames} target frames): "
              + ", ".join(f"{name}={value:.6f}" for name, value in means.items()))
        if device.type == "cuda":
            # allocated = tensors actually in use; reserved = held by PyTorch's caching allocator.
            # nvidia-smi shows reserved + CUDA context (typically a few hundred MB more).
            print(f"[GPU memory] stage={args.stage} crop={args.crop_size} K_max={max(batch_sampler.groups)} "
                  f"batch={args.batch_size} amp={args.amp} | "
                  f"peak allocated {torch.cuda.max_memory_allocated(device) / 2 ** 30:.2f} GB, "
                  f"peak reserved {torch.cuda.max_memory_reserved(device) / 2 ** 30:.2f} GB")
        if device.type == "cuda":
            allocated, reserved = gpu_memory_gib(device)
            print(f"Peak GPU memory (stage={args.stage}, crop={args.crop_size}, "
                  f"skips={'/'.join(map(str, args.skips))}, target_mode={args.target_mode}, amp={args.amp}): "
                  f"allocated {allocated:.2f} GiB, reserved {reserved:.2f} GiB")

        if args.no_save:
            continue
        save_checkpoint(
            output_dir,
            "last.pth",
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            global_step,
            args,
        )
        if (epoch + 1) % args.save_every == 0:
            path = save_checkpoint(
                output_dir,
                f"epoch_{epoch + 1:03d}.pth",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                global_step,
                args,
            )
            print(f"Saved checkpoint: {path}")


if __name__ == "__main__":
    train(parse_args())
