"""Check that the environment can run this project.

    python tools/check_env.py [--raft-ckpt checkpoints/raft-things.pth]

Prints library versions and GPU info, imports every project module, checks the
APIs the code relies on, and (if given) loads the RAFT weights into the
project's RAFT so that a wrong / corrupted checkpoint is caught before training.
"""
import argparse
import importlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

failures = []


def check(name, fn):
    try:
        result = fn()
        print(f"[ok]   {name}" + (f": {result}" if result else ""))
    except Exception as error:  # noqa: BLE001 - report everything
        failures.append(name)
        print(f"[FAIL] {name}: {type(error).__name__}: {error}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raft-ckpt", default="checkpoints/raft-things.pth")
    args = parser.parse_args()

    print(f"Python {sys.version.split()[0]}")
    for package in ("torch", "torchvision", "numpy", "scipy", "cv2", "PIL", "matplotlib", "tqdm", "lpips"):
        check(f"import {package}", lambda p=package: getattr(importlib.import_module(p), "__version__", ""))

    import torch
    import torch.nn.functional as F

    def cuda_info():
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available (CPU-only torch build or driver problem)")
        props = torch.cuda.get_device_properties(0)
        return (f"torch CUDA {torch.version.cuda}, {torch.cuda.device_count()} GPU(s), "
                f"GPU0 = {props.name}, {props.total_memory / 2 ** 30:.1f} GB")
    check("CUDA", cuda_info)

    def deform_conv():
        from torchvision.ops import DeformConv2d  # noqa: F401  (used by model/blocks.py)
        return ""
    check("torchvision.ops.DeformConv2d", deform_conv)

    def tensor_apis():
        assert F.pixel_unshuffle(torch.zeros(1, 1, 4, 4), 2).shape == (1, 4, 2, 2)
        torch.meshgrid(torch.arange(2), torch.arange(2), indexing="ij")
        return ""
    check("F.pixel_unshuffle / meshgrid(indexing)", tensor_apis)

    def unbatched_conv():
        assert F.conv2d(torch.zeros(1, 8, 8), torch.ones(1, 1, 3, 3), padding=1).shape == (1, 8, 8)
        return ""
    check("unbatched conv2d (util/utils_func.process_mask)", unbatched_conv)

    for module in ("model.ourNet", "model.ourModel", "model.flow_modules", "model.texture", "model.warp",
                   "datasets", "util.voxelization", "util.event", "train_bsergb"):
        check(f"import {module}", lambda m=module: importlib.import_module(m) and "")

    def voxel_numpy2():
        import numpy as np
        from util.event import EventSequence
        from util.voxelization import to_voxel_grid
        feats = np.array([[1, 1, 0.0, 1], [2, 2, 0.5, -1], [3, 3, 1.0, 1]], dtype=np.float32)
        return f"voxel shape {tuple(to_voxel_grid(EventSequence(feats, 8, 8), 4).shape)}"
    check("to_voxel_grid (NumPy compatibility)", voxel_numpy2)

    def lpips_alex():
        import lpips
        lpips.LPIPS(net="alex", verbose=False)
        return "AlexNet weights available"
    check("LPIPS(alex)", lpips_alex)

    def raft_weights():
        if not os.path.isfile(args.raft_ckpt):
            raise FileNotFoundError(f"{args.raft_ckpt} not found (needed for --stage flow)")
        from model.flow_modules import CoarseFlowEstimator
        CoarseFlowEstimator().load_pretrained(args.raft_ckpt)
        return f"{args.raft_ckpt} loads strictly into the project's RAFT"
    check("RAFT checkpoint", raft_weights)

    print()
    if failures:
        print(f"{len(failures)} check(s) failed: {', '.join(failures)}")
        sys.exit(1)
    print("Environment OK.")


if __name__ == "__main__":
    main()
