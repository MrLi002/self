"""Stage 2-4 modules: coarse flow (cost volume), event-guided refinement, selective fusion."""
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.blocks import DenseBlock, up
from model.raft import RAFT, default_raft_args
from model.submodules import conv_resblock_one
from model.warp import backwarp, bgr_to_gray


def _strip_prefix(state_dict, prefix="module."):
    return {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in state_dict.items()}


class CoarseFlowEstimator(nn.Module):
    """Cost-volume flow between the event reconstruction at t and the gray key frames.

    Both directions are computed in one RAFT call by stacking them on the batch axis:
        image1 = [ref_t, ref_t],  image2 = [gray_0, gray_1]
    so the outputs are F_{t->0} and F_{t->1} on the grid of time t, i.e. exactly the
    flows needed for backward warping I_0 / I_1 to time t.
    """

    def __init__(self, iters: int = 12, freeze_bn: bool = True):
        super().__init__()
        # input_scale=1.0: inputs are in [0, 1]; RAFT maps them to [-1, 1] as in the original.
        self.raft = RAFT(default_raft_args(), input_scale=1.0)
        self.iters = int(iters)
        self.freeze_bn = bool(freeze_bn)

    def load_pretrained(self, path: str) -> None:
        """Load official RAFT weights (e.g. raft-things.pth); 'module.' prefixes are stripped."""
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # torch < 1.13 has no weights_only argument
            checkpoint = torch.load(path, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        self.raft.load_state_dict(_strip_prefix(checkpoint), strict=True)

    def train(self, mode: bool = True):
        super().train(mode)
        # RAFT's context encoder uses BatchNorm; with batch size 1 its statistics are
        # meaningless, so (like RAFT fine-tuning) keep BN layers in eval mode.
        if mode and self.freeze_bn:
            self.raft.freeze_bn()
        return self

    def forward(self, ref_t: torch.Tensor, gray0: torch.Tensor, gray1: torch.Tensor,
                iters: Optional[int] = None) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        batch = ref_t.shape[0]
        # RAFT runs its encoders with autocast disabled, so feed fp32 even under --amp.
        image1 = torch.cat((ref_t, ref_t), dim=0).repeat(1, 3, 1, 1).float()
        image2 = torch.cat((gray0, gray1), dim=0).repeat(1, 3, 1, 1).float()
        predictions = self.raft(image1, image2, iters=iters or self.iters)
        flows_t0 = [flow[:batch] for flow in predictions]
        flows_t1 = [flow[batch:] for flow in predictions]
        return flows_t0, flows_t1


class EventFlowRefiner(nn.Module):
    """Refine a coarse flow F_{t->k} with the events between t and k.

    Inputs per direction (weights are shared by both directions because the events are
    always given in the t -> k direction, see ``reverse_voxel``):
        coarse flow, I_k warped to t, the reference at t, |gray(warp) - ref|,
        voxel E_{t->k}, event mask.
    Outputs: refined flow (coarse + residual) and an occlusion logit for frame k.
    """

    def __init__(self, bins: int, flow_norm: float = 20.0, iters: int = 1):
        super().__init__()
        in_ch = 2 + 3 + 1 + 1 + bins + 1
        self.flow_norm = float(flow_norm)
        self.iters = int(iters)
        self.enc1 = conv_resblock_one(in_ch, 32)
        self.enc2 = conv_resblock_one(32, 64, stride=2)
        self.enc3 = conv_resblock_one(64, 96, stride=2)
        self.dec2 = up(96, 64)
        self.dec1 = up(64, 32)
        self.head = nn.Conv2d(32, 3, 3, 1, 1)
        nn.init.zeros_(self.head.weight)  # start exactly from the coarse flow
        nn.init.zeros_(self.head.bias)

    def _step(self, flow, image_k, ref_t, events_tk, mask):
        warped = backwarp(image_k, flow)
        error = (bgr_to_gray(warped) - ref_t).abs()
        x = torch.cat((flow / self.flow_norm, warped, ref_t, error, events_tk, mask), dim=1)
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        y = self.dec1(self.dec2(s3, s2), s1)
        out = self.head(y)
        return flow + out[:, :2], out[:, 2:3]

    def forward(self, coarse_flow, image_k, ref_t, events_tk, mask):
        flow = coarse_flow
        # iters == 0 disables the refinement (ablation): coarse flow, neutral occlusion logit.
        occ_logit = coarse_flow.new_zeros(coarse_flow.shape[0], 1, *coarse_flow.shape[-2:])
        for _ in range(self.iters):
            flow, occ_logit = self._step(flow, image_k, ref_t, events_tk, mask)
        return flow, occ_logit


class SelectiveFusion(nn.Module):
    """Pixel-wise selection between the two warped frames: I_t = M * W_0 + (1 - M) * W_1."""

    def __init__(self, flow_norm: float = 20.0):
        super().__init__()
        # warp0(3) + warp1(3) + ref(1) + vis0(1) + vis1(1) + flows(4) + event mask(1) + t(1)
        self.net = DenseBlock(15, 1)
        self.flow_norm = float(flow_norm)

    def forward(self, warp0, warp1, ref_t, vis0, vis1, flow_t0, flow_t1, mask, t_map):
        x = torch.cat((warp0, warp1, ref_t, vis0, vis1,
                       flow_t0 / self.flow_norm, flow_t1 / self.flow_norm, mask, t_map), dim=1)
        weight0 = self.net(x)  # sigmoid output in (0, 1)
        fused = weight0 * warp0 + (1.0 - weight0) * warp1
        return fused, weight0
