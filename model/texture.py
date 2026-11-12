"""Stage 5-6: motion regions from events (DSER++ style) and flow-guided texture mapping
(VTinker-inspired) followed by a multi-scale refinement decoder.

The texture mapper is an independent re-implementation of the *idea* described in
VTinker (Wu et al., AAAI 2026): texture candidates are taken from both key frames along
the estimated flows, a local search window is matched against a query computed from the
coarse frame, and the best-matching 2x2 blocks are gathered. Differences to VTinker:
the selection is a soft (differentiable) attention instead of a hard argmax, the
candidates are backward-warped with the refined F_{t->k} flows, and occlusion
visibility is used as a prior.
"""
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.blocks import event_encoder, frame_encoder, ref_encoder, up
from model.submodules import conv1x1, conv3x3, conv_resblock_one
from model.warp import backwarp, resize_flow


@torch.no_grad()
def hcs_patch_masks(img0: torch.Tensor, img1: torch.Tensor, coarse: torch.Tensor,
                    event_mask: torch.Tensor, patch_size: int = 32, beta: float = 0.1,
                    mode: str = "sum") -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """Vectorised equivalent of DSER++ ``pyramid_patch_mask`` + ``mask_reverse``.

    A patch is selected for frame k when (a) it contains at least one event and
    (b) the coarse frame differs from frame k there: sum_{patch}|I_k - coarse| >= beta
    (``mode='sum'``, identical to DSER++) or mean_{patch}|I_k - coarse| >= beta
    (``mode='mean'``, independent of the patch size).

    DSER++'s ``mask_reverse`` swaps the two masks wherever they differ, which is the
    same as swapping them completely; this is reproduced below.

    Returns (mask0, mask1, union), each a list at full, 1/2 and 1/4 resolution.
    """
    ps = int(patch_size)
    height, width = coarse.shape[-2:]
    if height % ps or width % ps:
        raise ValueError(f"Input size {height}x{width} must be divisible by patch size {ps}")
    if ps % 4:
        raise ValueError("patch_size must be divisible by 4 to build the 1/4 scale mask")
    coarse = coarse.detach()
    events_in_patch = (F.max_pool2d(event_mask.float(), ps) > 0).float()

    def select(frame):
        score = F.avg_pool2d((frame - coarse).abs().sum(1, keepdim=True), ps)
        if mode == "sum":
            score = score * (ps * ps)
        elif mode != "mean":
            raise ValueError("mode must be 'sum' or 'mean'")
        return (score >= beta).float() * events_in_patch

    sel0, sel1 = select(img0), select(img1)
    patch0, patch1 = sel1, sel0  # DSER++ mask_reverse
    union = torch.clamp(patch0 + patch1, max=1.0)

    def pyramid(mask):
        return [F.interpolate(mask, scale_factor=ps // s, mode="nearest") for s in (1, 2, 4)]

    return pyramid(patch0), pyramid(patch1), pyramid(union)


class FlowGuidedTextureMapper(nn.Module):
    """Gather 2x2 texture blocks from both key frames along F_{t->0} and F_{t->1}.

    * descriptors (queries/keys) live at 1/4 resolution, values at 1/2 resolution;
      one 1/4-scale descriptor therefore selects one 2x2 block of 1/2-scale values;
    * for every block a (window x window) neighbourhood around the flow target is
      searched in both frames -> 2 * window^2 candidates;
    * candidate scores = cosine similarity / temperature + log(visibility) prior.
    """

    def __init__(self, nf: int = 16, window: int = 3):
        super().__init__()
        if window % 2 == 0:
            raise ValueError("window must be odd")
        self.window = int(window)
        self.img_enc = frame_encoder(3, nf)      # [nf @ H, 2nf @ H/2, 4nf @ H/4]
        self.query_enc = ref_encoder(4, nf)      # input: coarse frame (3) + reference (1)
        desc = 4 * nf
        self.q_proj = conv1x1(desc, desc)
        self.k_proj = conv1x1(desc, desc)
        self.log_scale = nn.Parameter(torch.tensor(math.log(10.0)))  # inverse temperature
        self.vis_weight = nn.Parameter(torch.tensor(1.0))
        self.out_proj = conv_resblock_one(2 * nf, 2 * nf)
        self.out_channels = 2 * nf

    def _unfold(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        n = self.window * self.window
        cols = F.unfold(x, self.window, padding=self.window // 2)
        return cols.view(batch, channels, n, height, width)

    def forward(self, img0, img1, coarse, ref_t, flow_t0, flow_t1, vis0, vis1):
        batch = img0.shape[0]
        feats = self.img_enc(torch.cat((img0, img1), dim=0))
        f_half, f_quarter = feats[1], feats[2]
        query = self.query_enc(torch.cat((coarse, ref_t), dim=1))[2]

        half = f_half.shape[-2:]
        quarter = f_quarter.shape[-2:]
        flows_half = torch.cat((resize_flow(flow_t0, size=half), resize_flow(flow_t1, size=half)), 0)
        flows_quarter = torch.cat((resize_flow(flow_t0, size=quarter), resize_flow(flow_t1, size=quarter)), 0)
        values = backwarp(f_half, flows_half)           # (2B, 2nf, H/2, W/2), aligned to t
        keys = backwarp(f_quarter, flows_quarter)       # (2B, 4nf, H/4, W/4), aligned to t

        q = F.normalize(self.q_proj(query), dim=1)
        k = F.normalize(self.k_proj(keys), dim=1)
        k0, k1 = k[:batch], k[batch:]
        score0 = (q.unsqueeze(2) * self._unfold(k0)).sum(1)   # (B, n, H/4, W/4)
        score1 = (q.unsqueeze(2) * self._unfold(k1)).sum(1)
        scale = self.log_scale.exp()
        prior0 = self.vis_weight * torch.log(F.adaptive_avg_pool2d(vis0, quarter).clamp_min(1e-4))
        prior1 = self.vis_weight * torch.log(F.adaptive_avg_pool2d(vis1, quarter).clamp_min(1e-4))
        logits = torch.cat((score0 * scale + prior0, score1 * scale + prior1), dim=1)
        weights = torch.softmax(logits, dim=1)
        n = self.window * self.window
        w0, w1 = weights[:, :n], weights[:, n:]

        v0 = F.pixel_unshuffle(values[:batch], 2)       # (B, 8nf, H/4, W/4): 2x2 blocks
        v1 = F.pixel_unshuffle(values[batch:], 2)
        gathered = (self._unfold(v0) * w0.unsqueeze(1)).sum(2) + (self._unfold(v1) * w1.unsqueeze(1)).sum(2)
        texture = self.out_proj(F.pixel_shuffle(gathered, 2))  # (B, 2nf, H/2, W/2)
        confidence = weights.max(dim=1, keepdim=True)[0]       # (B, 1, H/4, W/4)
        frame0_share = w0.sum(dim=1, keepdim=True)             # how much texture came from I_0
        return texture, confidence, frame0_share


class TextureRefineDecoder(nn.Module):
    """U-Net refinement that predicts residuals on top of the fused coarse frame.

    The texture features are injected at 1/2 resolution only inside the event-defined
    motion (HCS) regions. Returns predictions at full, 1/2 and 1/4 resolution, which is
    what ``multiscale_l1`` in train_bsergb.py expects.
    """

    def __init__(self, bins: int, tex_channels: int = 32):
        super().__init__()
        in_ch = 3 + 3 + 3 + 1 + 1 + 1 + 1 + 1   # coarse, warp0, warp1, ref, M, mask, vis0, vis1
        self.ev_enc = event_encoder(2 * bins, 16)   # [16 @ H, 32 @ H/2, 64 @ H/4]
        self.enc1 = conv_resblock_one(in_ch, 32)
        self.fuse1 = conv_resblock_one(32 + 16, 32)
        self.enc2 = conv_resblock_one(32, 64, stride=2)
        self.fuse2 = conv_resblock_one(64 + 32 + tex_channels + 1, 64)
        self.enc3 = conv_resblock_one(64, 96, stride=2)
        self.fuse3 = conv_resblock_one(96 + 64 + 2, 96)
        self.head3 = conv3x3(96, 3)
        self.up2 = up(96, 64)
        self.head2 = conv3x3(64, 3)
        self.up1 = up(64, 32)
        self.head1 = conv3x3(32, 3)
        for head in (self.head1, self.head2, self.head3):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, coarse, warp0, warp1, ref_t, fusion_mask, event_mask, vis0, vis1,
                v01, texture, hcs_half, tex_conf, frame0_share):
        ev = self.ev_enc(v01)
        x1 = self.enc1(torch.cat((coarse, warp0, warp1, ref_t, fusion_mask, event_mask, vis0, vis1), 1))
        x1 = self.fuse1(torch.cat((x1, ev[0]), 1))
        x2 = self.enc2(x1)
        x2 = self.fuse2(torch.cat((x2, ev[1], texture * hcs_half, hcs_half), 1))
        x3 = self.enc3(x2)
        x3 = self.fuse3(torch.cat((x3, ev[2], tex_conf, frame0_share), 1))

        coarse_half = F.interpolate(coarse, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        coarse_quarter = F.interpolate(coarse, size=x3.shape[-2:], mode="bilinear", align_corners=False)
        pred_quarter = coarse_quarter + self.head3(x3)
        y2 = self.up2(x3, x2)
        pred_half = coarse_half + self.head2(y2)
        y1 = self.up1(y2, x1)
        pred_full = coarse + self.head1(y1)
        return [p.clamp(0.0, 1.0) for p in (pred_full, pred_half, pred_quarter)]
