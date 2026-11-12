"""Event-reference guided flow interpolation network.

Pipeline (one target time t per forward pass):
  1. E2V      : EventRecNet(E_{0->t}, E_{t->1}) -> gray reconstruction at t, supervised by gray GT (masked)
  2. coarse   : RAFT cost volume between the reference at t and gray(I_0) / gray(I_1) -> F_{t->0}, F_{t->1}
  3. refine   : events E_{t->k} + coarse flow -> refined flow + occlusion
  4. fuse     : backward-warp I_0, I_1 -> W_0, W_1 -> selective fusion -> coarse frame
  5. HCS      : event-defined motion patches (DSER++)
  6. texture  : flow-guided texture blocks from both ends (VTinker-style) inside HCS -> multi-scale refinement

Return value (same 6-tuple layout as DSER++ so train_bsergb.py / ourModel.py keep working):
  (pure_rec[3ch], masked_rec[3ch], coarse, [pred_full, pred_half, pred_quarter], hcs_union, aux_dict)
For ``stage == "rec"`` only (pure_rec[3ch], masked_rec[3ch]) is returned.
"""
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from model.blocks import EventRecNet
from model.flow_modules import CoarseFlowEstimator, EventFlowRefiner, SelectiveFusion
from model.texture import FlowGuidedTextureMapper, TextureRefineDecoder, hcs_patch_masks
from model.warp import backwarp, bgr_to_gray, reverse_voxel

STAGES = ("rec", "flow", "full")


class MyNet(nn.Module):
    def __init__(self, args):
        super(MyNet, self).__init__()
        self.bins = int(args.bins)
        self.stage = getattr(args, "stage", "full")
        if self.stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}, got {self.stage!r}")
        self.beta = float(getattr(args, "beta", 0.1))
        self.mask_patch_size = int(getattr(args, "mask_patch_size", 32))
        self.hcs_mode = getattr(args, "hcs_beta_mode", "sum")
        self.ref_mode = getattr(args, "ref_mode", "composite")
        self.rec_use_frames = bool(getattr(args, "rec_use_frames", False))
        self.detach_rec_for_flow = bool(getattr(args, "detach_rec_for_flow", False))
        flow_norm = 20.0

        # 1. E2V (identical to DSER++ when rec_use_frames=False, so its weights can be loaded)
        rec_in = 2 * self.bins + (2 if self.rec_use_frames else 0)
        self.event_rec = EventRecNet(rec_in)
        # 2. coarse flow from the cost volume
        self.coarse_flow = CoarseFlowEstimator(iters=int(getattr(args, "raft_iters", 12)),
                                               freeze_bn=not getattr(args, "no_freeze_raft_bn", False))
        # 3. event-guided refinement (shared by both directions)
        self.refiner = EventFlowRefiner(self.bins, flow_norm=flow_norm,
                                        iters=int(getattr(args, "refine_iters", 1)))
        # 4. selective fusion
        self.fusion = SelectiveFusion(flow_norm=flow_norm)
        # 5-6. texture mapping + refinement
        self.texture = FlowGuidedTextureMapper(nf=16, window=int(getattr(args, "texture_window", 3)))
        self.decoder = TextureRefineDecoder(self.bins, tex_channels=self.texture.out_channels)

    # ------------------------------------------------------------------ helpers
    def set_stage(self, stage: str) -> None:
        if stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}")
        self.stage = stage

    def build_reference(self, rec, gray0, gray1, mask):
        """Image at t that is matched against gray(I_0) / gray(I_1) in the cost volume.

        The E2V output is only supervised inside the event mask, so outside it the
        reconstruction is unconstrained. In 'composite' mode those pixels are replaced
        by the mean of the two gray key frames (no events -> no brightness change), with
        a soft transition at the mask border.
        """
        if self.ref_mode == "rec":
            return rec
        if self.ref_mode != "composite":
            raise ValueError("ref_mode must be 'composite' or 'rec'")
        soft = F.avg_pool2d(mask, kernel_size=7, stride=1, padding=3)
        return soft * rec + (1.0 - soft) * 0.5 * (gray0 + gray1)

    @staticmethod
    def _time_map(t: Optional[torch.Tensor], like: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = like.shape
        if t is None:
            return like.new_full((batch, 1, height, width), 0.5)
        t = torch.as_tensor(t, dtype=like.dtype, device=like.device).reshape(-1, 1, 1, 1)
        if t.shape[0] == 1 and batch > 1:
            t = t.expand(batch, 1, 1, 1)
        return t.expand(batch, 1, height, width)

    # ------------------------------------------------------------------ forward
    def forward(self, imgs, voxels, mask, bins=None, t=None):
        bins = self.bins if bins is None else int(bins)
        img0 = imgs[:, :3]
        img1 = imgs[:, 6:9]          # imgs[:, 3:6] is the GT target, never used here
        v0t = voxels[:, :bins]
        vt1 = voxels[:, bins:bins * 2]
        v01 = torch.cat((v0t, vt1), 1)
        gray0 = bgr_to_gray(img0)
        gray1 = bgr_to_gray(img1)

        # ---- 1. E2V ------------------------------------------------------------
        rec_in = torch.cat((v01, gray0, gray1), 1) if self.rec_use_frames else v01
        pure_rec = self.event_rec(rec_in)        # (B, 1, H, W), sigmoid
        rec = pure_rec * mask
        if self.stage == "rec":
            return pure_rec.repeat(1, 3, 1, 1), rec.repeat(1, 3, 1, 1)

        # ---- 2. coarse flow from the cost volume ------------------------------
        rec_for_flow = pure_rec.detach() if self.detach_rec_for_flow else pure_rec
        ref_t = self.build_reference(rec_for_flow, gray0, gray1, mask)
        flows_t0, flows_t1 = self.coarse_flow(ref_t, gray0, gray1)
        coarse_t0, coarse_t1 = flows_t0[-1], flows_t1[-1]

        # ---- 3. event-guided refinement (events always in the t -> k direction) -
        events_t0 = reverse_voxel(v0t)           # E_{t->0}
        events_t1 = vt1                          # E_{t->1}
        flow_t0, occ0 = self.refiner(coarse_t0, img0, ref_t, events_t0, mask)
        flow_t1, occ1 = self.refiner(coarse_t1, img1, ref_t, events_t1, mask)
        vis0 = torch.sigmoid(-occ0)              # visibility of I_0 at t
        vis1 = torch.sigmoid(-occ1)

        # ---- 4. warp + selective fusion ---------------------------------------
        warp0 = backwarp(img0, flow_t0)
        warp1 = backwarp(img1, flow_t1)
        t_map = self._time_map(t, img0)
        coarse, fusion_mask = self.fusion(warp0, warp1, ref_t, vis0, vis1, flow_t0, flow_t1, mask, t_map)

        # ---- 5. event-defined motion regions (DSER++ HCS) ----------------------
        hcs0, hcs1, hcs_union = hcs_patch_masks(img0, img1, coarse, mask, self.mask_patch_size,
                                                self.beta, self.hcs_mode)

        aux: Dict[str, object] = {
            "ref": ref_t,
            "flows_t0": flows_t0, "flows_t1": flows_t1,      # RAFT iterations (coarse)
            "flow_t0": flow_t0, "flow_t1": flow_t1,          # refined
            "occ0": occ0, "occ1": occ1,
            "warp0": warp0, "warp1": warp1,
            "fusion_mask": fusion_mask,
            "hcs0": hcs0, "hcs1": hcs1, "hcs_list": hcs_union,
        }

        if self.stage == "flow":
            # Flow stage: texture/decoder are not trained; return the coarse frame pyramid.
            predictions = [coarse,
                           F.interpolate(coarse, scale_factor=0.5, mode="bilinear", align_corners=False),
                           F.interpolate(coarse, scale_factor=0.25, mode="bilinear", align_corners=False)]
        else:
            # ---- 6. flow-guided texture mapping + refinement ------------------
            texture, tex_conf, frame0_share = self.texture(img0, img1, coarse, ref_t, flow_t0, flow_t1,
                                                           vis0, vis1)
            predictions = self.decoder(coarse, warp0, warp1, ref_t, fusion_mask, mask, vis0, vis1,
                                       v01, texture, hcs_union[1], tex_conf, frame0_share)
            aux["tex_conf"] = tex_conf
            aux["tex_frame0_share"] = frame0_share

        return (pure_rec.repeat(1, 3, 1, 1), rec.repeat(1, 3, 1, 1), coarse, predictions,
                hcs_union[0], aux)
