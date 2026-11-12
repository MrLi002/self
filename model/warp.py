"""Warping and flow utilities shared by the flow, fusion and texture modules.

Flow convention used everywhere in this project
------------------------------------------------
``flow_t0`` (B, 2, H, W) is defined on the pixel grid of the *target* time t and
points into frame 0: pixel ``x`` of I_t corresponds to ``x + flow_t0(x)`` in I_0.
Channel 0 is the horizontal (x) displacement, channel 1 the vertical (y) one,
both in pixels of the tensor they are applied to.

With this convention ``backwarp(I_0, flow_t0)`` produces an estimate of I_t.
This is the direction RAFT returns for ``RAFT(image_t, image_0)``.
"""
import torch
import torch.nn.functional as F


def bgr_to_gray(image: torch.Tensor) -> torch.Tensor:
    """OpenCV BGR (channel order of the BS-ERGB loader) -> luminance, keeps [0, 1]."""
    weights = image.new_tensor((0.114, 0.587, 0.299)).view(1, 3, 1, 1)
    return (image * weights).sum(dim=1, keepdim=True)


def _base_grid(flow: torch.Tensor):
    _, _, height, width = flow.shape
    ys = torch.arange(height, device=flow.device, dtype=flow.dtype)
    xs = torch.arange(width, device=flow.device, dtype=flow.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return grid_x, grid_y


def backwarp(image: torch.Tensor, flow: torch.Tensor, padding_mode: str = "border") -> torch.Tensor:
    """Sample ``image`` at ``x + flow(x)`` (backward warping, bilinear)."""
    if image.shape[-2:] != flow.shape[-2:]:
        raise ValueError(f"image {tuple(image.shape)} and flow {tuple(flow.shape)} sizes differ")
    _, _, height, width = flow.shape
    grid_x, grid_y = _base_grid(flow)
    x = grid_x.unsqueeze(0) + flow[:, 0]
    y = grid_y.unsqueeze(0) + flow[:, 1]
    x = 2.0 * x / max(width - 1, 1) - 1.0
    y = 2.0 * y / max(height - 1, 1) - 1.0
    grid = torch.stack((x, y), dim=-1)
    return F.grid_sample(image, grid.to(image.dtype), mode="bilinear",
                         padding_mode=padding_mode, align_corners=True)


def out_of_frame_mask(flow: torch.Tensor) -> torch.Tensor:
    """1 where ``x + flow(x)`` falls inside the source frame, else 0. Shape (B, 1, H, W)."""
    _, _, height, width = flow.shape
    grid_x, grid_y = _base_grid(flow)
    x = grid_x.unsqueeze(0) + flow[:, 0]
    y = grid_y.unsqueeze(0) + flow[:, 1]
    inside = (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)
    return inside.unsqueeze(1).to(flow.dtype)


def resize_flow(flow: torch.Tensor, scale: float = None, size=None) -> torch.Tensor:
    """Resize a flow field and rescale its vectors accordingly."""
    _, _, height, width = flow.shape
    if size is None:
        size = (int(round(height * scale)), int(round(width * scale)))
    new_h, new_w = size
    resized = F.interpolate(flow, size=(new_h, new_w), mode="bilinear", align_corners=False)
    factor = flow.new_tensor((new_w / width, new_h / height)).view(1, 2, 1, 1)
    return resized * factor


def reverse_voxel(voxel: torch.Tensor) -> torch.Tensor:
    """Time-reverse a voxel grid (bins on dim 1): flip the bin order and the polarity.

    For the bilinear temporal voxelisation in util/voxelization.py this equals
    voxelising the time-reversed, polarity-flipped event stream, i.e. it turns
    E_{0->t} into E_{t->0}.
    """
    return -torch.flip(voxel, dims=[1])


def edge_aware_smoothness(flow: torch.Tensor, image: torch.Tensor, alpha: float = 10.0) -> torch.Tensor:
    """First-order edge-aware smoothness (lower weight across image edges)."""
    gray = bgr_to_gray(image) if image.shape[1] == 3 else image
    dx_img = (gray[..., :, 1:] - gray[..., :, :-1]).abs()
    dy_img = (gray[..., 1:, :] - gray[..., :-1, :]).abs()
    dx_flow = (flow[..., :, 1:] - flow[..., :, :-1]).abs().mean(1, keepdim=True)
    dy_flow = (flow[..., 1:, :] - flow[..., :-1, :]).abs().mean(1, keepdim=True)
    return (dx_flow * torch.exp(-alpha * dx_img)).mean() + (dy_flow * torch.exp(-alpha * dy_img)).mean()
