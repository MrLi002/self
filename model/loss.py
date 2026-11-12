import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.fft
from math import exp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EPE(nn.Module):
    def __init__(self):
        super(EPE, self).__init__()

    def forward(self, flow, gt, loss_mask):
        loss_map = (flow - gt.detach()) ** 2
        loss_map = (loss_map.sum(1, True) + 1e-6) ** 0.5
        return (loss_map * loss_mask)


class Ternary(nn.Module):
    def __init__(self):
        super(Ternary, self).__init__()
        patch_size = 7
        out_channels = patch_size * patch_size
        self.w = np.eye(out_channels).reshape(
            (patch_size, patch_size, 1, out_channels))
        self.w = np.transpose(self.w, (3, 2, 0, 1))
        self.w = torch.tensor(self.w).float().to(device)

    def transform(self, img):
        patches = F.conv2d(img, self.w, padding=3, bias=None)
        transf = patches - img
        transf_norm = transf / torch.sqrt(0.81 + transf ** 2)
        return transf_norm

    def rgb2gray(self, rgb):
        r, g, b = rgb[:, 0:1, :, :], rgb[:, 1:2, :, :], rgb[:, 2:3, :, :]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    def hamming(self, t1, t2):
        dist = (t1 - t2) ** 2
        dist_norm = torch.mean(dist / (0.1 + dist), 1, True)
        return dist_norm

    def valid_mask(self, t, padding):
        n, _, h, w = t.size()
        inner = torch.ones(n, 1, h - 2 * padding, w - 2 * padding).type_as(t)
        mask = F.pad(inner, [padding] * 4)
        return mask

    def forward(self, img0, img1):
        img0 = self.transform(self.rgb2gray(img0))
        img1 = self.transform(self.rgb2gray(img1))
        return self.hamming(img0, img1) * self.valid_mask(img0, 1)


class SOBEL(nn.Module):
    def __init__(self):
        super(SOBEL, self).__init__()
        self.kernelX = torch.tensor([
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1],
        ]).float()
        self.kernelY = self.kernelX.clone().T
        self.kernelX = self.kernelX.unsqueeze(0).unsqueeze(0).to(device)
        self.kernelY = self.kernelY.unsqueeze(0).unsqueeze(0).to(device)

    def forward(self, pred, gt):
        N, C, H, W = pred.shape[0], pred.shape[1], pred.shape[2], pred.shape[3]
        img_stack = torch.cat(
            [pred.reshape(N * C, 1, H, W), gt.reshape(N * C, 1, H, W)], 0)
        sobel_stack_x = F.conv2d(img_stack, self.kernelX, padding=1)
        sobel_stack_y = F.conv2d(img_stack, self.kernelY, padding=1)
        pred_X, gt_X = sobel_stack_x[:N * C], sobel_stack_x[N * C:]
        pred_Y, gt_Y = sobel_stack_y[:N * C], sobel_stack_y[N * C:]

        L1X, L1Y = torch.abs(pred_X - gt_X), torch.abs(pred_Y - gt_Y)
        loss = (L1X + L1Y)
        return loss


class MeanShift(nn.Conv2d):
    def __init__(self, data_mean, data_std, data_range=1, norm=True):
        c = len(data_mean)
        super(MeanShift, self).__init__(c, c, kernel_size=1)
        std = torch.Tensor(data_std)
        self.weight.data = torch.eye(c).view(c, c, 1, 1)
        if norm:
            self.weight.data.div_(std.view(c, 1, 1, 1))
            self.bias.data = -1 * data_range * torch.Tensor(data_mean)
            self.bias.data.div_(std)
        else:
            self.weight.data.mul_(std.view(c, 1, 1, 1))
            self.bias.data = data_range * torch.Tensor(data_mean)
        self.requires_grad = False


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def expand_patches(mask, patch_size):
    B, C, H, W = mask.shape
    patch_H = H // patch_size
    patch_W = W // patch_size

    small_mask = mask.view(B, C, patch_H, patch_size, patch_W, patch_size).max(dim=5)[0].max(dim=3)[0]

    kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=mask.device)

    dilated_small_mask = F.conv2d(small_mask, kernel, padding=1, stride=1)
    dilated_small_mask = (dilated_small_mask > 0).float()

    dilated_mask = F.interpolate(dilated_small_mask, size=(H, W), mode='nearest')

    return dilated_mask


def ssim(img1, img2, mask, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    valid_mask = F.conv2d(mask.float(), window, padding=padd, groups=1)
    valid_mask = (valid_mask == 1)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    ssim_map = ssim_map * valid_mask

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


# Classes to re-use window
class SSIM_with_patch_mask(torch.nn.Module):
    def __init__(self, patch_size, window_size=11, size_average=True, val_range=None):
        super(SSIM_with_patch_mask, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range
        self.patch_size = patch_size

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2, mask):
        (_, channel, _, _) = img1.size()

        mask = expand_patches(mask, self.patch_size)

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, mask, window=window, window_size=self.window_size, size_average=self.size_average)


class MaskedSSIMLoss(torch.nn.Module):
    """SSIM loss averaged only over centers selected by a spatial mask.

    This follows the region-normalized structural-consistency objective used
    by DSER++: pixels outside the selected HCS regions do not affect either the
    numerator or denominator.
    """

    def __init__(self, window_size=11, sigma=1.5, val_range=1.0):
        super(MaskedSSIMLoss, self).__init__()
        self.window_size = int(window_size)
        self.sigma = float(sigma)
        self.val_range = float(val_range)

    def _window(self, channels, device, dtype):
        coords = torch.arange(
            self.window_size, device=device, dtype=dtype
        ) - self.window_size // 2
        gaussian_1d = torch.exp(-(coords ** 2) / (2 * self.sigma ** 2))
        gaussian_1d = gaussian_1d / gaussian_1d.sum()
        window_2d = gaussian_1d[:, None] * gaussian_1d[None, :]
        return window_2d.expand(channels, 1, -1, -1).contiguous()

    def forward(self, img1, img2, mask):
        if img1.shape != img2.shape:
            raise ValueError("MaskedSSIMLoss inputs must have identical shapes")
        if mask.ndim != 4 or mask.shape[0] != img1.shape[0]:
            raise ValueError("mask must have shape [B, 1, H, W]")
        if mask.shape[-2:] != img1.shape[-2:]:
            mask = F.interpolate(mask.float(), size=img1.shape[-2:], mode="nearest")

        channels = img1.shape[1]
        window = self._window(channels, img1.device, img1.dtype)
        padding = self.window_size // 2
        mu1 = F.conv2d(img1, window, padding=padding, groups=channels)
        mu2 = F.conv2d(img2, window, padding=padding, groups=channels)
        mu1_sq = mu1.square()
        mu2_sq = mu2.square()
        mu1_mu2 = mu1 * mu2

        sigma1_sq = (
            F.conv2d(img1 * img1, window, padding=padding, groups=channels)
            - mu1_sq
        )
        sigma2_sq = (
            F.conv2d(img2 * img2, window, padding=padding, groups=channels)
            - mu2_sq
        )
        sigma12 = (
            F.conv2d(img1 * img2, window, padding=padding, groups=channels)
            - mu1_mu2
        )

        c1 = (0.01 * self.val_range) ** 2
        c2 = (0.03 * self.val_range) ** 2
        numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
        denominator = (mu1_sq + mu2_sq + c1) * (
            sigma1_sq + sigma2_sq + c2
        )
        ssim_map = numerator / denominator.clamp_min(torch.finfo(img1.dtype).eps)

        selected = (mask > 0.5).to(dtype=img1.dtype)
        selected = selected.expand(-1, channels, -1, -1)
        selected_count = selected.sum()
        if selected_count.item() == 0:
            return img1.sum() * 0.0
        return ((1.0 - ssim_map) * selected).sum() / selected_count


if __name__ == '__main__':
    img0 = torch.zeros(16, 3, 256, 256).float().to(device)
    img1 = torch.tensor(np.random.normal(
        0, 1, (16, 3, 256, 256))).float().to(device)
    ternary_loss = Ternary()
    print(ternary_loss(img0, img1).shape)
