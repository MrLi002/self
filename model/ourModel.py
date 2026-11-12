import os
from torch.optim import AdamW
from common.size_adapter import ImgAndEventSizeAdapter
from model.ourNet import *
from model.loss import *
from common.laplacian import *
from util.utils_func import pyramid_Img

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model:
    def __init__(self, args, inference_only=False):
        self.my_net = MyNet(args)
        self.args = args
        self.device()
        self.device_num = torch.cuda.device_count()
        self.is_multiple()
        self.size_adapter = ImgAndEventSizeAdapter()
        self.optimG = None
        self.lap = None
        self.perc = None
        self.patch_ssim = None
        if not inference_only:
            import lpips

            self.optimG = AdamW(self.my_net.parameters(), lr=1e-6, weight_decay=1e-3)
            self.lap = LapLoss()
            self.perc = lpips.LPIPS(net='alex').to(device)
            self.patch_ssim = SSIM_with_patch_mask(args.mask_patch_size)

    def is_multiple(self):
        if self.device_num > 1:
            print("Start with", self.device_num, "GPUs!")
            self.my_net = nn.DataParallel(self.my_net, device_ids=list(range(self.device_num)))

    def train(self):
        self.my_net.train()

    def eval(self):
        self.my_net.eval()

    def device(self):
        self.my_net.to(device)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=device)
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get(
                "net", checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
            )
        else:
            state_dict = checkpoint
        if not isinstance(state_dict, dict):
            raise TypeError("Checkpoint does not contain a model state dictionary")

        normalized_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[7:]
            # Released checkpoints used the name below before the embedding
            # layer was wrapped in Rearrange + Linear. The tensors are the
            # same Linear weights and biases, so migrate only these exact keys.
            key = key.replace(
                ".transformer_block.to_patch_embedding2.0.",
                ".transformer_block.to_patch_embedding.1.",
            )
            normalized_state_dict[key] = value

        network = self.my_net.module if isinstance(self.my_net, nn.DataParallel) else self.my_net
        network.load_state_dict(normalized_state_dict, strict=True)

    def save_model(self, path):
        torch.save(self.my_net.state_dict(), '{}/work.pkl'.format(path))

    def save_model_min_loss(self, path):
        torch.save(self.my_net.state_dict(), '{}/min_loss.pkl'.format(path))

    def save_checkpoint(self, type, epoch):
        checkpoint = {
            "net": self.my_net.state_dict(),
            'optimizer': self.optimG.state_dict(),
            "epoch": epoch
        }
        if not os.path.isdir("./train/checkpoint"):
            os.mkdir("./train/checkpoint")
        torch.save(checkpoint, './train/checkpoint/ckpt_%s.pth' % (str(epoch)))

    @torch.no_grad()
    def inference(self, imgs, voxels, mask, t=None):
        """imgs: (B, 9, H, W) BGR in [0, 1] (the middle 3 channels are ignored),
        voxels: (B, 3*bins, H, W) = [E_0->t, E_t->1, E_1->t], mask: (B, 1, H, W) event mask,
        t: optional (B,) normalised target time. Returns the final prediction (B, 3, H, W)."""
        imgs, voxels, mask = self.size_adapter.pad(imgs, voxels, mask)
        self.eval()
        _, _, _, pred, _, _ = self.my_net(imgs, voxels, mask, self.args.bins, t)
        pred = self.size_adapter.unpad(pred[0])
        return pred

    @torch.no_grad()
    def inference_window(self, endpoints, voxels, masks, t, chunk=None):
        """All K intermediate frames of one window (e.g. skip=3 -> 3 frames).

        endpoints: (B, 6, H, W) = [I_0, I_1]; voxels: (B, K, 3*bins, H, W);
        masks: (B, K, 1, H, W); t: (B, K). Returns (B, K, 3, H, W).
        ``chunk`` limits how many targets are processed per forward pass (memory).
        """
        batch, num_targets = voxels.shape[:2]
        height, width = endpoints.shape[-2:]
        img0 = endpoints[:, None, :3].expand(batch, num_targets, 3, height, width)
        img1 = endpoints[:, None, 3:6].expand(batch, num_targets, 3, height, width)
        imgs = torch.cat((img0, torch.zeros_like(img0), img1), dim=2).reshape(-1, 9, height, width)
        voxels = voxels.reshape(batch * num_targets, -1, height, width)
        masks = masks.reshape(batch * num_targets, 1, height, width)
        t = t.reshape(batch * num_targets)
        step = chunk or imgs.shape[0]
        preds = [self.inference(imgs[i:i + step], voxels[i:i + step], masks[i:i + step], t[i:i + step])
                 for i in range(0, imgs.shape[0], step)]
        return torch.cat(preds, dim=0).reshape(batch, num_targets, 3, height, width)
