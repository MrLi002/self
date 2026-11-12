#!/usr/bin/env bash
# 一键配置训练环境：conda 环境 -> PyTorch(CUDA) -> 其余依赖 -> RAFT 权重 -> LPIPS 权重 -> 环境检查
#
# 用法（在项目根目录执行）：
#   bash setup_env.sh                       # 默认：环境名 evfi，Python 3.10，torch 2.5.1 + CUDA 12.1
#   CUDA_TAG=cu118 bash setup_env.sh        # 驱动较老（nvidia-smi 显示 CUDA Version < 12.1）时用 CUDA 11.8
#   CUDA_TAG=cu124 bash setup_env.sh        # 驱动支持 CUDA 12.4 及以上
#   PIP_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple bash setup_env.sh   # 普通依赖走国内镜像
#   SKIP_RAFT=1 bash setup_env.sh           # 不下载 RAFT 权重（已手动放好时）
#   CONDA_CHANNEL= bash setup_env.sh        # 置空：按 ~/.condarc 的 channels 创建环境（默认用清华镜像且忽略 defaults）
#   TORCH_INDEX=<镜像地址> bash setup_env.sh  # download.pytorch.org 太慢时换 PyTorch wheel 源
set -euo pipefail
cd "$(dirname "$0")"

ENV_NAME="${ENV_NAME:-evfi_ljh}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.20.1}"   # 必须与 torch 版本配套：2.4.1↔0.19.1, 2.5.1↔0.20.1, 2.6.0↔0.21.0
CUDA_TAG="${CUDA_TAG:-cu121}"                          # cu118 / cu121 / cu124，不能高于 nvidia-smi 显示的 CUDA Version
PIP_MIRROR="${PIP_MIRROR:-}"
SKIP_RAFT="${SKIP_RAFT:-0}"
# 只从这个 channel 创建环境（--override-channels），避免 conda 去连 repo.anaconda.com（国内常连不上）
CONDA_CHANNEL="${CONDA_CHANNEL-https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/$CUDA_TAG}"
RAFT_CKPT="checkpoints/raft-things.pth"

echo ">>> 显卡与驱动信息"
nvidia-smi || echo "警告：nvidia-smi 不可用，请确认已安装 NVIDIA 驱动"

# ---------- 1. conda 环境 ----------
if ! command -v conda >/dev/null 2>&1; then
  echo "未找到 conda，请先安装 Miniconda：https://docs.conda.io/en/latest/miniconda.html"
  exit 1
fi
set +u   # conda 的激活脚本在 set -u 下可能报 "unbound variable"
eval "$(conda shell.bash hook)"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo ">>> conda 环境 $ENV_NAME 已存在，直接使用"
else
  echo ">>> 创建 conda 环境 $ENV_NAME (python=$PYTHON_VERSION)"
  if [[ -n "$CONDA_CHANNEL" ]]; then
    conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION" --override-channels -c "$CONDA_CHANNEL"
  else
    conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
  fi
fi
conda activate "$ENV_NAME"
set -u
python -m pip install --upgrade pip

# ---------- 2. PyTorch（带 CUDA） ----------
echo ">>> 安装 torch==$TORCH_VERSION torchvision==$TORCHVISION_VERSION (源: $TORCH_INDEX)"
pip install "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" --index-url "$TORCH_INDEX"
python -c "import torch; assert torch.cuda.is_available(), 'torch 装好了但检测不到 CUDA'; print('    torch', torch.__version__, 'CUDA', torch.version.cuda, torch.cuda.get_device_name(0))"

# ---------- 3. 其余依赖 ----------
echo ">>> 安装 requirements.txt"
if [[ -n "$PIP_MIRROR" ]]; then
  pip install -r requirements.txt -i "$PIP_MIRROR"
else
  pip install -r requirements.txt
fi

# ---------- 4. RAFT 官方权重（princeton-vl/RAFT 的 download_models.sh） ----------
mkdir -p checkpoints
if [[ "$SKIP_RAFT" != "1" && ! -f "$RAFT_CKPT" ]]; then
  echo ">>> 下载 RAFT 权重"
  tmp_dir="$(mktemp -d)"
  url="https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip"
  if command -v wget >/dev/null 2>&1; then fetch=(wget -O "$tmp_dir/models.zip" "$url"); else fetch=(curl -fL -o "$tmp_dir/models.zip" "$url"); fi
  if "${fetch[@]}"; then
    unzip -o -q "$tmp_dir/models.zip" -d "$tmp_dir"
    cp "$tmp_dir/models/raft-things.pth" "$RAFT_CKPT"
    echo "    已保存到 $RAFT_CKPT"
  else
    echo "    下载失败（Dropbox 可能无法访问）。请手动从 RAFT 的 Google Drive 下载 raft-things.pth："
    echo "    https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT"
    echo "    并放到 $RAFT_CKPT"
  fi
  rm -rf "$tmp_dir"
fi

# ---------- 5. 预下载 LPIPS 所需的 AlexNet 权重（首次训练时也会自动下载） ----------
echo ">>> 预下载 LPIPS(alex) 权重"
python -c "import lpips; lpips.LPIPS(net='alex'); print('    LPIPS ok')" \
  || echo "    LPIPS 权重下载失败：训练首次启动时会再尝试，或手动放到 ~/.cache/torch/hub/checkpoints/"

# ---------- 6. 环境检查 + 冒烟测试 ----------
python tools/check_env.py --raft-ckpt "$RAFT_CKPT"
python tools/smoke_test.py --device cpu

echo
echo "环境配置完成。之后使用前先执行：conda activate $ENV_NAME"
