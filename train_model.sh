#!/usr/bin/env bash
# 分阶段训练：P1 重建(E2V) -> P2 光流+粗插帧 -> P3 端到端(纹理映射+精修)
# 每个 skip=N 窗口同时生成全部 N 个中间时刻（--target-mode all）
# 用法：
#   bash train_model.sh            # 从 P1 开始依次跑完三个阶段
#   bash train_model.sh flow       # 从 P2 开始（需要已有 P1 的 last.pth，或设置 REC_INIT）
#   bash train_model.sh full       # 只跑 P3（需要已有 P2 的 last.pth）
#   REC_INIT=/path/dser.pth bash train_model.sh flow   # 用 DSER++ 的 E2V 权重代替 P1
#   EVAL_ROOT=/path/BSERGB/1_TEST bash train_model.sh  # P2、P3 结束后自动在测试集上评测 PSNR/SSIM
#
# 输出：终端上每个 epoch 只有一行进度条
#   Train Epoch(0/20): 100%|██████| 16984/16984 [4:40:32<00:00, 1.01it/s, sum_loss=897.4082, learn_rate=0.000099]
# 详细信息（各项 loss 均值、峰值显存、加载了哪些权重）写在 train/logs/*.log，
# 另开一个终端用 tail -f train/logs/p1_rec.log 查看。报错信息仍显示在终端。
set -euo pipefail
cd "$(dirname "$0")"

BSERGB_ROOT="/home/yuaobo/桌面/ljh/dataset/3_TRAINING"
RAFT_CKPT="${RAFT_CKPT:-checkpoints/raft-things.pth}"   # princeton-vl/RAFT 官方权重
REC_INIT="${REC_INIT:-}"                                # 可选：DSER++ 权重，跳过 P1
START_STAGE="${1:-rec}"                                 # rec | flow | full
EVAL_ROOT="${EVAL_ROOT:-}"                              # 可选：BS-ERGB 测试集根目录（如 .../1_TEST）
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1                               # 日志文件里的内容立即写入，不攒在缓冲区
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"   # 减少显存碎片

REC_DIR="train/bsergb_rec"
FLOW_DIR="train/bsergb_flow"
FULL_DIR="train/bsergb_full"
LOG_DIR="train/logs"
mkdir -p "$LOG_DIR"

# 三个阶段共用的数据与基本设置
COMMON=(
  --data-root "$BSERGB_ROOT"
  --bins 8 --skips 1 3
  --crop-size 448 --mask-patch-size 32 --beta 0.1
  --batch-size 1 --workers 8
  --seed 1234 --log-interval 20
  # skip=3 的窗口一次生成 3 张（P1: 3 张灰度图；P2/P3: 3 张预测帧），skip=1 生成 1 张
  --target-mode all --target-weighting window
  # 同一窗口相邻目标帧之间的时间一致性（可设为 0 做消融）
  --lambda-tc 0.1
)

# 只把 stdout 写进日志；stderr（tqdm 进度条、报错）直接留在终端。
# 不要写成 2>&1 | tee：进度条经过管道后 tqdm 拿不到终端宽度，长行一折行，每次刷新都会多出一行。
run_logged() {
  local log="$1"; shift
  echo ">>> $(date '+%F %T') 开始，详细日志：$log"
  "$@" > "$log"
}

# 评测（设置了 EVAL_ROOT 才执行）：P2 评粗插帧结果，P3 评最终结果（评测没有进度条，照常 tee）
run_eval() {
  [[ -n "$EVAL_ROOT" ]] || return 0
  python eval_bsergb.py --data-root "$EVAL_ROOT" --checkpoint "$1" --skips 1 3 \
    --output-dir "$(dirname "$1")/eval_bsergb" 2>&1 | tee "$LOG_DIR/eval_$2.log"
}

case "$START_STAGE" in
  rec|flow|full) ;;
  *) echo "未知阶段: $START_STAGE（可选 rec / flow / full）"; exit 1 ;;
esac

# P0：冒烟测试（检查光流方向、HCS 掩码、各阶段前向/反向是否正常）
python tools/smoke_test.py --device cpu

# P1：只训练 E2V（EventRecNet），监督为 事件掩码 × 灰度 GT
if [[ "$START_STAGE" == "rec" ]]; then
  run_logged "$LOG_DIR/p1_rec.log" \
    python train_bsergb.py "${COMMON[@]}" \
    --stage rec \
    --epochs 20 --lr 1e-4 --min-lr 1e-6 \
    --output-dir "$REC_DIR"
fi

# P2：加载 P1（或 DSER++）+ RAFT 官方权重，冻结 E2V，训练 粗光流 + 事件修正 + 选择性融合
if [[ "$START_STAGE" == "rec" || "$START_STAGE" == "flow" ]]; then
  if [[ -n "$REC_INIT" ]]; then
    FLOW_INIT=(--pretrained "$REC_INIT" --allow-partial-pretrained)
  else
    FLOW_INIT=(--pretrained "$REC_DIR/last.pth")
  fi
  [[ -f "${FLOW_INIT[1]}" ]] || { echo "找不到初始化权重: ${FLOW_INIT[1]}"; exit 1; }
  [[ -f "$RAFT_CKPT" ]] || { echo "找不到 RAFT 权重: $RAFT_CKPT"; exit 1; }

  run_logged "$LOG_DIR/p2_flow.log" \
    python train_bsergb.py "${COMMON[@]}" \
    --stage flow \
    "${FLOW_INIT[@]}" \
    --raft-ckpt "$RAFT_CKPT" --raft-iters 12 \
    --freeze-rec --raft-lr-mult 0.25 \
    --ref-mode composite --refine-iters 1 \
    --lambda-syn 1.0 --lambda-warp 0.5 \
    --epochs 20 --lr 1e-4 --min-lr 1e-6 \
    --output-dir "$FLOW_DIR"
  run_eval "$FLOW_DIR/last.pth" p2_flow
fi

# P3：加载 P2，端到端训练全部模块（纹理映射 + 多尺度精修），RAFT 与 E2V 用较小学习率
[[ -f "$FLOW_DIR/last.pth" ]] || { echo "找不到 P2 权重: $FLOW_DIR/last.pth"; exit 1; }
run_logged "$LOG_DIR/p3_full.log" \
  python train_bsergb.py "${COMMON[@]}" \
  --stage full \
  --pretrained "$FLOW_DIR/last.pth" \
  --raft-iters 12 --raft-lr-mult 0.1 --rec-lr-mult 0.1 \
  --ref-mode composite --refine-iters 1 --texture-window 3 \
  --lambda-rec 1.0 --lambda-syn 1.0 --lambda-warp 0.5 --lambda-ref 1.0 --lambda-ps 1.0 \
  --epochs 40 --lr 1e-4 --min-lr 1e-6 \
  --output-dir "$FULL_DIR"
run_eval "$FULL_DIR/last.pth" p3_full

echo "训练完成：最终权重 $FULL_DIR/last.pth"
