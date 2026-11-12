#!/usr/bin/env bash
# 显存测试：每个阶段 × 每个裁剪尺寸只跑 1 个 epoch 的前 STEPS 步，记录峰值显存。
# 不加载任何权重（权重数值不影响显存），checkpoint 写到 train/memtest/，测完可以删掉。
#
# 用法（在项目根目录执行）：
#   bash tools/mem_test.sh                                  # 三个阶段 × crop 256/384/512
#   CROPS="256 448" STEPS=20 bash tools/mem_test.sh
#   STAGES="flow full" EXTRA="--amp --raft-iters 8" bash tools/mem_test.sh
#   SKIPS="1 3" bash tools/mem_test.sh                      # 与正式训练相同的 skip 组合
set -uo pipefail          # 不用 -e：某个配置 OOM 时继续测下一个
cd "$(dirname "$0")/.."

BSERGB_ROOT="${BSERGB_ROOT:-/home/yuaobo/桌面/ljh/dataset/3_TRAINING}"
CROPS="${CROPS:-256 384 512}"
STAGES="${STAGES:-rec flow full}"
STEPS="${STEPS:-20}"
SKIPS="${SKIPS:-3}"       # 默认只用 skip=3：一个窗口 3 个目标（K=3），是显存最大的情况
EXTRA="${EXTRA:-}"        # 额外参数，例如 "--amp --raft-iters 8"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

OUT="train/memtest"
mkdir -p "$OUT"
SUMMARY="$OUT/summary.txt"
: > "$SUMMARY"

for stage in $STAGES; do
  case "$stage" in
    rec)  STAGE_ARGS=(--stage rec) ;;
    flow) STAGE_ARGS=(--stage flow --freeze-rec --raft-iters 12) ;;   # 与 train_model.sh 的 P2 一致
    full) STAGE_ARGS=(--stage full --raft-iters 12) ;;                # 与 train_model.sh 的 P3 一致
    *) echo "未知阶段: $stage"; exit 1 ;;
  esac
  for crop in $CROPS; do
    log="$OUT/${stage}_${crop}.log"
    echo "=== stage=$stage crop=$crop steps=$STEPS skips=$SKIPS $EXTRA ==="
    # shellcheck disable=SC2086
    python train_bsergb.py --data-root "$BSERGB_ROOT" \
      --bins 8 --skips $SKIPS --crop-size "$crop" --mask-patch-size 32 \
      --batch-size 1 --workers 4 \
      --target-mode all --target-weighting window --lambda-tc 0.1 \
      "${STAGE_ARGS[@]}" \
      --epochs 1 --max-steps-per-epoch "$STEPS" --log-interval 1 \
      --output-dir "$OUT/ckpt_${stage}_${crop}" $EXTRA > "$log" 2>&1
    status=$?
    line=$(grep "\[GPU memory\]" "$log" | tail -1)
    if [[ $status -ne 0 ]]; then
      if grep -qi "out of memory" "$log"; then
        line="[GPU memory] stage=$stage crop=$crop -> OOM（显存不足）"
      else
        line="[GPU memory] stage=$stage crop=$crop -> 运行失败，见 $log"
      fi
    fi
    echo "$line" | tee -a "$SUMMARY"
  done
done

echo
echo "===== 汇总（$SUMMARY）====="
cat "$SUMMARY"
