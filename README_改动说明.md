# 事件参考引导的光流插帧：代码评审与补全说明

## 1. 方案是否合理

整体思路是成立的，各步骤都有对应的已发表工作支撑：

| 你的步骤 | 相近的已有做法 | 评价 |
|---|---|---|
| 用事件做 E2V，得到 t 时刻灰度图；用灰度 GT 监督 | DSER++ / Direct Synthesis 的事件参考帧 | 合理；可直接加载 DSER++ 的 `EventRecNet` 权重 |
| 在 E2V 图与两端灰度帧之间建代价体，估计粗光流 | CBMNet 的"t 时刻锚特征 + 代价体" | 合理。区别是 CBMNet 的锚在特征域，你的锚在图像域。好处是可以复用 RAFT 预训练权重 |
| 粗光流 + 事件 → 精修光流 | Time Lens / TimeReplayer 的残差细化 | 合理 |
| 双向后向变形 + 选择性融合 | 常规做法 | 合理 |
| 事件帧确定运动区域（DSER++ 的 HCS） | DSER++ | 合理，但阈值有问题，见 1.2 |
| 沿光流从两端选择纹理（VTinker） | VTinker 纹理映射 | 合理；改成了可导的软选择 |

### 1.1 需要特别注意的四个问题

1. **光流方向写反了（必须改）。**
   - `self.raft(gray_img0, pure_rec)` 得到的是 F_{0→t}，定义在 I₀ 的像素网格上。
   - 用后向变形 `backwarp(I₀, F)` 生成 t 时刻的帧，需要的是定义在 t 网格上的 F_{t→0}，也就是 `RAFT(ref_t, gray0)`。
   - 已改为一次 RAFT 调用、在 batch 维拼接两个方向，同时输出 F_{t→0} 和 F_{t→1}。
2. **E2V 图只在事件掩码内可靠。**
   - 重建损失是 `rec * mask` 对 `gray_gt * mask`，掩码外的输出完全不受约束。
   - 纯事件本身也无法恢复绝对亮度，所以 E2V 图和真实灰度帧之间存在光度差。
   - 代价体在无事件区域因此不可靠。已提供两种处理：
     - 默认 `--ref-mode composite`：掩码内用 E2V 图，掩码外用两端灰度帧的均值（没有事件意味着亮度没有变化），掩码边界做平滑过渡；
     - 可选 `--rec-use-frames`：把灰度 I₀ / I₁ 也输入 E2V，让它输出完整、亮度正确的参考图。注意开启后 E2V 第一层的通道数变化，不能再严格加载 DSER++ 的权重。
3. **误差会传播。** E2V 的误差会依次影响代价体和光流。TimeTracker 对 TimeLens-XL 的批评"合成帧误差传播到光流"在这里同样适用。因此保留了：
   - 事件残差细化；
   - 对 RAFT 每次迭代输出和精修光流都施加光度损失（`--lambda-warp`）；
   - 分阶段训练（见第 4 节）。
4. **每个 t 都要重算光流。** 每个时刻要跑一次 RAFT（双向在 batch 维合并），属于"两段式"方法。多帧插值时成本随帧数线性增长。对 BS-ERGB 的 ×2 / ×4 没问题；如果以后要做 ×16 以上，需要考虑 OF-AF 那样的一次估计方案。

### 1.2 关于 DSER++ HCS 掩码的两点发现

- `pyramid_patch_mask` 的阈值是 **patch 内差值之和 ≥ beta**。32×32×3 的绝对差之和几乎总是远大于 0.1，所以 beta = 0.1 时，HCS 基本等于"含有事件的 patch"。
  - 已新增 `--hcs-beta-mode mean`：改为与 patch 的平均差比较，阈值与 patch 大小无关。例如可以取 `--beta 0.05`。
  - 默认仍为 `sum`，与 DSER++ 完全一致。`tools/smoke_test.py` 会对比验证两者的输出相同。
- `mask_reverse` 只在两张掩码不同的位置交换。逐元素分析可知，这等价于**把两张掩码整体互换**。代码中按这个等价形式实现，并保留了注释。

### 1.3 关于 VTinker

- VTinker 用 2×2 纹理块、3×3 局部搜索，在两帧共 18 个候选中做**硬 argmax** 选择，这一步不可导。
- 这里的实现改成 softmax 软注意力，并加入由遮挡图得到的可见性先验 log(vis)。
- 候选纹理用精修后的 F_{t→k} 做后向变形对齐。纹理特征**只在 HCS 运动区域内**注入解码器。
- VTinker 仓库的许可证（Pi-Lab 1.0）不是标准开源许可证，因此**没有复制它的代码**，`model/texture.py` 是按其算法思路独立实现的。

---

## 2. 原代码中的错误与修复

| 位置 | 问题 | 修复 |
|---|---|---|
| `model/blocks.py` | 依赖的 `model/submodules.py` 缺失 | 按 DSER++（Apache-2.0）恢复，定义保持一致，并补上 `import torch`（blocks.py 依赖星号导入得到 `torch`） |
| `model/ourNet.py` | `from raft import *` 路径错误 | 改为 `from model.raft import RAFT` 等 |
| `model/raft.py`、`utils/corr.py` | `from utils.update ...` 是绝对路径，作为包导入时会失败 | 改成相对导入 `.utils...` |
| `model/ourNet.py` | `RAFT()` 没有传 `args` | 新增 `default_raft_args()`，`RAFT(args=None)` 时使用默认配置 |
| RAFT 输入 | 编码器要求 3 通道、取值 [0, 255]；灰度图是 1 通道、[0, 1] | 灰度复制成 3 通道；新增 `input_scale=1.0`，归一化后的范围与原版相同，预训练权重可直接使用 |
| RAFT 输出 | 返回的是每次迭代结果组成的 list | 取最后一次作为粗光流，全部迭代用于序列光度损失 |
| RAFT 的 BatchNorm | batch = 1 时统计量无意义 | 训练时冻结 BN（与 RAFT 官方微调一致），可用 `--no-freeze-raft-bn` 关闭 |
| 光流方向 | 见 1.1 第 1 条 | 改为 F_{t→0} / F_{t→1} |
| `args.stage` | 训练脚本没有这个参数，会报 AttributeError | 新增 `--stage {rec, flow, full}` |
| `util/voxelization.py` | `np.reshape(newshape=...)` 在 NumPy ≥ 2.1 已移除，**会导致数据加载崩溃** | 改成 tensor 的 `.reshape` |
| `util/*.py` | `np.int` / `np.bool` 在新版 NumPy 中已移除 | 改为 `int` / `bool` |
| LPIPS | 输入是 OpenCV 的 BGR 顺序，而 LPIPS 按 RGB 训练 | 计算前交换通道 |
| 未使用的模块 | `encoder_f / encoder_e / encoder_ref` 没有被调用 | 删除；同类编码器在纹理模块和解码器中重新使用 |

---

## 3. 补全后的数据流

输入：`imgs` (B, 9, H, W)，为 BGR、[0, 1]，其中中间 3 个通道是 GT，前向时不使用；`voxels` = [E₀→t, E_t→1, E₁→t]；`mask` 为事件掩码；`t` 为归一化目标时刻。

```text
v0t, vt1 ──► EventRecNet ──► pure_rec (B,1,H,W) ──×mask──► rec           [L_rec: 灰度GT*mask, L1+LPIPS]
                                │
            composite: soft(mask)·rec + (1-soft)·(g0+g1)/2 = ref_t
                                │
ref_t ↔ gray(I0), gray(I1) ──► RAFT (双向在 batch 维合并) ──► F_t→0^c, F_t→1^c   [L_warp: 每次迭代, γ=0.8]
                                │
E_t→0 = reverse_voxel(v0t), E_t→1 = vt1
coarse F + warp(I_k) + ref + |err| + E_t→k + mask ──► EventFlowRefiner (两方向共享权重)
                                │                   ──► F_t→k (精修), occ_k
backwarp(I0,F_t→0)=W0, backwarp(I1,F_t→1)=W1 ──► SelectiveFusion (DenseBlock→M)
                                │                   ──► coarse = M·W0 + (1-M)·W1   [L_syn, L_warp(min)]
coarse vs I0 / I1 + 事件 patch ──► HCS 掩码 (DSER++) ──► hcs_union (full / 1/2 / 1/4)
                                │
I0, I1 特征沿 F_t→k 后向变形 ──► 3×3 局部搜索 × 2 帧 = 18 个候选，软选择 2×2 纹理块
                                │                   ──► texture (1/2 分辨率, 只在 HCS 内注入)
coarse + W0 + W1 + 事件特征 + texture ──► TextureRefineDecoder ──► [pred_full, pred_half, pred_quarter]
                                                         [L_ref: 多尺度 L1 + LPIPS;  L_ps: HCS 内的 SSIM]
```

`forward` 返回值与 DSER++ 保持相同的 6 元组：
`(pure_rec_3ch, rec_3ch, coarse, predictions, hcs_union, aux)`。`aux` 中包含粗光流和精修光流、遮挡图、两张变形帧、融合掩码、HCS 等，便于可视化。`stage == "rec"` 时只返回前两项。

### 新增 / 修改的文件

- `model/submodules.py`（恢复）
- `model/__init__.py`（新增，空文件）
- `model/warp.py`（新增）：后向变形、光流缩放、voxel 时间反转、边缘感知平滑
- `model/flow_modules.py`（新增）：`CoarseFlowEstimator`（RAFT 封装，可加载 raft-things.pth）、`EventFlowRefiner`、`SelectiveFusion`
- `model/texture.py`（新增）：`hcs_patch_masks`、`FlowGuidedTextureMapper`、`TextureRefineDecoder`
- `model/ourNet.py`（补全）
- `model/raft.py`、`model/utils/corr.py`、`model/utils/utils.py`（修复导入和参数）
- `train_bsergb.py`：新增分阶段训练、光度 / 平滑损失、分组学习率、冻结选项，并把 `t` 传入网络
- `datasets/bsergb.py`：返回值中新增 `t`
- `model/ourModel.py`：`inference(..., t=None)`
- `util/voxelization.py`、`util/event.py`、`util/utils_func.py`：NumPy 兼容性修复
- `tools/smoke_test.py`（新增）

---

## 4. 训练流程（建议分三个阶段）

```bash
# 0) 先跑冒烟测试（CPU 即可，不需要任何权重）
python tools/smoke_test.py

# 1) E2V 阶段（如果已有 DSER++ 的权重，可以跳过，见下方说明）
python train_bsergb.py --data-root /path/BSERGB/3_TRAINING --stage rec --epochs 20 --output-dir train/rec

# 2) 光流阶段：加载阶段 1 的权重和 RAFT 官方权重，冻结 E2V
python train_bsergb.py --data-root ... --stage flow \
    --pretrained train/rec/last.pth --raft-ckpt raft-things.pth --freeze-rec \
    --epochs 20 --output-dir train/flow

# 3) 端到端阶段
python train_bsergb.py --data-root ... --stage full --pretrained train/flow/last.pth \
    --raft-lr-mult 0.1 --rec-lr-mult 0.1 --epochs 40 --output-dir train/full
```

- RAFT 权重使用官方的 `raft-things.pth`（princeton-vl/RAFT）。所有阶段的网络结构完全相同，因此可以用 `--pretrained` 严格加载上一阶段的权重。`--raft-ckpt` 在 `--pretrained` 之后加载，不会被阶段 1 中未训练的 RAFT 权重覆盖。
- 用 DSER++ 的权重初始化 E2V：`--pretrained dser.pth --allow-partial-pretrained`。这只会加载 `event_rec.*`，其余键会被忽略。
- `--resume` 时请保持冻结相关参数不变，因为参数分组需要一致。

---

## 5. 未在本环境验证的部分（请先确认）

- 本次的运行环境无法安装 PyTorch（软件包源被网络策略拦截）。因此只做了语法编译和跨模块导入的静态检查，**没有实际运行前向和反向传播**。请先在你的环境中运行 `python tools/smoke_test.py`。它会检查：
  - 后向变形的光流方向约定；
  - `reverse_voxel` 与对反转事件流做体素化的结果是否一致；
  - HCS 掩码与 DSER++ 原实现是否逐元素相同；
  - 三个阶段的前向、反向传播和输出形状；
  - 任意尺寸输入时的 padding / unpad。
- 超参数（`flow_norm = 20`、细化网络通道数、`texture_window = 3`、各损失权重）都是经验初值，没有调参。
- 显存：训练时 256 的裁剪尺寸没有问题。全分辨率推理（970×625，padding 到 1024×640）时，RAFT 的全对相关体大约需要 1 GB 量级的显存。

## 6. 建议的消融实验

- 参考图：`--ref-mode rec` vs `composite`，以及加不加 `--rec-use-frames`。
- 有无事件细化：`--refine-iters 0`（直接使用 RAFT 的粗光流，遮挡 logit 为 0）。
- 有无纹理模块：用 `stage flow` 的结果作为对照。
- HCS 阈值：`--hcs-beta-mode sum --beta 0.1`（DSER++ 原设置）vs `mean --beta 0.03 / 0.05`。
- `--detach-rec-for-flow`：E2V 是否接收来自光流和合成损失的梯度。

---

## 7. 多目标训练：skip=3 的窗口一次生成 3 张（新增）

**数据**（`datasets/bsergb.py`，默认 `--target-mode all`）

- 每个 skip=N 窗口是一个样本，返回全部 N 个中间时刻：
  - `endpoints` (6,H,W)
  - `gts` (K,3,H,W)
  - `voxels` (K,3·bins,H,W)
  - `masks` (K,1,H,W)
  - `t` (K,)

  其中 K = N，按时间升序排列。skip=3 时 t = 0.25 / 0.5 / 0.75。
- 窗口内每个区间的事件文件只读一次，再按各目标帧的位置切分成 E₀→t / E_t→1。同一窗口的所有目标共用同一个随机裁剪和同一套数据增强。
- `--target-mode random`：恢复旧行为，每次只随机选一个目标（K=1）。
- `SkipGroupedBatchSampler`：保证同一个 batch 内的窗口 K 相同（skip=1 与 skip=3 不混在一起），batch 的顺序每个 epoch 重新打乱。batch_size=1 时等价于普通的随机打乱。

**训练**（`train_bsergb.py`）

- `flatten_targets` 把 (B, K, …) 展开成 (B·K, …)，仍然拼成 `[I₀, GT_t, I₁]` 的 9 通道输入，所以 `MyNet` 和原有损失函数都不需要改。一次前向就得到 K 个结果：
  - 第一阶段：K 张 E2V 灰度图；
  - 第二阶段：K 张粗插帧结果；
  - 第三阶段：K 张最终预测。
- `--target-weighting window`（默认）：每个窗口权重相同，损失先对 K 个目标求平均。`frame`：每一帧权重相同，skip=3 窗口的权重是 skip=1 的 3 倍。
- `--lambda-tc`：同一窗口内相邻目标帧之间的时间一致性损失，|(P_{k+1} − P_k) − (G_{k+1} − G_k)|。它不依赖光流，惩罚误差随时间跳变（闪烁）。第一阶段在两帧的事件掩码交集内计算，第二阶段用粗插帧结果，第三阶段用最终预测。K = 1 时自动为 0。代码默认值为 0，`train_model.sh` 中设为 0.1。
- 进度条改为显示各项损失的**均值**（原来显示的是累加和）；每个 epoch 结束时打印窗口数和目标帧数。

**推理**：新增 `Model.inference_window(endpoints, voxels, masks, t, chunk=None)`，一次返回一个窗口的全部 K 帧，形状为 (B,K,3,H,W)。显存不够时可以用 `chunk` 分批计算。

**代价**：K 个目标目前直接展开到 batch 维计算，两端帧相关的计算没有在目标之间共享。所以一个 skip=3 窗口的计算量和显存约为 skip=1 的 3 倍（RAFT 的等效 batch 为 2K）。如果显存不足，可以减小 `--crop-size` 或 `--raft-iters`。

---

## 8. 环境配置

```bash
bash setup_env.sh                                   # 默认：conda 环境 evfi、Python 3.10、torch 2.5.1 + CUDA 12.1
CUDA_TAG=cu118 bash setup_env.sh                    # nvidia-smi 显示的 CUDA Version 低于 12.1 时
PIP_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple bash setup_env.sh   # 普通依赖走国内镜像
conda activate evfi && python tools/check_env.py    # 之后随时可以重新检查环境
```

- `requirements.txt` 不包含 torch 和 torchvision，因为这两个包必须按显卡驱动选择对应的 CUDA 版本，由 `setup_env.sh` 单独安装。CUDA_TAG 不能高于 nvidia-smi 右上角显示的 CUDA Version。
- RAFT 权重会自动下载到 `checkpoints/raft-things.pth`，使用的是 RAFT 官方 download_models.sh 中的 Dropbox 链接。如果下载失败，请手动从 RAFT README 提供的 Google Drive 下载，再放到这个位置。
- `tools/check_env.py` 会检查以下内容：
  - 各依赖库的版本，以及 CUDA 和显卡信息；
  - 代码依赖的几个 API：DeformConv2d、pixel_unshuffle、无 batch 维的 conv2d；
  - 项目各模块能否正常导入；
  - NumPy 兼容性；
  - LPIPS 权重是否可用；
  - RAFT 权重能否严格加载进项目里的 RAFT。
