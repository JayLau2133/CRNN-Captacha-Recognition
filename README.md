# CRNN 验证码识别

一个基于 **CNN + 双向 LSTM + CTC** 的端到端验证码识别项目。模型直接从整张灰度图预测数字和小写字母字符串，不需要逐字符切割或字符位置标注。

本仓库在原始 Colab Notebook 的基础上完成了模块化整理，并保留原训练权重。现在可以独立执行数据生成、训练、断点续训、评估和单图预测。

> 面向面试的完整原理讲解、项目陈述和 28 组问答见 [docs/INTERVIEW.md](docs/INTERVIEW.md)。

## 原实验结果

原 Notebook 中保存的日志如下，指标为**整串完全匹配准确率**：

| 记录 | 准确率 |
|---|---:|
| Epoch 45 | 99.00% |
| Epoch 50 | 98.67% |
| 最终评估 | 98.67% |

附带的 `gig4i.png` 在原 Notebook 中的预测结果为 `gig4i`。

这些数字来自原合成验证码实验。原仓库没有完整保留实际训练/测试图片数量与数据划分，本次整理也没有重新训练复现该指标，因此不能将其理解为真实网站验证码上的泛化准确率。

## 项目结构

```text
.
├── crnn/
│   ├── config.py          # 字符集、图片尺寸和 blank 索引
│   ├── data.py            # 预处理、Dataset 和 batch 标签整理
│   ├── model.py           # CRNN 网络
│   ├── ctc.py             # 贪心解码、编辑距离和指标
│   └── engine.py          # 训练与评估循环
├── checkpoints/
│   ├── crnn_best_model.pth # 最佳模型参数，用于推理
│   └── crnn_checkpoint.pth # 完整训练状态，用于断点续训
├── dataset/predict/gig4i.png
├── docs/INTERVIEW.md
├── notebooks/CRNN.ipynb   # 原始实验记录
├── generate_data.py
├── train.py
├── evaluate.py
├── predict.py
└── requirements.txt
```

## 环境安装

建议使用 Python 3.9 或更高版本。PyTorch 请根据自己的 CUDA 环境选择合适版本；也可以直接使用 CPU 版本。

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.venv\Scripts\Activate.ps1
```

Linux/macOS：

```bash
source .venv/bin/activate
```

然后安装依赖：

```bash
python -m pip install -r requirements.txt
```

## 快速预测

![验证码样例 gig4i](dataset/predict/gig4i.png)

在仓库根目录运行：

```bash
python predict.py
```

等价于：

```bash
python predict.py \
  --image dataset/predict/gig4i.png \
  --checkpoint checkpoints/crnn_best_model.pth
```

输出：

```text
gig4i
```

可通过 `--device cpu` 或 `--device cuda` 指定设备。输入会在预测前自动转换为 `60×160` 灰度图并归一化到 `[-1, 1]`。

## 生成合成数据

```bash
python generate_data.py --num-train 1000 --num-test 200
```

默认生成：

```text
dataset/train/<label>.png
dataset/test/<label>.png
```

默认标签为 5 位数字/小写字母，图片为 `160×60`。可调整参数：

```bash
python generate_data.py \
  --output-dir dataset \
  --num-train 5000 \
  --num-test 500 \
  --length 5 \
  --width 160 \
  --height 60 \
  --seed 42
```

生成器不会覆盖同名文件，并保证本次生成及已有目录中的训练、测试标签互斥。该默认数据量仅用于快速开始，不代表原实验的数据规模。

## 训练

```bash
python train.py --data-dir dataset --epochs 50
```

常用参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--batch-size` | 128 | 训练批大小 |
| `--learning-rate` | 1e-4 | AdamW 初始学习率 |
| `--weight-decay` | 1e-2 | AdamW 解耦权重衰减 |
| `--workers` | 4 | DataLoader worker 数；Windows 可设为 0 排查多进程问题 |
| `--eval-interval` | 5 | 每隔多少轮评估并保存 checkpoint |
| `--output-dir` | checkpoints | 权重输出目录 |
| `--seed` | 42 | 随机种子 |
| `--device` | 自动选择 | `cpu` 或 `cuda` |

例如：

```bash
python train.py \
  --data-dir dataset \
  --output-dir checkpoints \
  --epochs 50 \
  --batch-size 128 \
  --learning-rate 1e-4 \
  --workers 4
```

训练使用 CTC loss、AdamW 和 `ReduceLROnPlateau`。CUDA 下启用自动混合精度，CPU 下自动关闭。每次评估都会更新完整 checkpoint，只有整串准确率提升时才覆盖最佳模型。

### 断点续训

```bash
python train.py \
  --data-dir dataset \
  --resume checkpoints/crnn_checkpoint.pth \
  --epochs 100
```

`--epochs` 表示目标总轮数，不是额外训练轮数。旧版 checkpoint 中的 `model_state`、`optimizer_state`、`scheduler_state`、`epoch` 和 `best_acc` 字段保持兼容；新版额外保存 `scaler_state`。

## 独立评估

```bash
python evaluate.py \
  --data-dir dataset/test \
  --checkpoint checkpoints/crnn_best_model.pth \
  --workers 4
```

输出包括：

- **Exact-match accuracy**：整串完全匹配准确率，越高越好。
- **Character error rate (CER)**：字符错误率，越低越好。

仅在原 `dataset/predict` 目录上评估不能代表模型性能，因为该目录只有一张演示图片。

## 模型实现

### 输入与输出约定

- 字符表：`0123456789abcdefghijklmnopqrstuvwxyz`。
- 字符索引：0～35；CTC blank 索引：36。
- 输入：`[B, 1, 60, 160]`。
- 输出：`[41, B, 37]`，依次为时间步、batch、类别。
- 模型返回 logits；训练时计算 `log_softmax`，解码时直接 `argmax`。

### 逐层尺寸

| 阶段 | 输出尺寸 |
|---|---|
| 输入 | `[B, 1, 60, 160]` |
| Conv(64) + MaxPool(2×2) | `[B, 64, 30, 80]` |
| Conv(128) + MaxPool(2×2) | `[B, 128, 15, 40]` |
| Conv(256) + Conv(256) | `[B, 256, 15, 40]` |
| MaxPool：高方向 stride=2，宽方向 stride=1 | `[B, 256, 7, 41]` |
| 合并通道与高度，再交换维度 | `[41, B, 1792]` |
| 两层双向 LSTM，隐藏单元 256 | `[41, B, 512]` |
| Linear | `[41, B, 37]` |

最后一次池化的宽方向计算为：

```text
floor((40 + 2×1 - 2) / 1 + 1) = 41
```

因此原 Notebook 中“40 个时间步”的旧注释已修正为 **41**。模型共 **6,754,981 个参数**。

### CTC 训练与解码

训练标签只有字符串，没有字符位置。CTC 对所有可折叠成目标字符串的对齐路径求概率和，并使用动态规划避免暴力枚举。

贪心解码过程：

```text
每个时间步取最大类别 → 合并相邻重复类别 → 删除 blank → 输出字符串
```

例如 `a, a, blank, b` 解码为 `ab`；而 `a, blank, a` 解码为 `aa`。blank 既表示不输出字符，也用于分隔连续重复字符。

## 权重兼容性

整理后的 `CRNN` 保留原模型的 `cnn`、`rnn`、`fc` 命名及参数形状，默认使用最佳权重对应代码中的 `dropout=0.4`。`crnn_best_model.pth` 通过 `strict=True` 加载，避免模型不匹配时静默忽略参数。

字符表、字符顺序、输入高度和预处理方式都属于权重接口的一部分。修改它们通常需要重新训练；仅改变 dropout 不影响参数形状，但会改变训练行为。

## 已知限制与改进方向

- 原实验仅验证同类合成数据，真实验证码泛化能力未知。
- 当前字符集不含大写字母和符号。
- 固定输入尺寸对应 41 个时间步；标签虽可变长，但必须能被有效时间步对齐。
- 贪心解码不保证整串概率最优，可进一步实现 prefix beam search。
- 可加入独立验证集、跨生成器测试、更多字体与噪声增广、错误样本分析和模型轻量化。

原始实验细节保留在 [notebooks/CRNN.ipynb](notebooks/CRNN.ipynb)，其中也包含未完成的探索代码；日常训练与预测请使用根目录的 Python 入口。
