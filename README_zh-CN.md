# SymbolicLight-PoC

[English README](README.md)

> **历史版本。** 本仓库保留 SymbolicLight 早期概念验证原型，不是当前
> SymbolicLight V1 论文对应的实现。当前 194M 和 0.8B 的仅推理发布请见
> [SymbolicLight V1](https://github.com/SymbolicLight-AGI/SymbolicLight-V1)。

SymbolicLight-PoC 是一个早期脉冲语言模型的仅推理快照。公开 checkpoint
约含 1.294 亿个可训练参数。实现使用标准 PyTorch 操作，用于代码查看、
本地推理和基础验证。

## 公开范围

包含：

- `src/model.py` 中的 forward-only 模型定义；
- `src/generate.py` 命令行文本生成；
- `src/validate.py` TinyStories 验证；
- `src/web_demo.py` 本地 Gradio Demo；
- 仅包含 `model` 和 `config` 的干净 checkpoint，单独托管在
  [Hugging Face](https://huggingface.co/SymbolicLight-AGI/SymbolicLight-PoC)。

不包含：

- 代理梯度或其他训练实现；
- 训练脚本、优化器状态、AMP scaler 状态或 scheduler 状态；
- 训练数据、分布式训练配置或训练日志；
- 当前 SymbolicLight V1 论文使用的实现。

## 仓库结构

```text
.
|-- LICENSE
|-- README.md
|-- README_zh-CN.md
`-- src
    |-- generate.py
    |-- model.py
    |-- validate.py
    `-- web_demo.py
```

GitHub 仓库有意不包含大型 checkpoint。运行示例前，请从 Hugging Face 仓库
下载 `src/best.pt`，然后运行 `sha256sum -c CHECKSUMS_SHA256` 核对文件。

干净 checkpoint 的兼容性检查结果见
[INFERENCE_VERIFICATION.md](INFERENCE_VERIFICATION.md)。

## 环境依赖

建议使用 Python 3.10 或更新版本。

```bash
pip install torch tiktoken datasets gradio
```

`validate.py` 会通过 `datasets` 下载 TinyStories 验证集。如果本地没有缓存，
需要网络连接。

## 使用方法

以下命令均在包含本 README 的目录中运行。

### 文本生成

```bash
python src/generate.py --checkpoint src/best.pt --prompt "Once upon a time"
```

交互模式：

```bash
python src/generate.py --checkpoint src/best.pt
```

### 验证

```bash
python src/validate.py --checkpoint src/best.pt --max_samples 500 --batch_size 8
```

### Web Demo

```bash
python src/web_demo.py --checkpoint src/best.pt
```

默认本地地址为 `http://127.0.0.1:7870`。

## 架构范围

该历史 PoC 包含脉冲编码器、累积上下文 SparseTCAM 原型、脉冲前馈模块、可选的
熵退出信号，以及带可学习 token prior 的输出投影。它是软件参考实现，不代表已实现
TCAM 或神经形态硬件加速。

所有 checkpoint 加载入口都使用 `torch.load(..., weights_only=True)`。只应加载信任
来源的模型文件，并在使用前核对公布的校验值。

## 许可

本项目采用 Apache License 2.0，具体条款见 [LICENSE](LICENSE)。
