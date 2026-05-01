# SymbolicLight-PoC

[Read in English](README.md)

SymbolicLight-PoC 是一个仅推理的概念验证发布包，用于展示一个脉冲语言模型架构的代码实现。本发布包包含模型定义、文本生成脚本、验证脚本、Gradio Web Demo 和预训练权重。

本发布包适合用于代码查看、本地推理和基础验证。不包含训练流程。

## 内容

```text
.
|-- LICENSE
|-- README.md
|-- README_zh-CN.md
`-- src
    |-- best.pt
    |-- generate.py
    |-- model.py
    |-- validate.py
    `-- web_demo.py
```

## 发布范围

包含：

- `src/model.py` 中的模型结构
- `src/best.pt` 预训练权重
- 命令行文本生成脚本
- TinyStories 验证脚本
- 本地 Gradio Web Demo

不包含：

- 训练脚本
- 训练数据集
- 优化器与学习率调度配置
- 分布式训练配置
- 当前权重的完整复现实验日志

## 环境依赖

建议使用 Python 3.10 或更新版本。

安装运行依赖：

```bash
pip install torch tiktoken datasets gradio
```

`validate.py` 会通过 `datasets` 下载 TinyStories 验证集。如果本地没有缓存，需要网络访问。

## 使用方法

以下命令默认在发布包根目录运行，也就是包含 `README.md` 和 `src` 的目录。

### 文本生成

单条 prompt 生成：

```bash
python src/generate.py --checkpoint src/best.pt --prompt "Once upon a time"
```

交互模式：

```bash
python src/generate.py --checkpoint src/best.pt
```

可选生成参数：

```bash
python src/generate.py --checkpoint src/best.pt --prompt "The cat" --max_tokens 100 --temperature 0.8 --top_k 50
```

### 验证

运行小规模验证：

```bash
python src/validate.py --checkpoint src/best.pt --max_samples 500 --batch_size 8
```

验证脚本会输出 loss 和 perplexity，并在验证后运行一段简短文本生成示例。

### Web Demo

启动本地 Gradio 界面：

```bash
python src/web_demo.py --checkpoint src/best.pt
```

默认地址：

```text
http://127.0.0.1:7870
```

如需更换端口：

```bash
python src/web_demo.py --checkpoint src/best.pt --port 7871
```

## 结构说明

当前实现包含以下组件：

- `SpikeEncoder`：token 与位置嵌入，随后生成 LIF 风格的脉冲表示
- `SparseTCAM`：使用 PyTorch 张量操作实现的脉冲条件路由
- `SpikingFeedForward`：中间层使用脉冲激活的前馈模块
- `EntropyGate`：基于信息熵的 early-exit 信号，当前发布配置默认关闭
- `BayesianHead`：带可学习 token prior 的输出投影
- `STDPUpdater`：用于推理实验的可选局部更新路径，默认关闭

这些组件均以标准 PyTorch 代码实现，便于阅读和本地运行。

## 权重说明

发布包中的权重通过 `torch.load` 加载。只应加载来源可信的 checkpoint。

脚本期望 checkpoint 中包含 `config` 配置项，并根据不同脚本从 `model` 或 `model_state_dict` 字段读取权重。

## 许可

本项目采用 Apache-2.0 license。具体条款见 [LICENSE](LICENSE)。
