# SymbolicLight-PoC

[Chinese README](README_zh-CN.md)

SymbolicLight-PoC is an inference-only proof-of-concept release for a spiking language model architecture. It includes the model definition, generation and validation entry points, a Gradio demo, and a pretrained checkpoint.

This package is intended for code inspection, local inference, and basic validation. It does not include the training pipeline.

## Contents

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

## Release Scope

Included:

- Model architecture in `src/model.py`
- Pretrained checkpoint at `src/best.pt`
- Command-line text generation script
- TinyStories validation script
- Local Gradio web demo

Not included:

- Training script
- Training dataset
- Optimizer and scheduler configuration
- Distributed training setup
- Reproduction logs for the released checkpoint

## Requirements

Python 3.10 or newer is recommended.

Install the runtime dependencies:

```bash
pip install torch tiktoken datasets gradio
```

`validate.py` downloads the TinyStories validation split through `datasets`, so it requires network access unless the dataset is already cached.

## Usage

Run commands from the package root directory, the directory that contains `README.md` and `src`.

### Text Generation

Single prompt mode:

```bash
python src/generate.py --checkpoint src/best.pt --prompt "Once upon a time"
```

Interactive mode:

```bash
python src/generate.py --checkpoint src/best.pt
```

Optional generation parameters:

```bash
python src/generate.py --checkpoint src/best.pt --prompt "The cat" --max_tokens 100 --temperature 0.8 --top_k 50
```

### Validation

Run a small validation pass:

```bash
python src/validate.py --checkpoint src/best.pt --max_samples 500 --batch_size 8
```

The validation script reports loss and perplexity. It also runs a short text generation demo after validation.

### Web Demo

Start the local Gradio interface:

```bash
python src/web_demo.py --checkpoint src/best.pt
```

The default address is:

```text
http://127.0.0.1:7870
```

To use another port:

```bash
python src/web_demo.py --checkpoint src/best.pt --port 7871
```

## Architecture Summary

The implementation contains the following components:

- `SpikeEncoder`: token and position embeddings followed by LIF-style spike generation
- `SparseTCAM`: spike-conditioned routing implemented with PyTorch tensor operations
- `SpikingFeedForward`: feed-forward block with spike activation in the intermediate layer
- `EntropyGate`: entropy-based early-exit signal, disabled by default in the released configuration
- `BayesianHead`: output projection with a learned token prior
- `STDPUpdater`: optional local update path for inference experiments, disabled by default

These components are implemented in standard PyTorch for inspection and local execution.

## Checkpoint Notes

The included checkpoint is loaded with `torch.load`. Only load checkpoints from trusted sources.

The scripts accept checkpoints with a `config` entry and model weights under either `model` or `model_state_dict`, depending on the script.

## License

This project is licensed under the Apache-2.0 license. See [LICENSE](LICENSE) for details.
