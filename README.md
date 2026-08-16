# SymbolicLight-PoC

[Chinese README](README_zh-CN.md)

> **Historical release.** This repository preserves an early SymbolicLight
> proof of concept. It is not the implementation associated with the current
> SymbolicLight V1 paper. For the current 194M and 0.8B inference release, see
> [SymbolicLight V1](https://github.com/SymbolicLight-AGI/SymbolicLight-V1).

SymbolicLight-PoC is an inference-only snapshot of an early spiking language
model. The released checkpoint contains approximately 129.4M trainable
parameters. The implementation uses standard PyTorch operations and is intended
for code inspection, local inference, and basic validation.

## Public Release Boundary

Included:

- forward-only model definition in `src/model.py`;
- command-line text generation in `src/generate.py`;
- TinyStories validation in `src/validate.py`;
- a local Gradio demo in `src/web_demo.py`;
- a cleaned checkpoint containing only `model` and `config`, hosted separately
  on [Hugging Face](https://huggingface.co/SymbolicLight-AGI/SymbolicLight-PoC).

Not included:

- surrogate-gradient or other training implementations;
- training scripts, optimizer state, AMP scaler state, or scheduler state;
- training data, distributed-training configuration, or training logs;
- the implementation used for the current SymbolicLight V1 paper.

## Repository Layout

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

The GitHub repository intentionally excludes large checkpoint files. Download
`src/best.pt` from the Hugging Face repository before running the examples,
then verify it with `sha256sum -c CHECKSUMS_SHA256`.

See [INFERENCE_VERIFICATION.md](INFERENCE_VERIFICATION.md) for the compatibility
check performed on the cleaned checkpoint.

## Requirements

Python 3.10 or newer is recommended.

```bash
pip install torch tiktoken datasets gradio
```

`validate.py` downloads the TinyStories validation split through `datasets`, so
it requires network access unless the dataset is already cached.

## Usage

Run commands from the directory containing this README.

### Text Generation

```bash
python src/generate.py --checkpoint src/best.pt --prompt "Once upon a time"
```

Interactive mode:

```bash
python src/generate.py --checkpoint src/best.pt
```

### Validation

```bash
python src/validate.py --checkpoint src/best.pt --max_samples 500 --batch_size 8
```

### Web Demo

```bash
python src/web_demo.py --checkpoint src/best.pt
```

The local interface listens on `http://127.0.0.1:7870` by default.

## Architecture Scope

The historical PoC contains a spike encoder, a cumulative-context SparseTCAM
prototype, a spiking feed-forward block, an optional entropy-exit signal, and an
output projection with a learned token prior. It is a software reference, not a
claim of realized TCAM or neuromorphic hardware acceleration.

All checkpoint-loading entry points use `torch.load(..., weights_only=True)`.
Only load model files obtained from a trusted source and verify their published
checksum before use.

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE).
