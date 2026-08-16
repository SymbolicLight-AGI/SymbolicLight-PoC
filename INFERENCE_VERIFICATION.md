# Inference Verification

The cleaned historical checkpoint was verified with Python 3.12 and
PyTorch 2.9.1+cpu.

- SHA-256: `0f68b22e79339f0506cdd0bb707a860de59c7a3a64351768542b030fe71b6aa4`
- checkpoint fields: `model`, `config`
- state-dict entries: 126
- trainable parameters: 129,393,757
- checkpoint loading: strict, with no missing or unexpected keys
- fixed input token IDs: `[[7454, 2402, 257, 640]]`
- output shape: `[1, 4, 50257]`
- output SHA-256: `11a30b95ed54810e383b963a6f2b5a39b6b251064267d5c54aafec978ed71c22`
- maximum absolute logits difference from the pre-cleanup forward
  implementation: `0.0`
- model parameters changed by the inference pass: no

The public checkpoint excludes optimizer state, AMP scaler state, progress
fields, loss metadata, and the runtime membrane buffer. All published loaders
use `torch.load(..., weights_only=True)`.
