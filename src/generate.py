#!/usr/bin/env python3
"""
SymbolicLight-PoC text generation script.

Load a checkpoint and run single-prompt or interactive text generation.

Usage:
  # Interactive mode, using the checkpoint next to this script
  python generate.py

  # Specify checkpoint
  python generate.py --checkpoint best.pt

  # Single prompt generation
  python generate.py --prompt "Hello world"

  # Enable experimental STDP updates
  python generate.py --enable_stdp
"""
import argparse
import sys
from pathlib import Path

import torch
import tiktoken

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = SCRIPT_DIR / "best.pt"

sys.path.insert(0, str(SCRIPT_DIR))
from model import SymbolicLightConfig, SymbolicLightModel


def parse_args():
    p = argparse.ArgumentParser(description="SymbolicLight-PoC Generator")
    p.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT),
                   help="Checkpoint path")
    p.add_argument("--prompt", type=str, default=None,
                   help="Single prompt generation mode (skip interactive chat)")
    p.add_argument("--max_tokens", type=int, default=200,
                   help="Max number of tokens to generate")
    p.add_argument("--temperature", type=float, default=0.8,
                   help="Sampling temperature (higher = more random, lower = more conservative)")
    p.add_argument("--top_k", type=int, default=50,
                   help="Top-K sampling")
    p.add_argument("--enable_stdp", action="store_true",
                   help="Enable experimental STDP updates during inference")
    p.add_argument("--save_stdp", type=str, default=None,
                   help="Save updated weights here after STDP learning")
    p.add_argument("--allow_random_init", action="store_true",
                   help="Allow random initialization when checkpoint is missing")
    p.add_argument("--trust_checkpoint_pickle", action="store_true",
                   help="Allow unsafe pickle checkpoint loading if weights_only=True fails")

    args = p.parse_args()
    if args.max_tokens < 1:
        p.error("--max_tokens must be >= 1")
    if args.temperature <= 0:
        p.error("--temperature must be > 0")
    if args.top_k < 0:
        p.error("--top_k must be >= 0")
    return args


class TiktokenWrapper:
    """tiktoken GPT-2 tokenizer wrapper."""
    def __init__(self, vocab_size=50257):
        self.vocab_size = vocab_size
        self.enc = tiktoken.get_encoding("gpt2")

    def encode(self, text: str) -> list:
        return self.enc.encode(text, allowed_special=set())

    def decode(self, ids: list) -> str:
        return self.enc.decode([int(i) for i in ids])


def _load_checkpoint(path: Path, device: torch.device, trust_pickle: bool):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception as exc:
        if not trust_pickle:
            raise RuntimeError(
                "Failed to load checkpoint with weights_only=True. "
                "If this is a trusted local checkpoint that requires pickle, "
                "rerun with --trust_checkpoint_pickle."
            ) from exc

        print("[Load] WARNING: falling back to weights_only=False for a trusted checkpoint.")
        return torch.load(path, map_location=device, weights_only=False)


def _format_metric(value) -> str:
    if value is None:
        return "?"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _select_device() -> torch.device:
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(checkpoint_path: str, enable_stdp: bool = False,
               allow_random_init: bool = False,
               trust_checkpoint_pickle: bool = False):
    """Load model and checkpoint"""
    device = _select_device()

    ckpt_path = Path(checkpoint_path).expanduser()
    if ckpt_path.exists():
        print(f"[Load] Loading checkpoint: {ckpt_path}")
        ckpt = _load_checkpoint(ckpt_path, device, trust_checkpoint_pickle)
        if not isinstance(ckpt, dict):
            raise ValueError(f"Checkpoint must be a dict, got {type(ckpt).__name__}")

        config_dict = ckpt.get("config")
        if not isinstance(config_dict, dict):
            raise KeyError("Checkpoint is missing a 'config' dictionary")

        if "model" in ckpt:
            state_dict = ckpt["model"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            raise KeyError("Checkpoint is missing model weights under 'model' or 'model_state_dict'")

        config = SymbolicLightConfig(**config_dict)
        config.enable_stdp = enable_stdp
        model = SymbolicLightModel(config).to(device)
        load_result = model.load_state_dict(state_dict, strict=False)

        if load_result.missing_keys:
            print(f"[Load] WARNING: missing keys: {load_result.missing_keys}")

        ignored_unexpected = {"spike_encoder.v_mem"}
        unexpected_keys = [
            key for key in load_result.unexpected_keys
            if key not in ignored_unexpected
        ]
        if unexpected_keys:
            print(f"[Load] WARNING: unexpected keys: {unexpected_keys}")

        step = ckpt.get("global_step", ckpt.get("step", "?"))
        loss = _format_metric(ckpt.get("best_loss", ckpt.get("loss")))
        print(f"[Load] Model loaded (step={step}, loss={loss})")
    else:
        if not allow_random_init:
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}. "
                "Pass --allow_random_init only for code smoke tests."
            )

        print(f"[Load] WARNING: checkpoint not found at {ckpt_path}")
        print("[Load] WARNING: initializing a random model for smoke testing only")
        config = SymbolicLightConfig(enable_stdp=enable_stdp)
        model = SymbolicLightModel(config).to(device)

    model.eval()
    return model, config, device


def generate_text(model, tokenizer, prompt: str, device,
                  max_tokens=200, temperature=0.8, top_k=50):
    """Generate text"""
    # Encode
    input_ids = tokenizer.encode(prompt)
    if not input_ids:
        raise ValueError("Prompt must contain at least one token")

    vocab_size = getattr(getattr(model, "config", None), "vocab_size", None)
    if vocab_size:
        invalid_ids = [token_id for token_id in input_ids if token_id < 0 or token_id >= vocab_size]
        if invalid_ids:
            sample = invalid_ids[:5]
            raise ValueError(
                f"Prompt contains token IDs outside model vocab_size={vocab_size}: {sample}"
            )

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    # Generate
    effective_top_k = min(top_k, vocab_size) if top_k > 0 and vocab_size else top_k

    with torch.no_grad():
        output_ids = model.generate(
            input_tensor,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=effective_top_k,
        )

    # Only keep newly generated part
    new_ids = output_ids[0, len(input_ids):].tolist()
    generated_text = tokenizer.decode(new_ids)

    # Calculate sparsity
    with torch.no_grad():
        test_input = input_tensor[:, :min(32, input_tensor.size(1))]
        spikes, _ = model.spike_encoder(test_input)
        sparsity = 1.0 - spikes.mean().item()

    return generated_text, sparsity


def interactive_chat(model, tokenizer, device, args):
    """Interactive chat"""
    print("\n" + "=" * 60)
    print(" SymbolicLight Interactive Chat")
    print("=" * 60)
    print(f"  Temperature: {args.temperature}")
    print(f"  Max tokens:  {args.max_tokens}")
    print(f"  STDP Learn:  {'ON' if args.enable_stdp else 'OFF'}")
    print(f"  Device:      {device}")
    print("-" * 60)
    print("  Type your message and press Enter.")
    print("  Type 'quit' to exit.")
    print("  Type 'sparsity' to see network sparsity stats.")
    if args.enable_stdp:
        print("  Type 'save' to save STDP-updated weights.")
    print("=" * 60 + "\n")

    conversation_history = ""
    turn = 0

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        if user_input.lower() == 'quit':
            print("Bye!")
            break

        if user_input.lower() == 'sparsity':
            try:
                stats = model.get_sparsity_stats()
            except Exception as exc:
                print(f"\n[Sparsity Stats] unavailable: {exc}\n")
                continue

            print("\n[Sparsity Stats]")
            for k, v in stats.items():
                print(f"  {k}: {v*100:.1f}% silent")
            print()
            continue

        if user_input.lower() == 'save' and args.enable_stdp:
            save_path = Path(args.save_stdp) if args.save_stdp else Path(args.checkpoint).with_name("stdp_updated.pt")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model": model.state_dict(),
                "config": model.config.__dict__,
            }, save_path)
            print(f"[STDP] Weights saved to {save_path}\n")
            continue

        # Build context
        turn += 1
        conversation_history += f"{user_input} "
        prompt = conversation_history

        # Generate
        try:
            response, sparsity = generate_text(
                model, tokenizer, prompt, device,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
        except Exception as exc:
            print(f"[Error] {exc}\n")
            continue

        # Update history
        conversation_history += f"{response} "

        # Display
        print(f"SymbolicLight: {response}")
        print(f"  [sparsity: {sparsity*100:.1f}% | "
              f"stdp: {'learning' if args.enable_stdp else 'off'}]\n")


def main():
    args = parse_args()

    # Load model
    try:
        model, config, device = load_model(
            args.checkpoint,
            enable_stdp=args.enable_stdp,
            allow_random_init=args.allow_random_init,
            trust_checkpoint_pickle=args.trust_checkpoint_pickle,
        )
    except Exception as exc:
        print(f"[Error] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    # Initialize tokenizer
    tokenizer = TiktokenWrapper(config.vocab_size)

    if args.prompt:
        # Single prompt generation mode
        print(f"\nPrompt: {args.prompt}")
        try:
            response, sparsity = generate_text(
                model, tokenizer, args.prompt, device,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
        except Exception as exc:
            print(f"[Error] {exc}", file=sys.stderr)
            raise SystemExit(1) from exc

        print(f"Response: {response}")
        print(f"Sparsity: {sparsity*100:.1f}%")
    else:
        # Interactive chat mode
        interactive_chat(model, tokenizer, device, args)


if __name__ == "__main__":
    main()
