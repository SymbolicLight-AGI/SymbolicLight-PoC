#!/usr/bin/env python3
"""
SymbolicLight — Inference and Chat Script
================================
Load a trained checkpoint and chat interactively with the SymbolicLight model.

Usage:
  # Interactive Chat
  python generate.py

  # Specify checkpoint
  python generate.py --checkpoint ./checkpoints/best.pt

  # Single sentence generation
  python generate.py --prompt "Hello world"

  # Enable STDP online learning (model will remember what you say)
  python generate.py --enable_stdp
"""
import sys
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from model import SymbolicLightConfig, SymbolicLightModel


def parse_args():
    p = argparse.ArgumentParser(description="SymbolicLight Generator")
    p.add_argument("--checkpoint", type=str, default="./checkpoints/best.pt",
                   help="Checkpoint path")
    p.add_argument("--prompt", type=str, default=None,
                   help="Single sentence generation mode (skip interactive chat)")
    p.add_argument("--max_tokens", type=int, default=200,
                   help="Max number of tokens to generate")
    p.add_argument("--temperature", type=float, default=0.8,
                   help="Sampling temperature (higher = more random, lower = more conservative)")
    p.add_argument("--top_k", type=int, default=50,
                   help="Top-K sampling")
    p.add_argument("--enable_stdp", action="store_true",
                   help="Enable STDP online learning (model will learn from chat)")
    p.add_argument("--save_stdp", type=str, default=None,
                   help="Save updated weights here after STDP learning")
    return p.parse_args()


import tiktoken


class TiktokenWrapper:
    """tiktoken GPT-2 tokenizer wrapper (consistent with train.py)"""
    def __init__(self, vocab_size=50257):
        self.vocab_size = vocab_size
        self.enc = tiktoken.get_encoding("gpt2")

    def encode(self, text: str) -> list:
        return self.enc.encode(text, allowed_special=set())

    def decode(self, ids: list) -> str:
        # Filter out padding (0)
        ids = [i for i in ids if i > 0]
        return self.enc.decode(ids)


def load_model(checkpoint_path: str, enable_stdp: bool = False):
    """Load model and checkpoint"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(checkpoint_path)
    if ckpt_path.exists():
        print(f"[Load] Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        config_dict = ckpt["config"]
        config = SymbolicLightConfig(**config_dict)
        config.enable_stdp = enable_stdp
        model = SymbolicLightModel(config).to(device)
        model.load_state_dict(ckpt["model"], strict=False)
        print(f"[Load] Model loaded (step={ckpt.get('global_step', '?')}, "
              f"loss={ckpt.get('best_loss', '?'):.4f})")
    else:
        print(f"[Load] No checkpoint found at {ckpt_path}")
        print(f"[Load] Initializing random model (for testing only)")
        config = SymbolicLightConfig(enable_stdp=enable_stdp)
        model = SymbolicLightModel(config).to(device)

    model.eval()
    return model, config, device


def generate_text(model, tokenizer, prompt: str, device,
                  max_tokens=200, temperature=0.8, top_k=50):
    """Generate text"""
    # Encode
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_tensor,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
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
            stats = model.get_sparsity_stats()
            print("\n[Sparsity Stats]")
            for k, v in stats.items():
                print(f"  {k}: {v*100:.1f}% silent")
            print()
            continue

        if user_input.lower() == 'save' and args.enable_stdp:
            save_path = args.save_stdp or "./checkpoints/stdp_updated.pt"
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
        response, sparsity = generate_text(
            model, tokenizer, prompt, device,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )

        # Update history
        conversation_history += f"{response} "

        # Display
        print(f"SymbolicLight: {response}")
        print(f"  [sparsity: {sparsity*100:.1f}% | "
              f"stdp: {'learning' if args.enable_stdp else 'off'}]\n")


def main():
    args = parse_args()

    # Load model
    model, config, device = load_model(args.checkpoint, args.enable_stdp)

    # Initialize tokenizer
    tokenizer = TiktokenWrapper(config.vocab_size)

    if args.prompt:
        # Single sentence generationmode
        print(f"\nPrompt: {args.prompt}")
        response, sparsity = generate_text(
            model, tokenizer, args.prompt, device,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        print(f"Response: {response}")
        print(f"Sparsity: {sparsity*100:.1f}%")
    else:
        # Interactive Chatmode
        interactive_chat(model, tokenizer, device, args)


if __name__ == "__main__":
    main()
