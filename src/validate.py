#!/usr/bin/env python3
"""
SymbolicLight-PoC — Inference Validation
==========================================
Evaluate the historical checkpoint on the TinyStories validation set.

Metrics:
  1. Validation Loss / Perplexity
  2. Optional text generation demo

Usage:
  python validate.py
  python validate.py --checkpoint best.pt
  python validate.py --generate --prompt "Once upon a time"
"""

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from model import SymbolicLightConfig, SymbolicLightModel

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "best.pt"


def parse_args():
    p = argparse.ArgumentParser(description="SymbolicLight Validation")
    p.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT),
                   help="Model checkpoint path")
    p.add_argument("--max_samples", type=int, default=5000,
                   help="Maximum number of validation samples (to reduce wait time)")
    p.add_argument("--batch_size", type=int, default=16,
                   help="Validation batch size")
    p.add_argument("--seq_len", type=int, default=256,
                   help="Sequence length")
    p.add_argument("--generate", action="store_true",
                   help="Whether to run text generation demo")
    p.add_argument("--prompt", type=str, default="Once upon a time",
                   help="Prompt for generation")
    p.add_argument("--max_new_tokens", type=int, default=200,
                   help="Maximum number of generated tokens")
    p.add_argument("--temperature", type=float, default=0.8,
                   help="Generation temperature")
    p.add_argument("--top_k", type=int, default=50,
                   help="Top-K sampling")
    return p.parse_args()


def load_model(checkpoint_path, device):
    """Load model and checkpoint"""
    print(f"[Model] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Checkpoint must be a dict, got {type(ckpt).__name__}")

    # Restore configuration from checkpoint
    cfg_dict = ckpt.get("config")
    if not isinstance(cfg_dict, dict):
        raise KeyError("Checkpoint is missing a 'config' dictionary")
    config = SymbolicLightConfig(**cfg_dict)
    config.enable_entropy_exit = False
    print("[Model] Config loaded from checkpoint")

    model = SymbolicLightModel(config)

    # Load the model-only state dictionary.
    if "model" in ckpt:
        state_dict = ckpt["model"]
    elif "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        raise KeyError("Checkpoint is missing model weights under 'model' or 'model_state_dict'")
    model.load_state_dict(state_dict, strict=True)

    model = model.to(device)
    model.eval()

    # Disable EntropyGate early exit during validation.
    # EntropyGate causes exit at layer 0 in eval mode, must be disabled for fair evaluation
    for block in model.blocks:
        block.entropy_gate.threshold = 0.0  # Do not early exit
    print("[Model] Disabled entropy gate early exit for validation")

    # Print model information
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    return model, config


def load_validation_data(seq_len, max_samples):
    """Load TinyStories validation set"""
    import tiktoken
    from datasets import load_dataset

    enc = tiktoken.get_encoding("gpt2")
    print(f"[Data] Loading TinyStories (validation) from HuggingFace...")
    ds = load_dataset("roneneldan/TinyStories", split="validation")
    print(f"[Data] Loaded {len(ds):,} validation stories")

    # Tokenize
    print(f"[Data] Tokenizing...")
    all_tokens = []
    for i, example in enumerate(ds):
        text = example.get("text", "")
        tokens = enc.encode(text, allowed_special=set())
        all_tokens.extend(tokens)
        if len(all_tokens) > max_samples * seq_len * 2:
            break  # Enough
        if (i + 1) % 50000 == 0:
            print(f"  ... tokenized {i+1:,} stories ({len(all_tokens):,} tokens)")

    n_samples = min(max_samples, (len(all_tokens) - 1) // seq_len)
    print(f"[Data] Total: {len(all_tokens):,} tokens, {n_samples:,} validation samples")

    # Convert to tensor
    tokens_tensor = torch.tensor(all_tokens[:n_samples * seq_len + 1], dtype=torch.long)

    return tokens_tensor, n_samples, enc


@torch.no_grad()
def validate(model, tokens_tensor, n_samples, seq_len, batch_size, device):
    """Calculate loss and perplexity on the validation set."""
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    n_batches = 0

    print(f"\n{'='*60}")
    print(f"  VALIDATION ({n_samples:,} samples, batch_size={batch_size})")
    print(f"{'='*60}")

    start_time = time.time()

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        actual_bs = end_idx - start_idx

        # Construct batch
        x_list = []
        y_list = []
        for i in range(start_idx, end_idx):
            offset = i * seq_len
            x_list.append(tokens_tensor[offset:offset + seq_len])
            y_list.append(tokens_tensor[offset + 1:offset + seq_len + 1])

        x = torch.stack(x_list).to(device)
        y = torch.stack(y_list).to(device)

        # Forward (model.forward only returns logits)
        with torch.amp.autocast(
            'cuda', dtype=torch.float16, enabled=device.type == 'cuda'
        ):
            logits = model(x)

        # Loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )

        total_loss += loss.item() * actual_bs * seq_len
        total_tokens += actual_bs * seq_len

        n_batches += 1
        if n_batches % 50 == 0:
            avg_loss_so_far = total_loss / total_tokens
            avg_ppl_so_far = math.exp(min(avg_loss_so_far, 20))  # Prevent overflow
            elapsed = time.time() - start_time
            print(f"  Batch {n_batches:4d} | Loss: {avg_loss_so_far:.4f} | "
                  f"PPL: {avg_ppl_so_far:7.2f} | "
                  f"Time: {elapsed:.1f}s")

    # Final results
    avg_loss = total_loss / total_tokens
    avg_ppl = math.exp(avg_loss)
    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"  VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"  Validation Loss:      {avg_loss:.4f}")
    print(f"  Validation Perplexity: {avg_ppl:.2f}")

    print(f"  Total tokens:         {total_tokens:,}")
    print(f"  Time:                 {elapsed:.1f}s")
    print(f"  Throughput:           {total_tokens/elapsed:,.0f} tok/s")
    print(f"{'='*60}\n")

    return avg_loss, avg_ppl


@torch.no_grad()
def generate_text(model, enc, prompt, max_new_tokens, temperature, top_k, device):
    """Autoregressive text generation"""
    model.eval()

    print(f"\n{'='*60}")
    print(f"  TEXT GENERATION")
    print(f"{'='*60}")
    print(f"  Prompt: \"{prompt}\"")
    print(f"  Temperature: {temperature}, Top-K: {top_k}")
    print(f"  Max new tokens: {max_new_tokens}")
    print(f"{'='*60}\n")

    # Encode prompt
    token_ids = enc.encode(prompt, allowed_special=set())
    tokens = torch.tensor([token_ids], dtype=torch.long, device=device)

    generated = list(token_ids)
    start_time = time.time()

    for i in range(max_new_tokens):
        # Truncate to max_seq_len
        input_ids = tokens[:, -model.config.max_seq_len:]

        with torch.amp.autocast(
            'cuda', dtype=torch.float16, enabled=device.type == 'cuda'
        ):
            logits = model(input_ids)

        # Take logits at the last position
        next_logits = logits[:, -1, :] / temperature

        # Top-K filtering
        if top_k > 0:
            values, _ = torch.topk(next_logits, top_k)
            min_val = values[:, -1].unsqueeze(-1)
            next_logits = torch.where(
                next_logits < min_val,
                torch.full_like(next_logits, float('-inf')),
                next_logits
            )

        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        generated.append(next_token.item())
        tokens = torch.cat([tokens, next_token], dim=1)

        # Can stop early when encountering eos token
        # (TinyStories has no special eos token, stop by length)

    elapsed = time.time() - start_time
    output_text = enc.decode(generated)

    print(f"--- Generated Text ---")
    print(output_text)
    print(f"--- End ---")
    print(f"\n[{max_new_tokens} tokens in {elapsed:.2f}s, "
          f"{max_new_tokens/elapsed:.1f} tok/s]")

    return output_text


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # Load model
    model, config = load_model(args.checkpoint, device)

    # Load validation data
    tokens_tensor, n_samples, enc = load_validation_data(args.seq_len, args.max_samples)

    # Validate
    val_loss, val_ppl = validate(model, tokens_tensor, n_samples,
                                 args.seq_len, args.batch_size, device)

    # Text generation demo
    if args.generate:
        generate_text(
            model,
            enc,
            args.prompt,
            args.max_new_tokens,
            args.temperature,
            args.top_k,
            device,
        )


if __name__ == "__main__":
    main()
