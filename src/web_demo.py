#!/usr/bin/env python3
"""
SymbolicLight — Web Demo
========================
Interactive Gradio interface for testing the trained 0.1B model.

Usage:
  python web_demo.py
  python web_demo.py --checkpoint ./checkpoints/best.pt
  python web_demo.py --share   # Generate public link
"""
print("0. Initializing scripts...")
import sys
import time
import argparse
from pathlib import Path

print("1. Importing torch...")
import torch
import torch.nn.functional as F
print("2. Importing gradio...")
import gradio as gr

sys.path.insert(0, str(Path(__file__).parent))
print("3. Importing model...")
from model import SymbolicLightConfig, SymbolicLightModel

print("4. Importing tiktoken...")
import tiktoken
print("5. Imports done.")


# ============================================================================
#  Model Loading
# ============================================================================
def load_model(checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(checkpoint_path)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"[Load] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config_dict = ckpt["config"]
    config = SymbolicLightConfig(**config_dict)
    config.enable_stdp = False
    config.enable_entropy_exit = False
    model = SymbolicLightModel(config).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    step = ckpt.get("global_step", "?")
    loss = ckpt.get("best_loss", 0)
    print(f"[Load] Model loaded | {n_params/1e6:.1f}M params | step={step} | loss={loss:.4f}")

    return model, config, device


# ============================================================================
#  Generation Function
# ============================================================================
def generate(model, enc, prompt, device, max_tokens=200, temperature=0.8, top_k=50):
    input_ids = enc.encode(prompt, allowed_special=set())
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    start_time = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_tensor,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
        )
    elapsed = time.time() - start_time

    new_ids = output_ids[0, len(input_ids):].tolist()
    generated_text = enc.decode(new_ids)
    tok_per_sec = len(new_ids) / elapsed if elapsed > 0 else 0

    # Calculate sparsity
    with torch.no_grad():
        test_input = input_tensor[:, :min(32, input_tensor.size(1))]
        spikes, _ = model.spike_encoder(test_input)
        sparsity = 1.0 - spikes.mean().item()

    return generated_text, sparsity, tok_per_sec, len(new_ids), elapsed


# ============================================================================
#  Gradio UI
# ============================================================================
def create_ui(model, config, device):
    enc = tiktoken.get_encoding("gpt2")
    n_params = sum(p.numel() for p in model.parameters())

    def on_generate(prompt, max_tokens, temperature, top_k):
        if not prompt.strip():
            return "", ""

        text, sparsity, tok_s, n_tokens, elapsed = generate(
            model, enc, prompt.strip(), device,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            top_k=int(top_k),
        )

        stats = (
            f"📊 **Stats**\n"
            f"- 🧠 Tokens generated: **{n_tokens}**\n"
            f"- ⚡ Speed: **{tok_s:.1f}** tok/s\n"
            f"- 🕐 Time: **{elapsed:.2f}** s\n"
            f"- 🔥 Sparsity: **{sparsity*100:.1f}%** (neurons silent)\n"
            f"- 📏 Model: **{n_params/1e6:.0f}M** params"
        )

        return prompt + text, stats

    def on_clear():
        return "", "", ""

    # Default prompts
    examples = [
        ["Once upon a time, there was a little girl named"],
        ["The cat sat on the"],
        ["Mom said to the children"],
        ["One day, a big dog found a"],
        ["The sun was shining and the birds were"],
        ["A brave knight went to the"],
    ]

    with gr.Blocks(title="SymbolicLight 0.1B Demo") as demo:

        gr.HTML("""
        <h1 style="text-align: center;">⚡ SymbolicLight 0.1B</h1>
        <p style="text-align: center; color: #666;">
            Spiking Neural Network Language Model — 129M params, 89% sparsity.
        </p>
        """)

        with gr.Row():
            with gr.Column(scale=3):
                prompt_input = gr.Textbox(
                    label="✏️ Prompt",
                    placeholder="Type your prompt here... (e.g., 'Once upon a time')",
                    lines=3,
                )
                output_text = gr.Textbox(
                    label="📝 Generated Text",
                    lines=15,
                    max_lines=100,
                    interactive=False,
                )

            with gr.Column(scale=1):
                temperature = gr.Slider(0.1, 2.0, value=0.8, step=0.1, label="🌡️ Temperature")
                max_tokens = gr.Slider(50, 1024, value=200, step=50, label="📏 Max Tokens")
                top_k = gr.Slider(10, 200, value=50, step=10, label="🎯 Top-K")

                with gr.Row():
                    generate_btn = gr.Button("🚀 Generate", variant="primary", scale=2)
                    clear_btn = gr.Button("🗑️ Clear", scale=1)

                stats_output = gr.Markdown(
                    label="Stats",
                    value="*Click Generate to see stats*"
                )

        gr.Examples(
            examples=examples,
            inputs=prompt_input,
            label="💡 Try these prompts",
        )

        generate_btn.click(
            fn=on_generate,
            inputs=[prompt_input, max_tokens, temperature, top_k],
            outputs=[output_text, stats_output],
        )
        prompt_input.submit(
            fn=on_generate,
            inputs=[prompt_input, max_tokens, temperature, top_k],
            outputs=[output_text, stats_output],
        )
        clear_btn.click(
            fn=on_clear,
            outputs=[prompt_input, output_text, stats_output],
        )

    return demo


# ============================================================================
#  Entry Point
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SymbolicLight Web Demo")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best.pt")
    parser.add_argument("--port", type=int, default=7870)
    args = parser.parse_args()

    print("[Init] Loading model...")
    model, config, device = load_model(args.checkpoint)
    print("[Init] Creating UI...")
    demo = create_ui(model, config, device)
    print(f"[Init] Launching on http://127.0.0.1:{args.port}")
    demo.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        inbrowser=True,
    )
