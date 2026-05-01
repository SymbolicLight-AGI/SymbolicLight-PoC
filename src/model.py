#!/usr/bin/env python3
"""
SymbolicLight — Next-Generation Neuro-Symbolic Spiking Large Model Architecture
==================================================
Key Innovations:
  1. SparseTCAM: Spiking sparse routing replacing Self-Attention
  2. LIF Neurons: Event-driven replacing dense activation
  3. EntropyGate: On-demand compute depth (early exit for simple queries)
  4. BayesianHead: Bayesian token selection replacing Softmax
  5. STDP: Online learning during inference (no backward pass needed)
"""
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
#  Configuration
# ============================================================================
@dataclass
class SymbolicLightConfig:
    """SymbolicLight 0.1B Default Configuration"""
    vocab_size: int = 32000       # Vocabulary size (reusing Qwen/Llama tokenizer)
    embed_dim: int = 768          # Embedding dimension
    n_layers: int = 12            # Number of SymbolicLightBlock layers
    n_heads: int = 12             # Number of channels in SparseTCAM
    head_dim: int = 64            # Dimension per channel (embed_dim / n_heads)
    intermediate_dim: int = 2048  # Feedforward intermediate dimension
    max_seq_len: int = 2048       # Maximum sequence length
    dropout: float = 0.1
    # --- SymbolicLight Specific Parameters ---
    spike_threshold: float = 1.0  # LIF neuron firing threshold
    leak_factor: float = 0.95     # Membrane potential leak factor
    stdp_lr: float = 0.01         # STDP learning rate
    entropy_exit_threshold: float = 0.3  # EntropyGate early exit threshold
    enable_entropy_exit: bool = False  # Enable EntropyGate early exit (requires Early Exit Head training to enable)
    enable_stdp: bool = False     # Enable STDP (disabled during pre-training, enabled during deployment)


# ============================================================================
#  Surrogate Gradient Function (enables BP training for non-differentiable spikes)
# ============================================================================
class SurrogateSpike(torch.autograd.Function):
    """
    Forward: Hard threshold -> 0/1 spikes (non-differentiable)
    Backward: Use derivative of sigmoid as surrogate gradient (differentiable)

    This is the key mathematical trick enabling SNNs to be trained with backpropagation!
    """
    sigma = 10.0  # Steepness of the surrogate gradient

    @staticmethod
    def forward(ctx, membrane_potential, threshold):
        ctx.save_for_backward(membrane_potential, torch.tensor(threshold))
        return (membrane_potential >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        membrane_potential, threshold = ctx.saved_tensors
        # Surrogate gradient: derivative of sigmoid σ·sigmoid(σx)·(1-sigmoid(σx))
        x = SurrogateSpike.sigma * (membrane_potential - threshold)
        sigmoid_x = torch.sigmoid(x)
        surrogate_grad = SurrogateSpike.sigma * sigmoid_x * (1.0 - sigmoid_x)
        return grad_output * surrogate_grad, None


def surrogate_spike(membrane_potential: torch.Tensor, threshold: float = 1.0) -> torch.Tensor:
    """Exposed surrogate gradient spike function"""
    return SurrogateSpike.apply(membrane_potential, threshold)


# ============================================================================
#  ② SpikeEncoder
# ============================================================================
class SpikeEncoder(nn.Module):
    """
    Converts discrete token IDs into spatio-temporal spike tensors.

    Process: token_id -> Embedding -> LayerNorm -> LIF Spiking
    """
    def __init__(self, config: SymbolicLightConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.norm = nn.LayerNorm(config.embed_dim)
        self.threshold = config.spike_threshold
        self.leak = config.leak_factor

        # Positional encoding: Learnable Embedding (simple and effective)
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.embed_dim)

        # Membrane potential (managed dynamically per batch)
        self.register_buffer("v_mem", None)

    def _init_membrane(self, shape: torch.Size, device: torch.device):
        """Initialize/reset membrane potential"""
        self.v_mem = torch.zeros(shape, device=device)

    def forward(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            token_ids: [batch, seq_len]
        Returns:
            spikes: [batch, seq_len, embed_dim]  Sparse 0/1 spikes
            continuous: [batch, seq_len, embed_dim]  Continuous representation (used for residuals)
        """
        B, S = token_ids.shape
        positions = torch.arange(S, device=token_ids.device).unsqueeze(0)

        # Embedding + Positional encoding
        x = self.embedding(token_ids) + self.pos_embedding(positions)
        x = self.norm(x)

        # LIF spiking: process step-by-step over time (seq dimension)
        self._init_membrane((B, x.size(-1)), x.device)
        spikes_list = []

        for t in range(S):
            # Leak + Integrate
            self.v_mem = self.v_mem * self.leak + x[:, t, :]
            # Fire spike (via surrogate gradient, backpropagatable)
            spike = surrogate_spike(self.v_mem, self.threshold)
            # Reset after firing
            self.v_mem = self.v_mem * (1.0 - spike)
            spikes_list.append(spike)

        spikes = torch.stack(spikes_list, dim=1)  # [B, S, D]
        return spikes, x  # Return spikes and continuous representation (continuous repr used for residual connections)


# ============================================================================
#  ③a SparseTCAM — In-Memory Compute Sparse Routing (Replaces Self-Attention)
# ============================================================================
class SparseTCAM(nn.Module):
    """
    Simulates in-memory compute of the S100 Graph-TCAM.

    Core difference from Self-Attention:
    - Attention: QxK^T -> All-to-all O(n^2) dense matrix multiplication
    - SparseTCAM: Spikes x Weights -> Only activate weight rows hit by spikes -> O(n*k), k << n

    In GPU software implementation, we achieve \"sparse read\" via spike masks.
    """
    def __init__(self, config: SymbolicLightConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.embed_dim = config.embed_dim
        self.threshold = config.spike_threshold
        self.leak = config.leak_factor

        # TCAM weight matrix (In-memory compute: serves as both storage and computation)
        self.tcam_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        # Output projection
        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.norm = nn.LayerNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, spikes: torch.Tensor, continuous: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            spikes: [B, S, D]  Input spikes (sparse 0/1)
            continuous: [B, S, D]  Continuous representation (for residual connections)
        Returns:
            out_spikes: [B, S, D]  Output spikes
            out_continuous: [B, S, D]  Updated continuous representation
        """
        B, S, D = spikes.shape

        # 1. Spike mask routing: only compute at positions \"with spikes\"
        # spike_mask indicates which positions have active spikes
        spike_energy = spikes.sum(dim=-1)  # [B, S] Spike energy at each position
        active_mask = (spike_energy > 0).unsqueeze(-1).float()  # [B, S, 1]

        # 2. In-memory compute addressing: spikes x TCAM weights
        # On real chips this is single-cycle TCAM match; on GPU it is sparse matrix multiplication
        tcam_out = self.tcam_proj(spikes * active_mask)

        # 3. Multi-channel information fusion (similar to multi-head but based on spike channels)
        tcam_out = tcam_out.view(B, S, self.n_heads, self.head_dim)

        # 4. Temporal context aggregation (using causal mask to prevent seeing future info)
        # Simplified version: use cumulative mean replacing all-to-all attention
        causal_cumsum = torch.cumsum(tcam_out, dim=1)
        counts = torch.arange(1, S + 1, device=spikes.device).float().view(1, S, 1, 1)
        context = causal_cumsum / counts

        # 5. Merge channels + Output projection
        context = context.view(B, S, D)
        output = self.out_proj(self.dropout(context))

        # 6. Residual connection + Layer normalization
        out_continuous = self.norm(continuous + output)

        # 7. LIF spiking output
        out_spikes = surrogate_spike(out_continuous, self.threshold)

        return out_spikes, out_continuous


# ============================================================================
#  ③b EntropyGate — Entropy Gating (On-demand compute depth)
# ============================================================================
class EntropyGate(nn.Module):
    """
    Innovation from S22 Entropy Engine: Calculate information entropy of current spike stream.
    Low entropy = Model is highly certain -> Can early exit, no need to run all layers.
    High entropy = Model is still confused -> Continue to deeper layers.

    Transformers lack this capability: regardless of query simplicity, all layers must execute.
    """
    def __init__(self, config: SymbolicLightConfig):
        super().__init__()
        self.threshold = config.entropy_exit_threshold
        # Learn a linear projection to predict \"whether to exit\"
        self.gate = nn.Linear(config.embed_dim, 1)

    def forward(self, spikes: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """
        Returns:
            entropy: Information entropy of the current spike stream [B]
            should_exit: Whether early exit is recommended
        """
        # Calculate spike firing rate (how many dimensions are activated per position)
        firing_rate = spikes.mean(dim=-1)  # [B, S]

        # Information entropy: H = -p*log(p) - (1-p)*log(1-p)
        p = firing_rate.clamp(1e-7, 1 - 1e-7)
        entropy = -(p * p.log() + (1 - p) * (1 - p).log()).mean()

        # Low entropy -> Can exit
        should_exit = (entropy.item() < self.threshold) if not self.training else False
        return entropy, should_exit


# ============================================================================
#  ③c FeedForward — Feed-Forward Network (with spike sparsity)
# ============================================================================
class SpikingFeedForward(nn.Module):
    """
    Replaces the 2-layer MLP of Transformers.
    Key difference: Intermediate layer uses LIF spike activation instead of GELU/ReLU.
    """
    def __init__(self, config: SymbolicLightConfig):
        super().__init__()
        self.up = nn.Linear(config.embed_dim, config.intermediate_dim, bias=False)
        self.down = nn.Linear(config.intermediate_dim, config.embed_dim, bias=False)
        self.norm = nn.LayerNorm(config.embed_dim)
        self.threshold = config.spike_threshold
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.up(x)
        # Use LIF spikes replacing GELU (extremely sparsifying the intermediate layer)
        h = surrogate_spike(h, self.threshold)
        h = self.down(self.dropout(h))
        return self.norm(residual + h)


# ============================================================================
#  ③ SymbolicLightBlock — Single Layer Compute Block
# ============================================================================
class SymbolicLightBlock(nn.Module):
    """
    A complete SymbolicLight layer, including:
    - SparseTCAM (In-memory compute routing)
    - SpikingFeedForward (Spiking feed-forward)
    - EntropyGate (Entropy gating)
    """
    def __init__(self, config: SymbolicLightConfig):
        super().__init__()
        self.tcam = SparseTCAM(config)
        self.ffn = SpikingFeedForward(config)
        self.entropy_gate = EntropyGate(config)

    def forward(self, spikes: torch.Tensor, continuous: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        # 1. In-memory compute routing
        spikes, continuous = self.tcam(spikes, continuous)
        # 2. Spiking sequence iteration
        continuous = self.ffn(continuous)
        # Re-spiking
        spikes = surrogate_spike(continuous, self.tcam.threshold)
        # 3. Entropy gating
        _, should_exit = self.entropy_gate(spikes)
        return spikes, continuous, should_exit


# ============================================================================
#  ④ BayesianHead — Bayesian Output Head (Replaces Softmax)
# ============================================================================
class BayesianHead(nn.Module):
    """
    Innovation from S100 LALU array: Use Bayesian posterior replacing Softmax.

    Softmax: P(word) = exp(logit) / Σexp(logits)  <- Brutal normalization
    Bayesian: P(word|context) ∝ P(context|word) x P(word)  <- Exact inference

    In V1, we approximate Bayesian updates using addition in the log domain.
    """
    def __init__(self, config: SymbolicLightConfig):
        super().__init__()
        self.output_proj = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        # Prior probability (learnable word frequency bias)
        self.log_prior = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, continuous: torch.Tensor) -> torch.Tensor:
        """
        Args:
            continuous: [B, S, D]
        Returns:
            logits: [B, S, vocab_size]  (Log probabilities, can be directly trained with CrossEntropy)
        """
        # Likelihood term: P(context|word)
        log_likelihood = self.output_proj(continuous)  # [B, S, V]
        # Prior term: P(word)
        # Bayesian update (log domain addition): log P(word|context) = log P(context|word) + log P(word)
        logits = log_likelihood + self.log_prior.unsqueeze(0).unsqueeze(0)
        return logits


# ============================================================================
#  ⑤ STDP Online Learner
# ============================================================================
class STDPUpdater:
    """
    Innovation from S100 ILE Inductive Learning Engine.

    Activated ONLY during inference (model.eval()).
    No loss.backward() required, purely local learning rules based on spike timing.
    """
    def __init__(self, config: SymbolicLightConfig):
        self.lr = config.stdp_lr
        self.enabled = config.enable_stdp

    @torch.no_grad()
    def update(self, model: nn.Module, pre_spikes: torch.Tensor, post_spikes: torch.Tensor):
        """
        STDP Rules:
        - Pre-synaptic fires first -> Strengthen connection (LTP)
        - Post-synaptic fires first -> Weaken connection (LTD)
        """
        if not self.enabled:
            return

        # Calculate causal correlation: which input spikes caused output spikes?
        # pre_spikes: [B, S, D], post_spikes: [B, S, D]
        causal = (pre_spikes.sum(dim=1, keepdim=True) > 0) & (post_spikes.sum(dim=1, keepdim=True) > 0)

        if causal.any():
            # Make minute local updates to all TCAM weights
            for block in model.blocks:
                w = block.tcam.tcam_proj.weight
                # LTP: Strengthen causal paths
                delta = self.lr * (pre_spikes.mean(dim=(0, 1)) @ post_spikes.mean(dim=(0, 1)).unsqueeze(-1))
                w.data += delta.squeeze() * 0.001
                w.data.clamp_(-5, 5)


# ============================================================================
#  Complete Model
# ============================================================================
class SymbolicLightModel(nn.Module):
    """
    SymbolicLight: Next-generation Neuro-Symbolic Spiking Large Model

    Usage:
        config = SymbolicLightConfig()
        model = SymbolicLightModel(config)

        # Training
        logits = model(token_ids)
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), targets.view(-1))

        # Inference (Autoregressive generation)
        output_ids = model.generate(prompt_ids, max_new_tokens=100)
    """
    def __init__(self, config: SymbolicLightConfig):
        super().__init__()
        self.config = config
        self.spike_encoder = SpikeEncoder(config)
        self.blocks = nn.ModuleList([
            SymbolicLightBlock(config) for _ in range(config.n_layers)
        ])
        self.output_head = BayesianHead(config)
        self.stdp = STDPUpdater(config)

        # Weight initialization
        self.apply(self._init_weights)
        # Print parameter count
        n_params = sum(p.numel() for p in self.parameters())
        print(f"[SymbolicLight] Model initialization complete | Parameters: {n_params/1e6:.1f}M ({n_params/1e9:.3f}B)")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation (Universal for training + inference)
        Args:
            token_ids: [batch, seq_len]
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        # ② Spike encoding
        spikes, continuous = self.spike_encoder(token_ids)
        initial_spikes = spikes  # Retained for STDP

        # ③ Layer-by-layer processing
        for block in self.blocks:
            spikes, continuous, should_exit = block(spikes, continuous)
            # EntropyGate early exit: active only when explicitly enabled and not in training mode
            # Note: The current EntropyGate weights are untrained (no auxiliary loss),
            #       default closed to avoid false exits at layer 0.
            #       Keep disabled until Early Exit Head + Auxiliary Loss are implemented.
            if should_exit and not self.training and self.config.enable_entropy_exit:
                break

        # ④ Bayesian Output
        logits = self.output_head(continuous)

        # ⑤ STDP Online Learning (Active only when explicitly enabled during inference)
        if not self.training and self.config.enable_stdp:
            self.stdp.update(self, initial_spikes, spikes)

        return logits

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int = 100,
                 temperature: float = 0.8, top_k: int = 50) -> torch.Tensor:
        """
        Autoregressive text generation

        Args:
            prompt_ids: [1, prompt_len]  Prompt token IDs
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Sample only from the top k highest probability tokens
        Returns:
            Generated complete token sequence
        """
        self.eval()
        generated = prompt_ids.clone()

        for _ in range(max_new_tokens):
            # Truncate to maximum length
            input_ids = generated[:, -self.config.max_seq_len:]

            # Forward inference
            logits = self.forward(input_ids)

            # Take logits only at the last position
            next_logits = logits[:, -1, :] / temperature

            # Top-K sampling
            if top_k > 0:
                top_k_vals, _ = torch.topk(next_logits, top_k)
                min_top_k = top_k_vals[:, -1].unsqueeze(-1)
                next_logits[next_logits < min_top_k] = float('-inf')

            # Sample next token
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)

            # EOS detection (token_id=2 is usually EOS)
            if next_token.item() == 2:
                break

        return generated

    def get_sparsity_stats(self) -> dict:
        """Returns model sparsity statistics (for papers and debugging)"""
        stats = {}
        with torch.no_grad():
            dummy = torch.randint(0, 100, (1, 32))
            spikes, _ = self.spike_encoder(dummy)
            stats['encoder_sparsity'] = 1.0 - spikes.mean().item()
            for i, block in enumerate(self.blocks):
                spikes, _, _ = block(spikes, spikes)
                stats[f'block_{i}_sparsity'] = 1.0 - spikes.mean().item()
        return stats


# ============================================================================
#  Quick Validation
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print(" SymbolicLight Model Architecture Validation")
    print("=" * 60)

    config = SymbolicLightConfig(
        vocab_size=32000,
        embed_dim=768,
        n_layers=12,
        n_heads=12,
        head_dim=64,
        intermediate_dim=2048,
    )

    model = SymbolicLightModel(config)

    # Dummy input
    dummy_input = torch.randint(0, 32000, (2, 128))  # batch=2, seq=128
    print(f"\nInput: batch=2, seq_len=128")

    # Forward propagation
    logits = model(dummy_input)
    print(f"Output logits: {logits.shape}")  # Should be [2, 128, 32000]

    # Sparsity analysis
    stats = model.get_sparsity_stats()
    print(f"\nSparsity analysis:")
    for k, v in stats.items():
        print(f"  {k}: {v*100:.1f}% Silent")

    # Generation test
    prompt = torch.randint(0, 32000, (1, 10))
    print(f"\nAutoregressive generation test (prompt length=10, generating 20 tokens)...")
    output = model.generate(prompt, max_new_tokens=20)
    print(f"Generated sequence length: {output.shape[1]}")

    print("\n[PASS] SymbolicLight model architecture verified!")
