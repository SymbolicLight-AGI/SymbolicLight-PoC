# SymbolicLight-PoC
SymbolicLight: A natively-trained Neuro-Symbolic Spiking Language Architecture. Replaces dense attention with SparseTCAM and LIF neurons for 87-91% activation sparsity. (168M PoC Snapshot)

### 📢 Release Notice: Inference-Only Snapshot

This repository contains the **inference-only snapshot** of the SymbolicLight V1 architecture (168M parameters). 
We provide the full model definition (`model.py`), autoregressive generation scripts (`generate.py`), and pre-trained weights (`best.pt`) to allow the community to verify the **90% activation sparsity**, the logic of **SparseTCAM routing**, and the effectiveness of the **Bayesian Decoding Head**.

**A Note on Training Infrastructure:**
Training spiking language models natively from scratch involves extremely non-trivial surrogate gradient calibration, customized optimizer scheduling, and distributed stabilization techniques. As we are actively advancing toward the 1B+ scale, these proprietary training infrastructures, scripts (`train.py`), and dynamic datasets are excluded from this initial public codebase.

---

## A Next-Generation Neuro-Symbolic Spiking Language Model

SymbolicLight is not just another connectionist deep learning model; it is a **true Neuro-Symbolic system natively fused at the lowest architectural level**.

We reject the superficial "LLM + external calculator/knowledge graph" paradigm. Instead, we weave continuous connectionist networks with discrete symbolic logic systems at the fundamental computational unit:
* **Neuro (The Connectionist Engine):** Retains biological generalization and learning capabilities based on synaptic plasticity (STDP), enabling it to process fuzzy, high-dimensional natural language representations and continuously evolve during inference.
* **Symbolic (The Logic Controller):** Utilizes discrete binary spikes (0/1) as information carriers. This inherent Boolean logic directly triggers deterministic content-addressable memory (SparseTCAM) in the backend—replacing probabilistic attention—and forcibly injects rule-based conditional branching via an EntropyGate and a Bayesian Head.

Through this deep fusion, SymbolicLight delivers profound cognitive reasoning alongside **transparent interpretability** and a theoretical **100x leap in hardware execution efficiency** on edge devices.

## Key Innovations

1.  **SparseTCAM Routing:** We completely abandon the $O(n^2)$ probabilistic self-attention of traditional Transformers. Instead, we deploy $O(n \cdot k)$ Sparse Content-Addressable Memory, routing signals deterministically based on explicit Boolean logic.
2.  **LIF Neurons (Ultra-Sparse Spiking Engine):** Event-driven neuron clusters maintain an up to 90% resting rate (zero spikes) during inference, entirely bypassing massive dense matrix multiplications.
3.  **EntropyGate (White-Box Logic Control):** Introduces `If-Else` conditional branching into the deepest layers of the neural network. Low-entropy tokens trigger rule-based "Early Exits," avoiding redundant computation.
4.  **Bayesian Head:** Replaces traditional blind-guessing Softmax with deterministic statistical inference, utilizing robust prior and posterior confidence boundaries.
5.  **STDP Online Learning:** Achieves maintenance-free lifelong learning at the edge. Neurons naturally reshape weights based on spike timing during inference, completely eliminating the need for backpropagation gradients.

## Quick Start

**1. Requirements**
```bash
pip install torch transformers gradio
```

**2. Validation**
Calculate perplexity and activation sparsity on the TinyStories validation set:

```bash
python src/validate.py --checkpoint src/best.pt
```

**3. Interactive Text Generation**

```bash
python src/generate.py
```

**4. Web UI**
Launch the Gradio UI for visual testing:

```bash
python src/web_demo.py
```

---

## License & Dual Licensing

SymbolicLight operates under a **Dual Licensing** model to support foundational open-source research while protecting the commercial rights of our core technology.

### 1. Open Source & Academic Use (AGPLv3)

This Proof-of-Concept (PoC) architecture validation snapshot and inference codebase are released under the **[GNU Affero General Public License v3.0 (AGPLv3)](LICENSE)**. We are committed to an open-architecture strategy. As we scale to 1B/3B edge models, we will continue to release upgrades under the AGPLv3 to build the Physical AI ecosystem alongside the community.

Under this license, you are free to use, modify, and distribute the code. **However, any modifications, derivative works, or services built upon this codebase (including wrapping the model as a cloud API, or embedding it into edge/terminal hardware) MUST also be entirely open-sourced under the AGPLv3.**

### 2. Commercial License (Closed-Source Use)

If you intend to integrate SymbolicLight into a proprietary, closed-source commercial product (e.g., edge AI hardware, mobile devices, autonomous driving systems, robotics, or proprietary SaaS/API platforms) without releasing your source code, you **MUST obtain a separate Commercial License**.

A Commercial License exempts you from the AGPLv3 copyleft constraints, allowing you to deploy our highly optimized sparse network code in closed-source products with full deployment support from our core engineering team.

## Hardware Partnerships & Early Access

While this PoC is built for architecture validation, we are actively collaborating with edge AI chip designers, robotics, and smart hardware manufacturers for our upcoming 1B+ parameter production models.

If you are interested in co-designing the next generation of ultra-low-power edge devices, or require a Commercial License for future closed-source deployment, let's talk:
📧 bd@symboliclight.com
