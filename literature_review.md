# Literature Review: Formalizing Latent Visual Reasoning

## Research Area Overview

Latent visual reasoning is an emerging paradigm in multimodal AI that enables reasoning directly in continuous latent spaces rather than through explicit natural language. This approach addresses fundamental limitations in current Vision-Language Models (VLMs) where reasoning is confined to the language space while visual information is treated as static input. The field builds on two main threads: (1) latent reasoning in language models (e.g., Coconut, Pause Tokens) and (2) multimodal token-based reasoning (e.g., Perception Tokens, LVR).

The core hypothesis of this research area is that language may not be the optimal medium for visual reasoning. Neuroimaging studies show that the language network remains largely inactive during reasoning tasks, and many visual reasoning problems (depth estimation, spatial relations, counting) resist mediation through natural language.

---

## Key Papers

### Paper 1: Latent Visual Reasoning (LVR)
- **Authors**: Bangzheng Li, Ximeng Sun, Jiang Liu, Ze Wang, et al.
- **Year**: 2025
- **Source**: arXiv:2509.24251
- **Key Contribution**: Introduces a paradigm where LLMs reason by generating latent states that reconstruct key visual tokens relevant to the query, operating directly in the visual embedding space.
- **Methodology**:
  - Visual encoder projects images into tokens in a joint semantic space with the LLM
  - LLM generates hidden states that reconstruct query-relevant visual tokens
  - Two-stage training: (1) SFT with reconstruction loss + next-token prediction, (2) RL with adapted GRPO algorithm
  - Special tokens `<|lvr_start|>` and `<|lvr_end|>` mark latent reasoning segments
- **Datasets Used**: MMVP, visual QA benchmarks
- **Results**: 71.67% on MMVP vs 66.67% for Qwen2.5-VL baseline
- **Code Available**: Promised to be released
- **Relevance**: Core paper for this research - defines the latent visual reasoning paradigm with mathematical formulation

### Paper 2: Coconut (Chain of Continuous Thought)
- **Authors**: Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, Yuandong Tian
- **Year**: 2024
- **Source**: arXiv:2412.06769
- **Key Contribution**: Foundational work showing LLMs can reason in continuous latent space by feeding hidden states back as input embeddings. Enables BFS-like reasoning through continuous thoughts.
- **Methodology**:
  - Replace standard token prediction with hidden state propagation
  - "Continuous thought" = last hidden state used as next input embedding
  - Multi-stage training inspired by iCoT
  - Special tokens `<bot>` and `<eot>` for latent mode
- **Datasets Used**: GSM8k, ProntoQA, ProsQA (new)
- **Results**: Outperforms CoT on logical reasoning requiring search/planning
- **Code Available**: https://github.com/facebookresearch/coconut (MIT License)
- **Relevance**: Provides theoretical foundation and training methodology for continuous latent reasoning

### Paper 3: Think Before You Speak: Pause Tokens
- **Authors**: Sachin Goyal, Ziwei Ji, Ankit Singh Rawat, et al.
- **Year**: 2024 (ICLR 2024)
- **Source**: arXiv:2310.02226
- **Key Contribution**: Shows that appending learnable pause tokens allows models to compute more before generating answers, providing empirical evidence for benefits of delayed prediction.
- **Methodology**:
  - Append learnable `<pause>` tokens to input
  - Delay extracting outputs until last pause token
  - Both pretrain and finetune with pause tokens
- **Datasets Used**: SQuAD, CommonSenseQA, GSM8k
- **Results**: +18% EM on SQuAD, +8% on CommonSenseQA, +1% on GSM8k (for 1B model)
- **Code Available**: Not specified
- **Relevance**: Provides empirical foundation for "extra computation" hypothesis

### Paper 4: Pause Tokens Strictly Increase Expressivity
- **Authors**: Various
- **Year**: 2025
- **Source**: arXiv:2505.21024
- **Key Contribution**: First formal separation result proving pause tokens strictly increase computational expressivity of constant-depth, logarithmic-width transformers.
- **Methodology**: Theoretical analysis of computational expressivity
- **Relevance**: Provides theoretical justification for pause/thinking tokens - essential for formal framework

### Paper 5: Perception Tokens (AURORA)
- **Authors**: Mahtab Bigverdi, Zelun Luo, Cheng-Yu Hsieh, et al.
- **Year**: 2024
- **Source**: arXiv:2412.03548
- **Key Contribution**: Introduces perception tokens as auxiliary reasoning tokens that encode visual representations (depth maps, bounding boxes) for visual reasoning.
- **Methodology**:
  - VQVAE transforms intermediate image representations (depth maps) into tokens
  - Multi-task training framework
  - Tokens act as chain-of-thought for visual reasoning
- **Datasets Used**: BLINK, CVBench, SEED-Bench
- **Results**: +10.8% on BLINK counting, +6% on BLINK depth
- **Code Available**: Project page promised
- **Relevance**: Demonstrates encoding visual semantics as tokens for reasoning

### Paper 6: Latent CoT Survey
- **Authors**: Xinghao Chen, Anhao Zhao, Heming Xia, et al.
- **Year**: 2025
- **Source**: arXiv:2505.16782
- **Key Contribution**: Comprehensive survey providing taxonomy of latent CoT methods
- **Taxonomy**:
  - **Token-wise Horizontal Level**:
    - Representation Initialization (hidden state, weighted embedding, special vector)
    - Model Optimization (pre-training, SFT, RL)
    - Inference Exploration (sequential scaling, parallel scaling)
  - **Layer-wise Vertical Level**:
    - Encoder-based models
    - Decoder-based models
- **Relevance**: Essential for understanding the landscape and positioning new contributions

### Paper 7: MCOUT (Multimodal Chain of Continuous Thought)
- **Authors**: Various
- **Year**: 2025
- **Source**: arXiv:2508.12587
- **Key Contribution**: Extends continuous thought to multimodal setting with latent attention mechanism
- **Methodology**:
  - MCOUT-Base: Uses LM's last hidden state as continuous thought
  - MCOUT-Multi: Integrates hidden state with image embeddings via multimodal latent attention
- **Relevance**: Provides multimodal extension of Coconut framework

### Paper 8: Rethinking Thinking Tokens
- **Authors**: Various
- **Year**: 2024
- **Source**: arXiv:2411.11371
- **Key Contribution**: Analyzes why thinking tokens underperform CoT in practice
- **Key Findings**:
  - Thinking tokens suffer from noisy gradients due to single-token embedding
  - Training instabilities limit effectiveness
- **Relevance**: Important for understanding limitations and designing better approaches

---

## Common Methodologies

### Latent Token Approaches

**Method A: Hidden State Propagation** (Used in: Coconut, LVR, MCOUT)
- Feed last hidden state directly as next input embedding
- Bypass language model head (no token prediction during reasoning)
- Enables continuous, differentiable reasoning

**Method B: Learnable Tokens** (Used in: Pause Tokens, Thinking Tokens)
- Insert special learnable tokens into sequence
- Model learns to use extra computation budget
- Simpler but potentially less expressive

**Method C: Perception/Auxiliary Tokens** (Used in: AURORA, LVR)
- Encode visual representations as discrete tokens
- VQVAE or similar for tokenization
- Enables chain-of-thought with visual semantics

### Training Paradigms

1. **Multi-stage Training** (Coconut, LVR)
   - Stage 1: SFT with reconstruction/prediction loss
   - Stage 2: RL for refinement (GRPO, PPO)

2. **Curriculum Learning** (AURORA)
   - Gradual introduction of auxiliary tokens
   - Prevents catastrophic forgetting

3. **Distillation** (AURORA, various)
   - Use specialist models to supervise auxiliary tokens
   - Knowledge transfer from perception models

---

## Standard Baselines

1. **Chain-of-Thought (CoT)**: Standard explicit reasoning in language
   - Typical performance: varies by task
   - Limitation: constrained to language space

2. **Direct Prediction**: No intermediate reasoning
   - Fast but limited for complex tasks

3. **Tool-Augmented**: External visual tools (crop, zoom, OCR)
   - "Thinking with Images" paradigm
   - Limitation: dependent on tool availability

---

## Evaluation Metrics

### Visual Perception Benchmarks
- **BLINK**: Accuracy across 14 perception tasks
- **CVBench**: Visual counting accuracy
- **SEED-Bench**: Counting accuracy

### Spatial Reasoning
- **VSR**: Binary accuracy on spatial relation classification

### Mathematical Reasoning
- **MathVista**: Multi-task accuracy
- **GSM8k**: Grade-school math accuracy

### General VQA
- **MMVP**: Multi-modal visual perception accuracy

---

## Datasets in the Literature

| Dataset | Used In | Task | Key Challenge |
|---------|---------|------|---------------|
| BLINK | LVR, AURORA | 14 perception tasks | Humans 95.7% vs SOTA 51.3% |
| MathVista | Various | Visual math | Combined perception + reasoning |
| VSR | Various | Spatial relations | Orientation relations at chance |
| GSM8k | Coconut, Pause | Math word problems | Multi-step reasoning |
| MMVP | LVR | Visual perception QA | Fine-grained visual understanding |

---

## Gaps and Opportunities

### Gap 1: Formal Mathematical Framework
Current work is empirically driven without comprehensive theoretical analysis of:
- What token dynamics enable effective latent reasoning
- Formal expressivity bounds for different approaches
- Optimal training objectives for latent reasoning

### Gap 2: Understanding Token Dynamics
Limited analysis of:
- How information flows through latent tokens
- What latent tokens actually encode
- Relationship between token count and reasoning depth

### Gap 3: Unified Multimodal Framework
Current approaches are fragmented:
- LVR focuses on visual token reconstruction
- Coconut focuses on pure latent reasoning
- No unified theory for visual + latent reasoning

### Gap 4: Interpretability
Latent reasoning is inherently less interpretable:
- Cannot inspect intermediate reasoning steps
- Difficult to diagnose failures
- Need for probing methods and analysis tools

### Gap 5: Efficiency Analysis
Limited understanding of:
- Computational trade-offs between explicit and latent reasoning
- Optimal allocation of latent vs explicit tokens
- Scaling properties of latent reasoning

---

## Recommendations for Our Experiment

### Recommended Datasets

1. **Primary**: BLINK (Counting + Relative_Depth subtasks)
   - Clear perception gap between humans and models
   - Tasks that "resist mediation through natural language"
   - Good for testing latent visual reasoning improvements

2. **Secondary**: MathVista testmini
   - Combined perception and reasoning
   - Tests chain-of-thought improvements

3. **Diagnostic**: VSR
   - Fine-grained spatial reasoning analysis
   - 66 relation types for detailed evaluation

### Recommended Baselines

1. **Coconut**: Foundation for continuous latent reasoning
2. **Standard CoT**: Explicit reasoning baseline
3. **Direct prediction**: No reasoning baseline

### Recommended Metrics

- Accuracy on perception tasks (BLINK)
- Accuracy on spatial reasoning (VSR)
- Computational efficiency (tokens generated, latency)

### Methodological Considerations

1. **Token Dynamics Analysis**: Probe what information latent tokens encode
2. **Ablation Studies**: Vary number of latent tokens, training objectives
3. **Theoretical Framework**: Formalize relationship between:
   - Token count and computational expressivity
   - Reconstruction loss and reasoning quality
   - Training objectives and token dynamics

---

## Summary

The literature reveals a rapidly evolving field of latent visual reasoning with several key insights:

1. **Language may not be optimal for visual reasoning** - continuous latent spaces offer advantages for certain tasks
2. **Hidden state propagation** enables BFS-like reasoning patterns not possible with explicit tokens
3. **Perception tokens** can encode visual semantics for chain-of-thought reasoning
4. **Formal theoretical analysis** is lacking - a major opportunity for contribution

The research hypothesis - that a formal mathematical framework can reveal strengths and weaknesses of latent token mechanisms - is well-motivated by the current state of the field where empirical results outpace theoretical understanding.
