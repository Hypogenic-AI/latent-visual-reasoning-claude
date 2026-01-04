# Downloaded Papers

This directory contains papers relevant to the research on "Formalizing Latent Visual Reasoning: A Theoretical Framework for Token Dynamics."

## Core Papers on Latent Visual Reasoning

### 1. Latent Visual Reasoning (LVR)
- **File**: `2509.24251_latent_visual_reasoning.pdf`
- **Authors**: Bangzheng Li, Ximeng Sun, Jiang Liu, Ze Wang, et al.
- **Year**: 2025
- **arXiv**: [2509.24251](https://arxiv.org/abs/2509.24251)
- **Why relevant**: Introduces the core paradigm of latent visual reasoning - enabling autoregressive reasoning directly in the visual embedding space. Uses reconstruction of visual tokens for reasoning.

### 2. Latent Chain-of-Thought for Visual Reasoning
- **File**: `2510.23925_latent_cot_visual_reasoning.pdf`
- **Authors**: Various
- **Year**: 2025
- **arXiv**: [2510.23925](https://arxiv.org/abs/2510.23925)
- **Why relevant**: Reformulates reasoning in LVLMs as posterior inference, proposes scalable training via amortized variational inference with diversity-seeking RL.

### 3. Machine Mental Imagery (Mirage)
- **File**: `2506.17218_machine_mental_imagery_mirage.pdf`
- **Authors**: Various
- **Year**: 2025
- **arXiv**: [2506.17218](https://arxiv.org/abs/2506.17218)
- **Why relevant**: Emits compact latent visual tokens rather than generating pixels, enabling interleaved visual-text reasoning without image generation overhead.

### 4. Latent Implicit Visual Reasoning
- **File**: `2512.21218_latent_implicit_visual_reasoning.pdf`
- **Authors**: Various
- **Year**: 2025
- **arXiv**: [2512.21218](https://arxiv.org/abs/2512.21218)
- **Why relevant**: Explores implicit visual reasoning in latent space, decoupling internal computation from external tokens.

## Foundational Papers on Latent Reasoning

### 5. Coconut: Chain of Continuous Thought
- **File**: `2412.06769_coconut_continuous_latent.pdf`
- **Authors**: Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, Yuandong Tian
- **Year**: 2024
- **arXiv**: [2412.06769](https://arxiv.org/abs/2412.06769)
- **Code**: https://github.com/facebookresearch/coconut
- **Why relevant**: Foundational work on continuous latent reasoning. Shows that continuous thought can encode multiple reasoning paths (BFS-like search), key theoretical insight for token dynamics.

### 6. Think Before You Speak: Pause Tokens
- **File**: `2310.02226_pause_tokens.pdf`
- **Authors**: Sachin Goyal, Ziwei Ji, Ankit Singh Rawat, et al.
- **Year**: 2024 (ICLR 2024)
- **arXiv**: [2310.02226](https://arxiv.org/abs/2310.02226)
- **Why relevant**: Introduces learnable pause tokens for extra computation before answer generation. Provides empirical evidence for benefits of delayed token prediction.

### 7. Pause Tokens Strictly Increase Expressivity
- **File**: `2505.21024_pause_tokens_expressivity.pdf`
- **Authors**: Various
- **Year**: 2025
- **arXiv**: [2505.21024](https://arxiv.org/abs/2505.21024)
- **Why relevant**: First formal separation result proving pause tokens strictly increase computational expressivity of constant-depth transformers - key theoretical contribution.

## Survey and Analysis Papers

### 8. Reasoning Beyond Language: Latent CoT Survey
- **File**: `2505.16782_latent_cot_survey.pdf`
- **Authors**: Xinghao Chen, Anhao Zhao, Heming Xia, et al.
- **Year**: 2025
- **arXiv**: [2505.16782](https://arxiv.org/abs/2505.16782)
- **Why relevant**: Comprehensive survey on latent chain-of-thought reasoning. Provides taxonomy of methods: token-wise horizontal level (representation, optimization, inference) and layer-wise vertical level.

### 9. Rethinking Thinking Tokens
- **File**: `2411.11371_thinking_tokens_rethinking.pdf`
- **Authors**: Various
- **Year**: 2024
- **arXiv**: [2411.11371](https://arxiv.org/abs/2411.11371)
- **Why relevant**: Analyzes why thinking tokens underperform in practice - identifies training instabilities from single-token embedding mechanism.

## Perception and Visual Token Papers

### 10. Perception Tokens (AURORA)
- **File**: `2412.03548_perception_tokens_aurora.pdf`
- **Authors**: Mahtab Bigverdi, Zelun Luo, Cheng-Yu Hsieh, et al.
- **Year**: 2024
- **arXiv**: [2412.03548](https://arxiv.org/abs/2412.03548)
- **Why relevant**: Introduces perception tokens as auxiliary reasoning tokens for visual tasks. Uses VQVAE to transform visual representations into tokens for chain-of-thought reasoning.

### 11. MCOUT: Multimodal Chain of Continuous Thought
- **File**: `2508.12587_mcout_multimodal_chain.pdf`
- **Authors**: Various
- **Year**: 2025
- **arXiv**: [2508.12587](https://arxiv.org/abs/2508.12587)
- **Why relevant**: Extends continuous thought paradigm to multimodal setting, proposes latent attention mechanism for visual-text reasoning.

## Benchmark Papers

### 12. Visual Spatial Reasoning (VSR)
- **File**: `2205.00363_vsr_spatial_reasoning.pdf`
- **Authors**: Fangyu Liu, Guy Emerson, Nigel Collier
- **Year**: 2023 (TACL)
- **arXiv**: [2205.00363](https://arxiv.org/abs/2205.00363)
- **Why relevant**: Benchmark with 10K+ examples and 66 spatial relation types. Shows large gap between human (95%) and model (70%) performance.

### 13. BLINK: Multimodal LLMs Can See but Not Perceive
- **File**: `2404.12390_blink_benchmark.pdf`
- **Authors**: Zeyuan Fu et al.
- **Year**: 2024 (ECCV 2024)
- **arXiv**: [2404.12390](https://arxiv.org/abs/2404.12390)
- **HuggingFace**: BLINK-Benchmark/BLINK
- **Why relevant**: 14 perception tasks including depth, counting, correspondence. Humans 95.7% vs GPT-4V 51.3% - shows fundamental perception gap.

### 14. MathVista: Mathematical Reasoning in Visual Contexts
- **File**: `2310.02255_mathvista.pdf`
- **Authors**: Various
- **Year**: 2024 (ICLR 2024 Oral)
- **arXiv**: [2310.02255](https://arxiv.org/abs/2310.02255)
- **HuggingFace**: AI4Math/MathVista
- **Why relevant**: 6,141 examples for visual mathematical reasoning. Tests both perception and reasoning capabilities.

## Summary Statistics

- **Total Papers**: 14
- **Core Visual Latent Reasoning**: 4 papers
- **Foundational Latent Reasoning**: 3 papers
- **Survey/Analysis**: 2 papers
- **Perception Tokens**: 2 papers
- **Benchmarks**: 3 papers
- **Years**: 2023-2025
