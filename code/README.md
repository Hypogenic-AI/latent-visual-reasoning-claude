# Cloned Repositories

This directory contains code repositories relevant to the research on "Formalizing Latent Visual Reasoning: A Theoretical Framework for Token Dynamics."

## Repository 1: Coconut (Chain of Continuous Thought)

- **URL**: https://github.com/facebookresearch/coconut
- **Location**: `code/coconut/`
- **Purpose**: Foundational implementation of latent reasoning in continuous space
- **License**: MIT

### Key Files
- `coconut/` - Main implementation
- `README.md` - Setup and usage instructions
- `requirements.txt` - Dependencies

### Why This Repository
Coconut is the foundational work for continuous latent reasoning. It demonstrates:
- How to feed hidden states back as input embeddings
- Multi-stage training for latent reasoning
- BFS-like reasoning through continuous thoughts
- Essential baseline for any latent reasoning research

### Usage
```bash
cd code/coconut
pip install -r requirements.txt
# Follow README for training/evaluation
```

---

## Repository 2: BLINK Benchmark

- **URL**: https://github.com/zeyofu/BLINK_Benchmark
- **Location**: `code/blink_benchmark/`
- **Purpose**: Evaluation code for visual perception benchmark
- **License**: MIT

### Key Files
- `eval/` - Evaluation scripts
- `models/` - Model interface code
- `README.md` - Evaluation instructions

### Why This Repository
BLINK provides standardized evaluation for visual perception tasks:
- 14 perception subtasks
- Model evaluation scripts
- Leaderboard comparison

### Usage
```bash
cd code/blink_benchmark
pip install -r requirements.txt
# See README for evaluation setup
```

---

## Repository 3: Visual Spatial Reasoning (VSR)

- **URL**: https://github.com/cambridgeltl/visual-spatial-reasoning
- **Location**: `code/vsr_benchmark/`
- **Purpose**: Benchmark for spatial reasoning evaluation
- **License**: Apache 2.0

### Key Files
- `data/` - Data processing scripts
- `models/` - Baseline model implementations
- `README.md` - Dataset and evaluation instructions

### Why This Repository
VSR provides:
- Spatial reasoning evaluation framework
- 66 spatial relation types
- Baseline implementations (VisualBERT, LXMERT, ViLT, CLIP)

### Usage
```bash
cd code/vsr_benchmark
pip install -r requirements.txt
# Follow README for data setup and evaluation
```

---

## Repository 4: MathVista

- **URL**: https://github.com/lupantech/MathVista
- **Location**: `code/mathvista/`
- **Purpose**: Visual mathematical reasoning benchmark
- **License**: CC BY-SA 4.0

### Key Files
- `evaluation/` - Evaluation scripts
- `data/` - Data processing
- `README.md` - Setup instructions

### Why This Repository
MathVista provides:
- Multi-task visual math reasoning evaluation
- Integration with various models (GPT-4V, Claude, etc.)
- Standardized metrics and leaderboard

### Usage
```bash
cd code/mathvista
pip install -r requirements.txt
# See README for evaluation
```

---

## Summary

| Repository | Purpose | Primary Use |
|------------|---------|-------------|
| coconut | Latent reasoning implementation | Baseline & methodology reference |
| blink_benchmark | Perception evaluation | Experimental evaluation |
| vsr_benchmark | Spatial reasoning evaluation | Diagnostic evaluation |
| mathvista | Math reasoning evaluation | Combined perception+reasoning |

---

## Recommendations

For implementing latent visual reasoning experiments:

1. **Start with Coconut**: Understand the continuous thought mechanism
2. **Evaluate on BLINK**: Test perception improvements
3. **Diagnose with VSR**: Analyze spatial reasoning capabilities
4. **Validate on MathVista**: Test reasoning chain improvements

The Coconut repository provides the core algorithmic approach, while the benchmark repositories provide standardized evaluation frameworks.
