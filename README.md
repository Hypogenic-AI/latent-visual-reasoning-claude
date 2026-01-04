# Formalizing Latent Visual Reasoning: A Theoretical Framework for Token Dynamics

A research project developing **LVRTD (Latent Visual Reasoning Token Dynamics)**, a formal mathematical framework for analyzing and comparing different latent reasoning token mechanisms in vision-language models.

## Key Findings

- **Continuous thought tokens** uniquely enable BFS-like parallel search through superposition of latent states (capacity ≈ √d × k for d-dimensional embeddings and k tokens)
- **Pause tokens** strictly increase computational depth (+O(k)) but cannot encode multiple reasoning paths simultaneously
- **Perception tokens** offer interpretability but suffer from severe discretization bottleneck (~10 bits/token vs ~6000 bits/token for continuous)
- **Structured reasoning** outperforms unstructured chain-of-thought (40% vs 20% on BLINK counting tasks)
- **More tokens ≠ better reasoning** without proper training - validates "Rethinking Thinking Tokens" findings

## Quick Start

```bash
# Create and activate environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install numpy scipy matplotlib seaborn pandas openai anthropic requests tqdm

# Run theoretical analysis only
python src/run_experiments.py --no-empirical

# Run full experiments (requires API key)
export OPENAI_API_KEY="your-key"
python src/run_experiments.py --samples 20
```

## Project Structure

```
latent-visual-reasoning-claude/
├── REPORT.md                    # Full research report with results
├── README.md                    # This file
├── planning.md                  # Research plan
├── literature_review.md         # Literature synthesis
├── resources.md                 # Resource catalog
├── src/
│   ├── theoretical_framework.py # LVRTD framework implementation
│   ├── empirical_validation.py  # API-based experiments
│   └── run_experiments.py       # Main experiment runner
├── results/
│   ├── all_results.json        # Complete results
│   ├── summary_table.md        # Summary table
│   └── figures/                # Visualizations
├── papers/                     # Downloaded research papers
├── datasets/                   # Pre-downloaded datasets
└── code/                       # Cloned repositories (Coconut, BLINK, etc.)
```

## Framework Overview

LVRTD characterizes latent reasoning tokens through:

1. **State Space**: z ∈ ℝ^d with information-theoretic properties
2. **Dynamics**: Transition function T: Z × X → Z with Jacobian analysis
3. **Token Types**:
   - Continuous Thought (hidden state propagation)
   - Pause Tokens (learnable delays)
   - Perception Tokens (visual semantics)

## Results Summary

| Token Type | Best For | Limitation |
|------------|----------|------------|
| Continuous Thought | Search/exploration tasks | Requires multi-stage training |
| Pause Token | Sequential depth | Cannot encode parallel paths |
| Perception Token | Interpretability | Discretization information loss |

See [REPORT.md](REPORT.md) for complete analysis and findings.

## Key References

- [Coconut: Chain of Continuous Thought](https://arxiv.org/abs/2412.06769) (Meta, 2024)
- [Latent Visual Reasoning](https://arxiv.org/abs/2509.24251) (2025)
- [Pause Tokens Strictly Increase Expressivity](https://arxiv.org/abs/2505.21024) (2025)
- [Latent CoT Survey](https://arxiv.org/abs/2505.16782) (2025)

## Citation

If you use this framework, please cite:

```bibtex
@misc{lvrtd2026,
  title={Formalizing Latent Visual Reasoning: A Theoretical Framework for Token Dynamics},
  year={2026},
  note={Research project on latent reasoning mechanisms}
}
```

## License

Research code provided for academic purposes.
