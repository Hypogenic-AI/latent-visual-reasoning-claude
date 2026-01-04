# Resources Catalog

## Summary

This document catalogs all resources gathered for the research project:
**"Formalizing Latent Visual Reasoning: A Theoretical Framework for Token Dynamics"**

| Resource Type | Count |
|---------------|-------|
| Papers | 14 |
| Datasets | 3 (4 subtasks) |
| Code Repositories | 4 |

---

## Papers

Total papers downloaded: **14**

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Latent Visual Reasoning | Li et al. | 2025 | papers/2509.24251_latent_visual_reasoning.pdf | Core LVR paradigm |
| Latent CoT for Visual Reasoning | Various | 2025 | papers/2510.23925_latent_cot_visual_reasoning.pdf | Variational inference approach |
| Machine Mental Imagery (Mirage) | Various | 2025 | papers/2506.17218_machine_mental_imagery_mirage.pdf | Compact latent visual tokens |
| Latent Implicit Visual Reasoning | Various | 2025 | papers/2512.21218_latent_implicit_visual_reasoning.pdf | Implicit latent reasoning |
| Coconut | Hao et al. (Meta) | 2024 | papers/2412.06769_coconut_continuous_latent.pdf | Foundational continuous thought |
| Pause Tokens | Goyal et al. | 2024 | papers/2310.02226_pause_tokens.pdf | ICLR 2024, learnable delays |
| Pause Tokens Expressivity | Various | 2025 | papers/2505.21024_pause_tokens_expressivity.pdf | Theoretical expressivity proof |
| Latent CoT Survey | Chen et al. | 2025 | papers/2505.16782_latent_cot_survey.pdf | Comprehensive taxonomy |
| Rethinking Thinking Tokens | Various | 2024 | papers/2411.11371_thinking_tokens_rethinking.pdf | Analysis of limitations |
| Perception Tokens (AURORA) | Bigverdi et al. | 2024 | papers/2412.03548_perception_tokens_aurora.pdf | Visual reasoning tokens |
| MCOUT | Various | 2025 | papers/2508.12587_mcout_multimodal_chain.pdf | Multimodal continuous thought |
| VSR | Liu et al. | 2023 | papers/2205.00363_vsr_spatial_reasoning.pdf | TACL 2023, spatial reasoning |
| BLINK | Fu et al. | 2024 | papers/2404.12390_blink_benchmark.pdf | ECCV 2024, perception benchmark |
| MathVista | Various | 2024 | papers/2310.02255_mathvista.pdf | ICLR 2024 Oral, visual math |

See [papers/README.md](papers/README.md) for detailed descriptions.

---

## Datasets

Total datasets downloaded: **3** (with multiple subtasks)

| Name | Source | Size | Task | Location | Status |
|------|--------|------|------|----------|--------|
| BLINK (Counting) | HuggingFace | 240 samples | Object counting | datasets/blink_counting/ | Downloaded |
| BLINK (Relative Depth) | HuggingFace | 248 samples | Depth ordering | datasets/blink_relative_depth/ | Downloaded |
| MathVista (testmini) | HuggingFace | 1,000 samples | Visual math | datasets/mathvista_testmini/ | Downloaded |
| VSR (random) | HuggingFace | 10,972 samples | Spatial relations | datasets/vsr_random/ | Downloaded |

See [datasets/README.md](datasets/README.md) for detailed descriptions and download instructions.

**Note**: Datasets are downloaded locally but excluded from git. Use `datasets/download_datasets.py` to reproduce.

---

## Code Repositories

Total repositories cloned: **4**

| Name | URL | Purpose | Location | License |
|------|-----|---------|----------|---------|
| Coconut | github.com/facebookresearch/coconut | Latent reasoning impl | code/coconut/ | MIT |
| BLINK Benchmark | github.com/zeyofu/BLINK_Benchmark | Perception eval | code/blink_benchmark/ | MIT |
| VSR | github.com/cambridgeltl/visual-spatial-reasoning | Spatial eval | code/vsr_benchmark/ | Apache 2.0 |
| MathVista | github.com/lupantech/MathVista | Math reasoning eval | code/mathvista/ | CC BY-SA 4.0 |

See [code/README.md](code/README.md) for detailed descriptions.

---

## Resource Gathering Notes

### Search Strategy

1. **Primary Sources**:
   - arXiv for recent papers (2024-2025)
   - HuggingFace for datasets
   - GitHub for code implementations

2. **Search Keywords**:
   - "latent visual reasoning"
   - "latent chain-of-thought"
   - "continuous thought reasoning"
   - "perception tokens"
   - "pause tokens transformer"
   - "visual reasoning tokens"

3. **Citation Following**:
   - Started with LVR paper
   - Followed references to Coconut, Pause Tokens
   - Identified survey papers for broader coverage

### Selection Criteria

**Papers**:
- Directly related to latent reasoning mechanisms
- Provides theoretical or empirical insights on token dynamics
- Recent work (2023-2025) prioritized
- With code/datasets available preferred

**Datasets**:
- Tests visual perception and reasoning
- Clear gap between human and model performance
- Available on HuggingFace for reproducibility
- Reasonable size for experimentation

**Code**:
- Official implementations preferred
- Well-documented repositories
- Active maintenance

### Challenges Encountered

1. **PDF Extraction**: Some papers had font encoding issues requiring fallback to web versions
2. **Dataset Size**: Full datasets are large; downloaded representative subsets
3. **Code Availability**: Some papers promise future code release; used available alternatives

### Gaps and Workarounds

| Gap | Workaround |
|-----|------------|
| LVR code not yet released | Use Coconut as foundation |
| AURORA code not yet released | Implement based on paper description |
| Some datasets very large | Downloaded key subsets |

---

## Recommendations for Experiment Design

Based on gathered resources:

### 1. Primary Dataset(s)

**BLINK (Counting + Relative_Depth)**
- Why: Directly tests perception abilities that latent reasoning aims to improve
- Gap: Humans 95.7% vs GPT-4V 51.3%
- Clear metrics, standardized evaluation

### 2. Baseline Methods

| Baseline | Source | Purpose |
|----------|--------|---------|
| Coconut | code/coconut/ | Latent reasoning foundation |
| Direct prediction | - | No-reasoning baseline |
| Standard CoT | - | Explicit reasoning baseline |

### 3. Evaluation Metrics

| Metric | Dataset | What It Measures |
|--------|---------|------------------|
| Accuracy | BLINK | Perception task performance |
| Accuracy | VSR | Spatial reasoning |
| Token efficiency | All | Computational efficiency |

### 4. Code to Adapt/Reuse

1. **Coconut** (`code/coconut/`):
   - Core hidden state propagation mechanism
   - Multi-stage training pipeline
   - Special token handling

2. **BLINK Benchmark** (`code/blink_benchmark/`):
   - Evaluation scripts
   - Model interface code

3. **VSR Benchmark** (`code/vsr_benchmark/`):
   - Spatial relation evaluation
   - Baseline comparisons

---

## Quick Start for Experiment Runner

### Step 1: Environment Setup
```bash
# Clone this repository
git clone <repo-url>
cd latent-visual-reasoning-claude

# Install dependencies
pip install datasets torch transformers
```

### Step 2: Download Datasets
```bash
cd datasets
python download_datasets.py
```

### Step 3: Explore Code
```bash
# Review Coconut implementation
cd code/coconut
cat README.md

# Review evaluation code
cd ../blink_benchmark
cat README.md
```

### Step 4: Run Experiments
See `literature_review.md` for recommended experimental design.

---

## File Structure

```
latent-visual-reasoning-claude/
├── papers/
│   ├── README.md                           # Paper catalog
│   ├── 2509.24251_latent_visual_reasoning.pdf
│   ├── 2412.06769_coconut_continuous_latent.pdf
│   └── ... (12 more papers)
├── datasets/
│   ├── README.md                           # Dataset documentation
│   ├── .gitignore                          # Exclude data from git
│   ├── download_datasets.py                # Download script
│   ├── blink_counting/                     # Downloaded data
│   ├── blink_relative_depth/
│   ├── mathvista_testmini/
│   └── vsr_random/
├── code/
│   ├── README.md                           # Code repository catalog
│   ├── coconut/                            # Cloned repo
│   ├── blink_benchmark/
│   ├── vsr_benchmark/
│   └── mathvista/
├── literature_review.md                    # Comprehensive review
├── resources.md                            # This file
└── .resource_finder_complete               # Completion marker
```

---

## Contact and References

### Key Paper Links
- [LVR Paper](https://arxiv.org/abs/2509.24251)
- [Coconut Paper](https://arxiv.org/abs/2412.06769)
- [Latent CoT Survey](https://arxiv.org/abs/2505.16782)

### Dataset Links
- [BLINK on HuggingFace](https://huggingface.co/datasets/BLINK-Benchmark/BLINK)
- [MathVista on HuggingFace](https://huggingface.co/datasets/AI4Math/MathVista)
- [VSR on HuggingFace](https://huggingface.co/datasets/cambridgeltl/vsr_random)

### Code Links
- [Coconut GitHub](https://github.com/facebookresearch/coconut)
- [BLINK GitHub](https://github.com/zeyofu/BLINK_Benchmark)
