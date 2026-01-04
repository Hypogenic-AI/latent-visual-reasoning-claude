# Downloaded Datasets

This directory contains datasets for the research project on "Formalizing Latent Visual Reasoning: A Theoretical Framework for Token Dynamics."

**Note**: Data files are NOT committed to git due to size. Follow the download instructions below.

## Quick Start

To download all datasets:
```bash
cd datasets
python download_datasets.py
```

Or using the HuggingFace datasets library directly:
```python
from datasets import load_dataset, load_from_disk

# Download from HuggingFace
dataset = load_dataset("BLINK-Benchmark/BLINK", "Counting")

# Or load pre-downloaded data
dataset = load_from_disk("datasets/blink_counting")
```

---

## Dataset 1: BLINK Benchmark

### Overview
- **Source**: [BLINK-Benchmark/BLINK](https://huggingface.co/datasets/BLINK-Benchmark/BLINK)
- **Paper**: BLINK: Multimodal Large Language Models Can See but Not Perceive (ECCV 2024)
- **arXiv**: [2404.12390](https://arxiv.org/abs/2404.12390)
- **Size**: 3,807 multiple-choice questions across 14 subtasks
- **Format**: HuggingFace Dataset with images
- **Task**: Visual perception (depth, counting, correspondence, etc.)
- **Splits**: val (1,901), test (1,906)
- **License**: CC BY 4.0

### Key Subtasks for This Research
- **Counting**: Object counting in images (120 val, 120 test)
- **Relative_Depth**: Depth ordering of points (124 val, 124 test)
- **Spatial_Relation**: Spatial relationship reasoning
- **Visual_Correspondence**: Matching across views

### Download Instructions

**Using HuggingFace:**
```python
from datasets import load_dataset

# Download specific subtask
dataset = load_dataset("BLINK-Benchmark/BLINK", "Counting")
dataset.save_to_disk("datasets/blink_counting")

# Available subtasks:
# Art_Style, Functional_Correspondence, Multi-view_Reasoning, Relative_Reflectance,
# Visual_Correspondence, Counting, IQ_Test, Object_Localization, Semantic_Correspondence,
# Visual_Similarity, Forensic_Detection, Jigsaw, Relative_Depth, Spatial_Relation
```

### Loading the Dataset
```python
from datasets import load_from_disk

dataset = load_from_disk("datasets/blink_counting")
example = dataset['val'][0]
print(example['question'])
print(example['choices'])
print(example['answer'])
```

### Why This Dataset
- Tests fundamental visual perception that MLLMs struggle with
- Humans: 95.7% vs GPT-4V: 51.3% - shows perception gap
- Tasks "resist mediation through natural language"
- Ideal for evaluating latent visual reasoning approaches

---

## Dataset 2: MathVista

### Overview
- **Source**: [AI4Math/MathVista](https://huggingface.co/datasets/AI4Math/MathVista)
- **Paper**: MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts (ICLR 2024 Oral)
- **arXiv**: [2310.02255](https://arxiv.org/abs/2310.02255)
- **Size**: 6,141 examples from 31 datasets
- **Format**: HuggingFace Dataset with images
- **Task**: Visual mathematical reasoning
- **Splits**: testmini (1,000), test (5,141)
- **License**: CC BY-SA 4.0 (test use only)

### Download Instructions

**Using HuggingFace:**
```python
from datasets import load_dataset

# Download testmini (development set)
dataset = load_dataset("AI4Math/MathVista", split="testmini")
dataset.save_to_disk("datasets/mathvista_testmini")

# Download full test set
dataset = load_dataset("AI4Math/MathVista", split="test")
dataset.save_to_disk("datasets/mathvista_test")
```

### Loading the Dataset
```python
from datasets import load_from_disk

dataset = load_from_disk("datasets/mathvista_testmini")
example = dataset[0]
print(example['question'])
print(example['choices'])
print(example['answer'])
# Access image: example['decoded_image']
```

### Why This Dataset
- Combines visual perception with mathematical reasoning
- Tests diverse visual contexts: charts, diagrams, geometry, scenes
- Requires both "seeing" and "reasoning" - ideal for latent reasoning study

---

## Dataset 3: VSR (Visual Spatial Reasoning)

### Overview
- **Source**: [cambridgeltl/vsr_random](https://huggingface.co/datasets/cambridgeltl/vsr_random)
- **Paper**: Visual Spatial Reasoning (TACL 2023)
- **arXiv**: [2205.00363](https://arxiv.org/abs/2205.00363)
- **Size**: 10,972 text-image pairs with 66 spatial relation types
- **Format**: HuggingFace Dataset (requires COCO images)
- **Task**: Binary classification - is spatial caption true/false?
- **Splits**: train (7,680), validation (1,097), test (2,195)
- **License**: Apache 2.0

### Download Instructions

**Using HuggingFace:**
```python
from datasets import load_dataset

# Download random split
dataset = load_dataset("cambridgeltl/vsr_random")
dataset.save_to_disk("datasets/vsr_random")

# Download zero-shot split (for testing on unseen relations)
dataset = load_dataset("cambridgeltl/vsr_zeroshot")
dataset.save_to_disk("datasets/vsr_zeroshot")
```

**Note**: VSR uses COCO images. You may need to download images separately:
```bash
# Images from COCO dataset
wget http://images.cocodataset.org/zips/val2017.zip
```

### Loading the Dataset
```python
from datasets import load_from_disk

dataset = load_from_disk("datasets/vsr_random")
example = dataset['test'][0]
print(example['caption'])  # e.g., "The cat is under the table"
print(example['label'])    # True or False
print(example['relation']) # e.g., "under"
```

### Why This Dataset
- Tests 66 types of spatial relations
- Humans: 95%+ vs models: ~70% - reveals spatial reasoning gap
- All models at chance level for "orientation" relations
- Ideal for testing visual reasoning in latent space

---

## Sample Data

Sample data files (without images) are included for reference:
- `blink_counting_samples.json` - 5 examples from BLINK Counting
- `mathvista_samples.json` - 5 examples from MathVista
- `vsr_samples.json` - 5 examples from VSR

---

## Dataset Comparison

| Dataset | Size | Task Type | Key Challenge | Human vs SOTA |
|---------|------|-----------|---------------|---------------|
| BLINK | 3,807 | Multi-task perception | Core visual perception | 95.7% vs 51.3% |
| MathVista | 6,141 | Visual math reasoning | Multi-step reasoning | 60.3% vs 63.8% |
| VSR | 10,972 | Spatial relations | Orientation relations | 95%+ vs 70% |

---

## Recommendations for Experiments

Based on literature review, we recommend:

1. **Primary Benchmark**: BLINK (Counting + Relative_Depth subtasks)
   - Directly tests perception abilities latent reasoning aims to improve
   - Clear metrics, large performance gap to close

2. **Secondary Benchmark**: MathVista testmini
   - Tests combined perception + reasoning
   - Good for evaluating chain-of-thought improvements

3. **Diagnostic Benchmark**: VSR
   - Tests specific spatial reasoning capabilities
   - Useful for understanding model limitations

---

## Storage Information

Approximate storage requirements:
- BLINK (all subtasks): ~5GB
- MathVista (testmini): ~2GB
- VSR (random): ~500MB (without images)

Note: BLINK and MathVista include images directly. VSR references external COCO images.
