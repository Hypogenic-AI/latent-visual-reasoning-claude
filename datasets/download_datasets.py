#!/usr/bin/env python3
"""
Download datasets for Latent Visual Reasoning research.

This script downloads the required datasets from HuggingFace.
Run with: python download_datasets.py

Requirements:
    pip install datasets pillow
"""

from datasets import load_dataset
import json
import os

DATASETS_DIR = os.path.dirname(os.path.abspath(__file__))


def download_blink():
    """Download BLINK benchmark datasets."""
    print("=" * 60)
    print("Downloading BLINK Benchmark")
    print("=" * 60)

    subtasks = [
        "Counting",
        "Relative_Depth",
        "Spatial_Relation",
        "Visual_Correspondence",
        "Multi-view_Reasoning"
    ]

    for subtask in subtasks:
        print(f"\nDownloading BLINK/{subtask}...")
        try:
            dataset = load_dataset("BLINK-Benchmark/BLINK", subtask)
            save_path = os.path.join(DATASETS_DIR, f"blink_{subtask.lower()}")
            dataset.save_to_disk(save_path)
            print(f"  Saved to: {save_path}")
            print(f"  Val samples: {len(dataset['val'])}")
            print(f"  Test samples: {len(dataset['test'])}")
        except Exception as e:
            print(f"  Error: {e}")


def download_mathvista():
    """Download MathVista dataset."""
    print("\n" + "=" * 60)
    print("Downloading MathVista")
    print("=" * 60)

    try:
        # Download testmini (1000 samples for development)
        print("\nDownloading testmini split...")
        testmini = load_dataset("AI4Math/MathVista", split="testmini")
        save_path = os.path.join(DATASETS_DIR, "mathvista_testmini")
        testmini.save_to_disk(save_path)
        print(f"  Saved to: {save_path}")
        print(f"  Samples: {len(testmini)}")

        # Optionally download full test set
        # test = load_dataset("AI4Math/MathVista", split="test")
        # test.save_to_disk(os.path.join(DATASETS_DIR, "mathvista_test"))

    except Exception as e:
        print(f"  Error: {e}")


def download_vsr():
    """Download Visual Spatial Reasoning dataset."""
    print("\n" + "=" * 60)
    print("Downloading VSR (Visual Spatial Reasoning)")
    print("=" * 60)

    try:
        # Download random split
        print("\nDownloading random split...")
        vsr = load_dataset("cambridgeltl/vsr_random")
        save_path = os.path.join(DATASETS_DIR, "vsr_random")
        vsr.save_to_disk(save_path)
        print(f"  Saved to: {save_path}")
        print(f"  Train samples: {len(vsr['train'])}")
        print(f"  Val samples: {len(vsr['validation'])}")
        print(f"  Test samples: {len(vsr['test'])}")

        # Optionally download zeroshot split
        # vsr_zs = load_dataset("cambridgeltl/vsr_zeroshot")
        # vsr_zs.save_to_disk(os.path.join(DATASETS_DIR, "vsr_zeroshot"))

    except Exception as e:
        print(f"  Error: {e}")


def main():
    print("=" * 60)
    print("Latent Visual Reasoning - Dataset Downloader")
    print("=" * 60)
    print(f"\nDatasets will be saved to: {DATASETS_DIR}")

    download_blink()
    download_mathvista()
    download_vsr()

    print("\n" + "=" * 60)
    print("Download Complete!")
    print("=" * 60)
    print("\nTo load datasets, use:")
    print("  from datasets import load_from_disk")
    print("  dataset = load_from_disk('datasets/blink_counting')")


if __name__ == "__main__":
    main()
