"""
Main Experiment Runner for Latent Visual Reasoning Framework

This script orchestrates all experiments:
1. Theoretical framework analysis
2. Empirical validation with real LLMs
3. Comparative analysis
4. Result visualization

Author: Research Agent
Date: 2025-01-04
"""

import os
import sys
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from theoretical_framework import (
    ContinuousThoughtMechanism,
    PauseTokenMechanism,
    PerceptionTokenMechanism,
    InformationFlowAnalyzer,
    ComputationalExpressivityAnalyzer,
    ReconstructionReasoningAnalyzer,
    LatentState,
    create_framework_summary
)

# Set random seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


class TheoreticalAnalysis:
    """
    Run theoretical analysis of latent token mechanisms.

    This computes expressivity bounds and other theoretical properties
    without requiring API calls.
    """

    def __init__(self, embedding_dim: int = 768, max_tokens: int = 10):
        self.embedding_dim = embedding_dim
        self.max_tokens = max_tokens

    def analyze_all_mechanisms(self) -> Dict[str, Any]:
        """Analyze all three token mechanisms"""
        results = {}

        # Create mechanisms with varying token counts
        for k in [1, 3, 5, 10]:
            ct = ContinuousThoughtMechanism(k, self.embedding_dim)
            pt = PauseTokenMechanism(k, self.embedding_dim)
            pk = PerceptionTokenMechanism(k, self.embedding_dim)

            results[f"k={k}"] = {
                "continuous_thought": ct.compute_expressivity_bound(),
                "pause_token": pt.compute_expressivity_bound(),
                "perception_token": pk.compute_expressivity_bound()
            }

        return results

    def analyze_information_flow(self) -> Dict[str, Any]:
        """Simulate and analyze information flow through latent tokens"""
        results = {}

        # Simulate a reasoning trajectory
        num_steps = 5
        ct = ContinuousThoughtMechanism(num_steps, self.embedding_dim)

        # Generate trajectory
        states = [ct.initialize_state()]
        for _ in range(num_steps - 1):
            context = np.random.randn(100)  # Simulated context
            new_state = ct.reason_step(states[-1], context)
            states.append(new_state)

        # Analyze trajectory
        trajectory_analysis = InformationFlowAnalyzer.analyze_trajectory(states)
        results["trajectory"] = trajectory_analysis

        # Compute information preservation across different step counts
        preservation_by_steps = {}
        for n in [1, 3, 5, 10]:
            if n <= len(states):
                preservation = InformationFlowAnalyzer.compute_information_preservation(
                    states[0], states[min(n, len(states)-1)]
                )
                preservation_by_steps[n] = preservation

        results["preservation_by_steps"] = preservation_by_steps

        return results

    def compare_expressivity(self) -> Dict[str, Any]:
        """Compare expressivity across mechanisms"""
        mechanisms = [
            ContinuousThoughtMechanism(5, self.embedding_dim),
            PauseTokenMechanism(5, self.embedding_dim),
            PerceptionTokenMechanism(5, self.embedding_dim)
        ]

        return ComputationalExpressivityAnalyzer.compare_mechanisms(mechanisms)

    def analyze_reconstruction_reasoning(self) -> Dict[str, Any]:
        """Analyze reconstruction-reasoning trade-off"""
        results = {}

        # Different compression ratios
        for input_dim in [1024, 2048, 4096]:
            for num_tokens in [1, 5, 10]:
                key = f"input_{input_dim}_tokens_{num_tokens}"
                results[key] = ReconstructionReasoningAnalyzer.compute_reconstruction_information_bound(
                    input_dim=input_dim,
                    hidden_dim=self.embedding_dim,
                    num_tokens=num_tokens
                )

        # Trade-off analysis
        trade_offs = []
        for recon_w in [0.1, 0.3, 0.5, 0.7, 0.9]:
            trade_offs.append(
                ReconstructionReasoningAnalyzer.analyze_reconstruction_reasoning_tradeoff(
                    reconstruction_weight=recon_w,
                    reasoning_weight=1.0 - recon_w
                )
            )
        results["trade_offs"] = trade_offs

        return results


class ResultVisualizer:
    """Generate visualizations for the research results"""

    def __init__(self, output_dir: str = "results/figures"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")

    def plot_expressivity_comparison(
        self,
        theoretical_results: Dict,
        save_name: str = "expressivity_comparison.png"
    ):
        """Plot expressivity comparison across token types"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        token_counts = [1, 3, 5, 10]

        # Extract data for each mechanism
        ct_depths = []
        pt_depths = []
        pk_bits = []

        for k in token_counts:
            key = f"k={k}"
            ct_depths.append(theoretical_results[key]["continuous_thought"]["computational_depth"])
            pt_depths.append(theoretical_results[key]["pause_token"]["computational_depth_increase"])
            pk_bits.append(theoretical_results[key]["perception_token"]["information_capacity_bits"])

        # Plot 1: Computational depth
        axes[0].bar(['CT', 'PT'], [ct_depths[-1], pt_depths[-1]], color=['#2ecc71', '#3498db'])
        axes[0].set_ylabel('Computational Depth')
        axes[0].set_title('Computational Depth (k=10)')

        # Plot 2: Depth vs token count
        axes[1].plot(token_counts, ct_depths, 'o-', label='Continuous Thought', linewidth=2)
        axes[1].plot(token_counts, pt_depths, 's--', label='Pause Token', linewidth=2)
        axes[1].set_xlabel('Number of Tokens')
        axes[1].set_ylabel('Computational Depth')
        axes[1].set_title('Depth Scaling with Token Count')
        axes[1].legend()

        # Plot 3: Information capacity
        axes[2].bar(token_counts, pk_bits, color='#9b59b6')
        axes[2].set_xlabel('Number of Tokens')
        axes[2].set_ylabel('Information Capacity (bits)')
        axes[2].set_title('Perception Token Information Capacity')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, save_name), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_name}")

    def plot_information_flow(
        self,
        flow_results: Dict,
        save_name: str = "information_flow.png"
    ):
        """Plot information flow analysis"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot 1: Information preservation by step
        steps = list(flow_results["preservation_by_steps"].keys())
        preservation = list(flow_results["preservation_by_steps"].values())

        axes[0].plot(steps, preservation, 'o-', linewidth=2, markersize=8, color='#e74c3c')
        axes[0].set_xlabel('Number of Reasoning Steps')
        axes[0].set_ylabel('Information Preservation (cosine similarity)')
        axes[0].set_title('Information Preservation Through Reasoning')
        axes[0].set_ylim([0, 1])
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Trajectory analysis summary
        traj = flow_results["trajectory"]
        metrics = ['avg_step_similarity', 'norm_growth']
        values = [traj['avg_step_similarity'], min(traj['norm_growth'], 2)]  # Cap norm growth for visibility

        axes[1].bar(metrics, values, color=['#3498db', '#2ecc71'])
        axes[1].set_ylabel('Value')
        axes[1].set_title('Reasoning Trajectory Properties')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, save_name), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_name}")

    def plot_reconstruction_tradeoff(
        self,
        recon_results: Dict,
        save_name: str = "reconstruction_tradeoff.png"
    ):
        """Plot reconstruction-reasoning trade-off"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot 1: Compression ratio vs info loss
        configs = []
        losses = []
        for key, val in recon_results.items():
            if key != "trade_offs" and "tokens" in key:
                configs.append(key.replace("input_", "").replace("_tokens_", ", k="))
                losses.append(val.get("information_loss_estimate", 0))

        axes[0].barh(configs, losses, color='#e67e22')
        axes[0].set_xlabel('Estimated Information Loss')
        axes[0].set_title('Information Loss by Configuration')

        # Plot 2: Trade-off curve
        trade_offs = recon_results.get("trade_offs", [])
        if trade_offs:
            recon_emphasis = [t["reconstruction_emphasis"] for t in trade_offs]
            reason_emphasis = [t["reasoning_emphasis"] for t in trade_offs]

            axes[1].plot(recon_emphasis, reason_emphasis, 'o-', linewidth=2, markersize=10, color='#9b59b6')
            axes[1].set_xlabel('Reconstruction Emphasis')
            axes[1].set_ylabel('Reasoning Emphasis')
            axes[1].set_title('Reconstruction-Reasoning Trade-off')
            axes[1].plot([0, 1], [1, 0], '--', alpha=0.5, color='gray', label='Trade-off line')
            axes[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, save_name), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_name}")

    def plot_empirical_results(
        self,
        empirical_results: Dict,
        save_name: str = "empirical_validation.png"
    ):
        """Plot empirical validation results"""
        if not empirical_results or "experiments" not in empirical_results:
            print("No empirical results to plot")
            return

        experiments = empirical_results["experiments"]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot 1: Approach comparison
        approach_data = {k.replace("approach_", ""): v
                        for k, v in experiments.items()
                        if k.startswith("approach_")}

        if approach_data:
            approaches = list(approach_data.keys())
            accuracies = [approach_data[a].get("accuracy", 0) for a in approaches]
            tokens = [approach_data[a].get("avg_tokens", 0) for a in approaches]

            x = np.arange(len(approaches))
            width = 0.35

            axes[0].bar(x - width/2, accuracies, width, label='Accuracy', color='#2ecc71')
            ax2 = axes[0].twinx()
            ax2.bar(x + width/2, tokens, width, label='Tokens', color='#3498db', alpha=0.7)

            axes[0].set_xticks(x)
            axes[0].set_xticklabels(approaches, rotation=15)
            axes[0].set_ylabel('Accuracy', color='#2ecc71')
            ax2.set_ylabel('Avg Response Tokens', color='#3498db')
            axes[0].set_title('Reasoning Approach Comparison')
            axes[0].set_ylim([0, 1])

        # Plot 2: Token count experiment
        token_data = experiments.get("token_count", {})
        if token_data:
            token_counts = sorted([int(k) for k in token_data.keys()])
            accuracies = [token_data.get(str(k), {}).get("accuracy", 0) for k in token_counts]

            axes[1].plot(token_counts, accuracies, 'o-', linewidth=2, markersize=8, color='#e74c3c')
            axes[1].set_xlabel('Number of Thinking Steps')
            axes[1].set_ylabel('Accuracy')
            axes[1].set_title('Accuracy vs Thinking Steps')
            axes[1].set_ylim([0, 1])
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, save_name), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_name}")

    def create_summary_table(
        self,
        theoretical_results: Dict,
        empirical_results: Dict = None
    ) -> str:
        """Create a summary table as markdown"""
        lines = ["# Summary of Results\n"]

        lines.append("## Theoretical Analysis\n")
        lines.append("| Token Type | Key Property | Value |")
        lines.append("|------------|--------------|-------|")

        for k in ["k=5"]:
            if k in theoretical_results:
                ct = theoretical_results[k]["continuous_thought"]
                pt = theoretical_results[k]["pause_token"]
                pk = theoretical_results[k]["perception_token"]

                lines.append(f"| Continuous Thought | Computational Depth | {ct['computational_depth']} |")
                lines.append(f"| Continuous Thought | Can Encode Multiple Paths | {ct['can_encode_multiple_paths']} |")
                lines.append(f"| Pause Token | Circuit Depth Boost | {pt['circuit_depth_boost']} |")
                lines.append(f"| Pause Token | Can Encode Multiple Paths | {pt['can_encode_multiple_paths']} |")
                lines.append(f"| Perception Token | Visual States | {pk['visual_capacity']} |")

        if empirical_results and "experiments" in empirical_results:
            lines.append("\n## Empirical Validation\n")
            lines.append("| Approach | Accuracy | Avg Tokens |")
            lines.append("|----------|----------|------------|")

            for k, v in empirical_results["experiments"].items():
                if k.startswith("approach_"):
                    name = k.replace("approach_", "")
                    acc = v.get("accuracy", 0)
                    tokens = v.get("avg_tokens", 0)
                    lines.append(f"| {name} | {acc:.1%} | {tokens:.0f} |")

        return "\n".join(lines)


def run_all_experiments(
    run_empirical: bool = True,
    num_samples: int = 15,
    output_dir: str = "results"
) -> Dict[str, Any]:
    """
    Run all experiments and generate results.

    Args:
        run_empirical: Whether to run API-based empirical experiments
        num_samples: Number of samples for empirical experiments
        output_dir: Directory for saving results

    Returns:
        Dictionary with all results
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "seed": SEED,
            "embedding_dim": 768,
            "run_empirical": run_empirical,
            "num_samples": num_samples
        }
    }

    # 1. Theoretical Analysis
    print("=" * 60)
    print("PHASE 1: Theoretical Analysis")
    print("=" * 60)

    theoretical = TheoreticalAnalysis()

    print("\n1.1 Analyzing all mechanisms...")
    results["mechanism_analysis"] = theoretical.analyze_all_mechanisms()

    print("1.2 Analyzing information flow...")
    results["information_flow"] = theoretical.analyze_information_flow()

    print("1.3 Comparing expressivity...")
    results["expressivity_comparison"] = theoretical.compare_expressivity()

    print("1.4 Analyzing reconstruction-reasoning trade-off...")
    results["reconstruction_reasoning"] = theoretical.analyze_reconstruction_reasoning()

    print("1.5 Creating framework summary...")
    results["framework_summary"] = create_framework_summary()

    # 2. Empirical Validation (if enabled)
    if run_empirical:
        print("\n" + "=" * 60)
        print("PHASE 2: Empirical Validation")
        print("=" * 60)

        # Check for API keys
        has_openai = bool(os.environ.get("OPENAI_API_KEY"))
        has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
        has_openrouter = bool(os.environ.get("OPENROUTER_API_KEY"))

        if has_openai or has_openrouter:
            try:
                from empirical_validation import run_validation_experiments
                empirical_results = run_validation_experiments(
                    output_dir=output_dir,
                    num_samples=num_samples
                )
                results["empirical_validation"] = empirical_results
            except Exception as e:
                print(f"Empirical validation failed: {e}")
                results["empirical_validation"] = {"error": str(e)}
        else:
            print("No API keys found. Skipping empirical validation.")
            print("Set OPENAI_API_KEY or OPENROUTER_API_KEY to enable.")
            results["empirical_validation"] = {"status": "skipped", "reason": "no_api_keys"}

    # 3. Generate Visualizations
    print("\n" + "=" * 60)
    print("PHASE 3: Generating Visualizations")
    print("=" * 60)

    visualizer = ResultVisualizer(os.path.join(output_dir, "figures"))

    visualizer.plot_expressivity_comparison(results["mechanism_analysis"])
    visualizer.plot_information_flow(results["information_flow"])
    visualizer.plot_reconstruction_tradeoff(results["reconstruction_reasoning"])

    if "empirical_validation" in results and "experiments" in results.get("empirical_validation", {}):
        visualizer.plot_empirical_results(results["empirical_validation"])

    # 4. Create Summary
    summary_table = visualizer.create_summary_table(
        results["mechanism_analysis"],
        results.get("empirical_validation")
    )

    with open(os.path.join(output_dir, "summary_table.md"), "w") as f:
        f.write(summary_table)

    # 5. Save all results
    results_file = os.path.join(output_dir, "all_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nAll results saved to {results_file}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run latent visual reasoning experiments")
    parser.add_argument("--no-empirical", action="store_true", help="Skip empirical validation")
    parser.add_argument("--samples", type=int, default=15, help="Number of samples for empirical tests")
    parser.add_argument("--output", type=str, default="results", help="Output directory")

    args = parser.parse_args()

    results = run_all_experiments(
        run_empirical=not args.no_empirical,
        num_samples=args.samples,
        output_dir=args.output
    )

    print("\n" + "=" * 60)
    print("Experiments Complete!")
    print("=" * 60)
    print(f"\nResults saved to: {args.output}/")
    print(f"- all_results.json: Complete results")
    print(f"- summary_table.md: Summary table")
    print(f"- figures/: Visualizations")
