"""
Empirical Validation of Theoretical Framework

This module validates the theoretical framework by testing predictions
against real LLM behavior using API calls.

Key experiments:
1. CoT vs Direct prediction on visual reasoning tasks
2. Token count vs reasoning accuracy
3. Probing for latent representations

Author: Research Agent
Date: 2025-01-04
"""

import os
import json
import time
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np
from tqdm import tqdm

# Import API clients
import openai
import anthropic

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


@dataclass
class ExperimentConfig:
    """Configuration for experiments"""
    model: str = "gpt-4o-mini"  # Default to cheaper model
    temperature: float = 0.0  # Deterministic for reproducibility
    max_tokens: int = 500
    num_samples: int = 50  # Number of samples to test
    seed: int = SEED


@dataclass
class ExperimentResult:
    """Result from a single experiment trial"""
    question: str
    ground_truth: str
    prediction: str
    is_correct: bool
    response_tokens: int
    latency_ms: float
    reasoning_type: str  # "direct", "cot", "latent_cot"


class DatasetLoader:
    """Load pre-downloaded datasets"""

    def __init__(self, base_path: str = "datasets"):
        self.base_path = base_path

    def load_blink_counting(self, num_samples: Optional[int] = None) -> List[Dict]:
        """Load BLINK counting dataset"""
        samples_file = os.path.join(self.base_path, "blink_counting_samples.json")
        with open(samples_file, 'r') as f:
            data = json.load(f)
        if num_samples:
            data = data[:num_samples]
        return data

    def load_vsr(self, num_samples: Optional[int] = None) -> List[Dict]:
        """Load VSR dataset samples"""
        samples_file = os.path.join(self.base_path, "vsr_samples.json")
        with open(samples_file, 'r') as f:
            data = json.load(f)
        if num_samples:
            data = data[:num_samples]
        return data

    def load_mathvista(self, num_samples: Optional[int] = None) -> List[Dict]:
        """Load MathVista samples"""
        samples_file = os.path.join(self.base_path, "mathvista_samples.json")
        with open(samples_file, 'r') as f:
            data = json.load(f)
        if num_samples:
            data = data[:num_samples]
        return data


class ReasoningExperimenter:
    """
    Run experiments comparing reasoning approaches.

    Tests theoretical predictions:
    1. More thinking tokens should improve reasoning
    2. Explicit CoT should outperform direct on complex tasks
    3. Structured reasoning should show specific patterns
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.openai_client = None
        self.anthropic_client = None
        self._initialize_clients()

    def _initialize_clients(self):
        """Initialize API clients"""
        # Check for OpenAI key
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key:
            self.openai_client = openai.OpenAI(api_key=openai_key)

        # Check for Anthropic key
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        if anthropic_key:
            self.anthropic_client = anthropic.Anthropic(api_key=anthropic_key)

        # Check for OpenRouter key
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        if openrouter_key and not self.openai_client:
            self.openai_client = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_key
            )

    def _call_openai(
        self,
        prompt: str,
        system: str = "You are a helpful assistant."
    ) -> Tuple[str, int, float]:
        """Call OpenAI API and return response, token count, latency"""
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")

        start = time.time()
        response = self.openai_client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            seed=self.config.seed
        )
        latency = (time.time() - start) * 1000

        content = response.choices[0].message.content
        tokens = response.usage.completion_tokens if response.usage else len(content.split())

        return content, tokens, latency

    def _call_anthropic(
        self,
        prompt: str,
        system: str = "You are a helpful assistant."
    ) -> Tuple[str, int, float]:
        """Call Anthropic API and return response, token count, latency"""
        if not self.anthropic_client:
            raise ValueError("Anthropic client not initialized")

        start = time.time()
        response = self.anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=self.config.max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}]
        )
        latency = (time.time() - start) * 1000

        content = response.content[0].text
        tokens = response.usage.output_tokens

        return content, tokens, latency

    def _extract_answer(self, response: str, choices: List[str]) -> str:
        """Extract answer from model response"""
        response_upper = response.upper()

        # Look for explicit answer patterns
        for i, choice in enumerate(choices):
            letter = chr(ord('A') + i)
            patterns = [
                f"({letter})",
                f"ANSWER: {letter}",
                f"ANSWER IS {letter}",
                f"THE ANSWER IS {letter}",
                f"ANSWER: ({letter})",
            ]
            for pattern in patterns:
                if pattern in response_upper:
                    return f"({letter})"

        # Fallback: look for choice content
        for i, choice in enumerate(choices):
            if choice.lower() in response.lower():
                return f"({chr(ord('A') + i)})"

        # Last resort: first letter found
        for letter in ['A', 'B', 'C', 'D']:
            if letter in response_upper:
                return f"({letter})"

        return "(A)"  # Default

    def run_direct_prediction(
        self,
        question: str,
        choices: List[str]
    ) -> Tuple[str, int, float]:
        """
        Run direct prediction without chain-of-thought.

        This simulates minimal latent reasoning.
        """
        prompt = f"""{question}

Choices:
{chr(10).join([f'({chr(ord("A")+i)}) {c}' for i, c in enumerate(choices)])}

Respond with only the letter of your answer (A, B, C, or D)."""

        system = "You are a visual reasoning assistant. Answer multiple choice questions concisely."

        return self._call_openai(prompt, system)

    def run_cot_prediction(
        self,
        question: str,
        choices: List[str]
    ) -> Tuple[str, int, float]:
        """
        Run chain-of-thought prediction.

        This simulates explicit reasoning tokens.
        """
        prompt = f"""{question}

Choices:
{chr(10).join([f'({chr(ord("A")+i)}) {c}' for i, c in enumerate(choices)])}

Think step by step about this question. Consider what the question is asking,
analyze the choices, and explain your reasoning. Then provide your final answer."""

        system = "You are a visual reasoning assistant. Explain your reasoning step by step before giving your final answer."

        return self._call_openai(prompt, system)

    def run_structured_cot_prediction(
        self,
        question: str,
        choices: List[str],
        num_thinking_steps: int = 3
    ) -> Tuple[str, int, float]:
        """
        Run structured chain-of-thought with explicit thinking steps.

        This simulates multiple latent reasoning tokens.
        """
        thinking_steps = "\n".join([f"Step {i+1}: [analyze one aspect]" for i in range(num_thinking_steps)])

        prompt = f"""{question}

Choices:
{chr(10).join([f'({chr(ord("A")+i)}) {c}' for i, c in enumerate(choices)])}

Follow this structured reasoning format:
{thinking_steps}
Final Answer: [A, B, C, or D]

Now reason through the problem:"""

        system = "You are a visual reasoning assistant that follows structured reasoning formats precisely."

        return self._call_openai(prompt, system)

    def compare_reasoning_approaches(
        self,
        samples: List[Dict],
        approaches: List[str] = ["direct", "cot", "structured_cot"]
    ) -> Dict[str, List[ExperimentResult]]:
        """
        Compare different reasoning approaches on the same samples.

        Tests theoretical prediction: More explicit reasoning improves accuracy.
        """
        results = {approach: [] for approach in approaches}

        for sample in tqdm(samples, desc="Running experiments"):
            question = sample.get("question", sample.get("prompt", ""))
            choices = sample.get("choices", [])
            ground_truth = sample.get("answer", "")

            for approach in approaches:
                try:
                    if approach == "direct":
                        response, tokens, latency = self.run_direct_prediction(question, choices)
                    elif approach == "cot":
                        response, tokens, latency = self.run_cot_prediction(question, choices)
                    elif approach == "structured_cot":
                        response, tokens, latency = self.run_structured_cot_prediction(question, choices)
                    else:
                        continue

                    prediction = self._extract_answer(response, choices)
                    is_correct = prediction.upper() == ground_truth.upper()

                    results[approach].append(ExperimentResult(
                        question=question,
                        ground_truth=ground_truth,
                        prediction=prediction,
                        is_correct=is_correct,
                        response_tokens=tokens,
                        latency_ms=latency,
                        reasoning_type=approach
                    ))

                except Exception as e:
                    print(f"Error with {approach} on sample: {e}")

                # Rate limiting
                time.sleep(0.5)

        return results


class TokenCountExperimenter:
    """
    Test the relationship between token count and reasoning accuracy.

    Tests theoretical prediction: More tokens = better reasoning capacity.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.experimenter = ReasoningExperimenter(config)

    def run_token_count_experiment(
        self,
        samples: List[Dict],
        token_counts: List[int] = [1, 3, 5, 10]
    ) -> Dict[int, Dict[str, float]]:
        """
        Run experiment varying the number of "thinking steps" (proxy for tokens).
        """
        results = {}

        for num_tokens in token_counts:
            print(f"\nRunning with {num_tokens} thinking steps...")
            token_results = []

            for sample in tqdm(samples, desc=f"Tokens={num_tokens}"):
                question = sample.get("question", sample.get("prompt", ""))
                choices = sample.get("choices", [])
                ground_truth = sample.get("answer", "")

                try:
                    response, tokens, latency = self.experimenter.run_structured_cot_prediction(
                        question, choices, num_thinking_steps=num_tokens
                    )
                    prediction = self.experimenter._extract_answer(response, choices)
                    is_correct = prediction.upper() == ground_truth.upper()

                    token_results.append({
                        "is_correct": is_correct,
                        "response_tokens": tokens,
                        "latency_ms": latency
                    })

                except Exception as e:
                    print(f"Error: {e}")

                time.sleep(0.5)

            # Aggregate results
            if token_results:
                results[num_tokens] = {
                    "accuracy": np.mean([r["is_correct"] for r in token_results]),
                    "avg_response_tokens": np.mean([r["response_tokens"] for r in token_results]),
                    "avg_latency_ms": np.mean([r["latency_ms"] for r in token_results]),
                    "n_samples": len(token_results)
                }

        return results


class LatentRepresentationProber:
    """
    Probe for evidence of latent representations in model responses.

    Tests what information is encoded in the reasoning process.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.openai_client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1" if not os.environ.get("OPENAI_API_KEY") else None
        )

    def probe_visual_encoding(
        self,
        question: str,
        probing_questions: List[str]
    ) -> Dict[str, str]:
        """
        Probe what visual information the model encodes during reasoning.
        """
        results = {}

        # First, get the model to reason about the question
        reasoning_prompt = f"""Consider this visual reasoning question:
{question}

Before answering, describe what visual information would be needed to answer this question.
What visual features or relationships would you need to identify?"""

        try:
            response = self.openai_client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": reasoning_prompt}],
                temperature=0.0,
                max_tokens=300
            )
            results["visual_encoding"] = response.choices[0].message.content
        except Exception as e:
            results["visual_encoding"] = f"Error: {e}"

        # Then probe specific aspects
        for probe in probing_questions:
            probe_prompt = f"""Based on this visual reasoning question:
{question}

{probe}"""
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.config.model,
                    messages=[{"role": "user", "content": probe_prompt}],
                    temperature=0.0,
                    max_tokens=150
                )
                results[probe[:50]] = response.choices[0].message.content
            except Exception as e:
                results[probe[:50]] = f"Error: {e}"

            time.sleep(0.3)

        return results


def run_validation_experiments(
    output_dir: str = "results",
    num_samples: int = 20
) -> Dict[str, any]:
    """
    Run all validation experiments.

    Returns comprehensive results comparing theoretical predictions
    with empirical observations.
    """
    os.makedirs(output_dir, exist_ok=True)

    config = ExperimentConfig(
        model="gpt-4o-mini",
        num_samples=num_samples
    )

    # Load datasets
    loader = DatasetLoader()
    blink_samples = loader.load_blink_counting(num_samples)

    results = {
        "config": {
            "model": config.model,
            "num_samples": num_samples,
            "seed": config.seed
        },
        "experiments": {}
    }

    # Experiment 1: Compare reasoning approaches
    print("\n" + "="*60)
    print("Experiment 1: Comparing Reasoning Approaches")
    print("="*60)

    experimenter = ReasoningExperimenter(config)
    approach_results = experimenter.compare_reasoning_approaches(
        blink_samples,
        approaches=["direct", "cot", "structured_cot"]
    )

    # Summarize results
    for approach, exp_results in approach_results.items():
        if exp_results:
            accuracy = np.mean([r.is_correct for r in exp_results])
            avg_tokens = np.mean([r.response_tokens for r in exp_results])
            avg_latency = np.mean([r.latency_ms for r in exp_results])
            print(f"\n{approach}:")
            print(f"  Accuracy: {accuracy:.2%}")
            print(f"  Avg tokens: {avg_tokens:.1f}")
            print(f"  Avg latency: {avg_latency:.0f}ms")

            results["experiments"][f"approach_{approach}"] = {
                "accuracy": accuracy,
                "avg_tokens": avg_tokens,
                "avg_latency_ms": avg_latency,
                "n_samples": len(exp_results)
            }

    # Experiment 2: Token count vs accuracy
    print("\n" + "="*60)
    print("Experiment 2: Token Count vs Accuracy")
    print("="*60)

    token_experimenter = TokenCountExperimenter(config)
    token_results = token_experimenter.run_token_count_experiment(
        blink_samples[:10],  # Smaller sample for this experiment
        token_counts=[1, 3, 5]
    )

    for k, v in token_results.items():
        print(f"\n{k} thinking steps:")
        print(f"  Accuracy: {v['accuracy']:.2%}")
        print(f"  Avg response tokens: {v['avg_response_tokens']:.1f}")

    results["experiments"]["token_count"] = token_results

    # Save results
    results_file = os.path.join(output_dir, "validation_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: x.__dict__ if hasattr(x, '__dict__') else str(x))

    print(f"\nResults saved to {results_file}")

    return results


if __name__ == "__main__":
    results = run_validation_experiments(num_samples=20)
    print("\n" + "="*60)
    print("Validation Complete")
    print("="*60)
