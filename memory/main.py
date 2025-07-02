#!/usr/bin/env python3
"""
LOCOMO Memory System - Unified Evaluation
========================================
Minimal implementation supporting multiple memory systems with comparative benchmarking.
"""

import dspy
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot, MIPROv2
import json
import os
from pathlib import Path
import click
from typing import Optional, Protocol, Dict, Any
from abc import abstractmethod

# Import LOCOMO components
from locomo.dataset import load_locomo_dataset
from locomo.evaluate import LOCOMOMetric
from locomo.llm_judge import create_locomo_judge

# Import memory systems
from simple_memory import DSPyMemory, LOCOMOMemory
from graph_memory import MemorySystem as GraphMemory

# Default configuration
DEFAULT_MODEL = "together_ai/Qwen/Qwen3-235B-A22B-fp8-tput"
MAX_TOKENS = 30000


class MemoryInterface(Protocol):
    """Common interface for all memory systems."""

    @abstractmethod
    def forward(self, conversation: str, question: str) -> dspy.Prediction:
        """Process conversation and answer question."""
        pass


class SimpleMemoryAdapter(dspy.Module):
    """Adapter for DSPyMemory to match common interface."""

    def __init__(self):
        super().__init__()
        self.memory = DSPyMemory()

    def forward(self, conversation: str, question: str) -> dspy.Prediction:
        return self.memory.forward(conversation, question)


class GraphMemoryAdapter(dspy.Module):
    """Adapter for GraphMemory to match common interface."""

    def __init__(self):
        super().__init__()
        self.memory = GraphMemory(persist=False, resource_limit=MAX_TOKENS)

    def forward(self, conversation: str, question: str) -> dspy.Prediction:
        # Clear previous state for fair comparison
        self.memory.state = ""
        self.memory.update(conversation)
        answer = self.memory.query(question)
        return dspy.Prediction(answer=answer)


class BaselineMemory(dspy.Module):
    """Baseline without explicit memory - direct QA."""

    def __init__(self):
        super().__init__()
        self.model = LOCOMOMemory()

    def forward(self, conversation: str, question: str) -> dspy.Prediction:
        return self.model.forward(conversation, question)


def get_memory_system(system_type: str) -> MemoryInterface:
    """Factory for memory systems."""
    systems = {
        "baseline": BaselineMemory,
        "simple": SimpleMemoryAdapter,
        "graph": GraphMemoryAdapter,
    }

    if system_type not in systems:
        raise ValueError(
            f"Unknown memory system: {system_type}. Choose from {list(systems.keys())}"
        )

    return systems[system_type]()


def configure_lm():
    """Configure language model."""
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError("Set TOGETHER_API_KEY environment variable")

    lm = dspy.LM(DEFAULT_MODEL, api_key=api_key, max_tokens=MAX_TOKENS)
    dspy.configure(lm=lm)


@click.group()
def cli():
    """LOCOMO Memory System Evaluation CLI"""
    pass


@cli.command()
@click.option(
    "--train-data", default="data/locomo_train.json", help="Training data path"
)
@click.option("--output", default="optimized_model.json", help="Output path")
@click.option(
    "--system",
    default="baseline",
    type=click.Choice(["baseline", "simple", "graph"]),
    help="Memory system type",
)
@click.option(
    "--method",
    default="bootstrap",
    type=click.Choice(["bootstrap", "mipro"]),
    help="Optimization method",
)
@click.option("--num-demos", default=5, help="Number of demonstrations")
@click.option("--limit", default=50, help="Training examples limit")
def optimize(train_data, output, system, method, num_demos, limit):
    """Optimize memory system using DSPy."""
    configure_lm()

    # Load data and create model
    train = load_locomo_dataset(train_data)[:limit]
    model = get_memory_system(system)

    # Use LOCOMO metric
    metric = LOCOMOMetric()

    # Optimize
    if method == "bootstrap":
        optimizer = BootstrapFewShot(metric=metric, max_bootstrapped_demos=num_demos)
    else:
        optimizer = MIPROv2(metric=metric, num_candidates=10, init_temperature=0.7)

    optimized = optimizer.compile(model, trainset=train)

    # Save with metadata
    output_path = output.replace(".json", f"_{system}.json")
    optimized.save(output_path)
    click.echo(f"Saved optimized {system} model to {output_path}")


@cli.command()
@click.option("--test-data", default="data/locomo_test.json", help="Test data path")
@click.option(
    "--system",
    default="baseline",
    type=click.Choice(["baseline", "simple", "graph"]),
    help="Memory system type",
)
@click.option("--model-path", help="Path to optimized model")
@click.option("--limit", type=int, help="Limit test examples")
@click.option("--threads", default=8, help="Number of threads")
@click.option("--output", help="Output JSON file for results")
def evaluate(test_data, system, model_path, limit, threads, output):
    """Evaluate memory system using LOCOMO LLM judge."""
    configure_lm()

    # Load model
    model = get_memory_system(system)
    if model_path and Path(model_path).exists():
        model.load(model_path)

    # Load test data
    test = load_locomo_dataset(test_data)[:limit]

    # Evaluate with LOCOMO metric
    evaluator = Evaluate(
        devset=test, metric=LOCOMOMetric(), num_threads=threads, display_progress=True
    )

    score = evaluator(model)

    # Save detailed results if requested
    if output:
        results = {
            "system": system,
            "model_path": model_path,
            "test_data": test_data,
            "num_examples": len(test),
            "overall_score": score,
            "model_type": system,
            "optimized": model_path is not None,
        }

        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        click.echo(f"Saved results to {output}")

    click.echo(f"\nScore ({system}): {score:.1%}")


@cli.command()
@click.option("--test-data", default="data/locomo_test.json", help="Test data path")
@click.option(
    "--systems", default="baseline,simple,graph", help="Comma-separated list of systems"
)
@click.option("--limit", default=10, help="Number of examples")
@click.option("--threads", default=8, help="Number of threads")
@click.option("--output-dir", default="results", help="Output directory")
def compare(test_data, systems, limit, threads, output_dir):
    """Compare multiple memory systems with and without optimization."""
    configure_lm()

    systems_list = systems.split(",")
    test = load_locomo_dataset(test_data)[:limit]

    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # Evaluate each system
    results = {}
    evaluator = Evaluate(
        devset=test, metric=LOCOMOMetric(), num_threads=threads, display_progress=True
    )

    for system_name in systems_list:
        click.echo(f"\n{'='*60}")
        click.echo(f"Evaluating {system_name.upper()}")
        click.echo(f"{'='*60}")

        # Base model
        click.echo("\nBase model (no optimization):")
        base_model = get_memory_system(system_name)
        base_score = evaluator(base_model)

        # Optimized model
        opt_path = f"optimized_model_{system_name}.json"
        opt_score = None

        if Path(opt_path).exists():
            click.echo(f"\nOptimized model ({opt_path}):")
            opt_model = get_memory_system(system_name)
            opt_model.load(opt_path)
            opt_score = evaluator(opt_model)

        results[system_name] = {
            "base_score": base_score,
            "optimized_score": opt_score,
            "improvement": opt_score - base_score if opt_score else None,
        }

    # Display comparison table
    click.echo(f"\n{'='*60}")
    click.echo("COMPARISON RESULTS")
    click.echo(f"{'='*60}")
    click.echo(f"{'System':<15} {'Base':<10} {'Optimized':<10} {'Delta':<10}")
    click.echo(f"{'-'*45}")

    for system, scores in results.items():
        base = f"{scores['base_score']:.1%}"
        opt = f"{scores['optimized_score']:.1%}" if scores["optimized_score"] else "N/A"
        delta = f"{scores['improvement']:+.1%}" if scores["improvement"] else "N/A"
        click.echo(f"{system:<15} {base:<10} {opt:<10} {delta:<10}")

    # Save results
    output_file = Path(output_dir) / "comparison_results.json"
    with open(output_file, "w") as f:
        json.dump(
            {"test_data": test_data, "num_examples": len(test), "systems": results},
            f,
            indent=2,
        )

    click.echo(f"\nSaved results to {output_file}")


@cli.command()
@click.option("--test-data", default="data/locomo_test.json", help="Test data path")
@click.option("--systems", default="baseline,simple,graph", help="Systems to benchmark")
@click.option("--configs", default="base,optimized", help="Configurations to test")
@click.option("--limit", default=10, help="Examples per evaluation")
@click.option("--threads", default=8, help="Number of threads")
@click.option("--output", default="benchmark_results.json", help="Output file")
def benchmark(test_data, systems, configs, limit, threads, output):
    """Run comprehensive benchmark across systems and configurations."""
    configure_lm()

    systems_list = systems.split(",")
    configs_list = configs.split(",")
    test = load_locomo_dataset(test_data)[:limit]

    evaluator = Evaluate(
        devset=test,
        metric=LOCOMOMetric(),
        num_threads=threads,
        display_progress=False,  # Less verbose for benchmark
    )

    results = {"benchmarks": []}

    for system_name in systems_list:
        for config in configs_list:
            click.echo(f"\nBenchmarking {system_name} ({config})...")

            model = get_memory_system(system_name)

            if config == "optimized":
                opt_path = f"optimized_model_{system_name}.json"
                if Path(opt_path).exists():
                    model.load(opt_path)
                else:
                    click.echo(f"  Skipping - no optimized model found")
                    continue

            score = evaluator(model)

            results["benchmarks"].append(
                {
                    "system": system_name,
                    "config": config,
                    "score": score,
                    "num_examples": len(test),
                }
            )

            click.echo(f"  Score: {score:.1%}")

    # Summary statistics
    results["summary"] = {
        "test_data": test_data,
        "num_examples": len(test),
        "best_system": max(results["benchmarks"], key=lambda x: x["score"]),
    }

    with open(output, "w") as f:
        json.dump(results, f, indent=2)

    click.echo(f"\nBenchmark complete. Results saved to {output}")


@cli.command()
@click.argument("question")
@click.option("--conversation", help="Conversation context")
@click.option(
    "--system",
    default="baseline",
    type=click.Choice(["baseline", "simple", "graph"]),
    help="Memory system type",
)
@click.option("--model-path", help="Path to optimized model")
def ask(question, conversation, system, model_path):
    """Interactive Q&A with chosen memory system."""
    configure_lm()

    # Load model
    model = get_memory_system(system)
    if model_path and Path(model_path).exists():
        model.load(model_path)

    # Get conversation if not provided
    if not conversation:
        conversation = click.prompt("Enter conversation context")

    # Get answer
    pred = model(conversation=conversation, question=question)
    click.echo(f"\nAnswer ({system}): {pred.answer}")

    # Optional: Show LLM judge evaluation
    if click.confirm("\nEvaluate with LLM judge?", default=False):
        expected = click.prompt("Expected answer")
        judge = create_locomo_judge()
        evaluation = judge.evaluate_answer(
            question=question, ground_truth=expected, generated_answer=pred.answer
        )
        click.echo(f"Judgment: {evaluation['judgment']}")
        click.echo(f"Reasoning: {evaluation['reasoning']}")


if __name__ == "__main__":
    cli()
