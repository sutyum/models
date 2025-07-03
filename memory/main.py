#!/usr/bin/env python3
"""
LOCOMO Memory System - Unified Evaluation
========================================
Minimal implementation supporting multiple memory systems with comparative benchmarking.
"""

import dspy
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot, MIPROv2, SIMBA
import json
import os
from pathlib import Path
import click
from typing import Protocol
from abc import abstractmethod

# Import LOCOMO components
from locomo.dataset import load_locomo_dataset
from locomo.evaluate import LOCOMOMetric
from locomo.llm_judge import create_locomo_judge

# Import memory systems
from simple_memory import DSPyMemory, LOCOMOMemory
from graph_memory import MemorySystem as GraphMemory

# Default configuration
DEFAULT_MODEL = "gemini_flash"
MAX_TOKENS = 35000

# Model configurations
MODELS = {
    "qwen": "groq/qwen/qwen3-32b",
    "deepseek": "together_ai/deepseek-ai/DeepSeek-R1-0528-tput",
    "o4": "o4-mini",
    "gemini_lite": "gemini/gemini-2.5-flash-lite-preview-06-17",
    "gemini_flash": "gemini/gemini-2.5-flash",
    "gemini_pro": "gemini/gemini-2.5-pro",
}


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


def get_memory_system(system_type: str = "graph") -> MemoryInterface:
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


def load_optimized_model(system_type: str, model_path: str) -> MemoryInterface:
    """Enhanced loader for optimized models with program/JSON fallback."""
    if not Path(model_path).exists():
        return get_memory_system(system_type)
        
    # Try program load first
    program_dir = model_path.replace(".json", "_program")
    if Path(program_dir).exists():
        try:
            return dspy.load(program_dir)
        except Exception as e:
            click.echo(f"Warning: Program load failed: {e}")
    
    # Fallback to JSON load
    memory_model = get_memory_system(system_type)
    try:
        memory_model.load(model_path)
    except Exception as e:
        click.echo(f"Warning: JSON load failed: {e}")
    
    return memory_model


def configure_lm(model: str):
    """Configure language model with provider detection."""
    model_path = MODELS.get(model, model)  # Allow custom model paths

    # Detect provider and get appropriate API key
    if model_path.startswith("groq/"):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            click.echo(f"Error: Model '{model}' requires GROQ_API_KEY")
            click.echo("Set it with: export GROQ_API_KEY='your-groq-key'")
            click.echo("Or use a Together AI model: --model qwen")
            raise click.Abort()
    elif model_path.startswith("together_ai/"):
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            click.echo(f"Error: Model '{model}' requires TOGETHER_API_KEY")
            click.echo("Set it with: export TOGETHER_API_KEY='your-together-key'")
            click.echo("Or use a Groq model: --model llama")
            raise click.Abort()
    elif model_path.startswith("openai/") or model_path in [
        "o4-mini",
    ]:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            click.echo(f"Error: Model '{model}' requires OPENAI_API_KEY")
            click.echo("Set it with: export OPENAI_API_KEY='your-openai-key'")
            raise click.Abort()
    elif model_path.startswith("gemini/"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            click.echo(f"Error: Model '{model}' requires GEMINI_API_KEY")
            click.echo("Set it with: export GEMINI_API_KEY='your-gemini-key'")
            raise click.Abort()
    else:
        # Try available keys for custom models
        api_key = (
            os.getenv("TOGETHER_API_KEY")
            or os.getenv("GROQ_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("GEMINI_API_KEY")
        )
        if not api_key:
            click.echo("Error: No API key found")
            click.echo("Set one of:")
            click.echo("  export TOGETHER_API_KEY='your-together-key'")
            click.echo("  export GROQ_API_KEY='your-groq-key'")
            click.echo("  export OPENAI_API_KEY='your-openai-key'")
            click.echo("  export GEMINI_API_KEY='your-gemini-key'")
            raise click.Abort()

    try:
        # Configure Gemini models with thinking enabled
        if model_path.startswith("gemini/"):
            # Enable thinking for Gemini models with moderate thinking budget
            thinking_budget = 1024  # Moderate reasoning for optimization tasks
            lm = dspy.LM(
                model_path,
                api_key=api_key,
                max_tokens=MAX_TOKENS,
                thinking_budget=thinking_budget,
            )
            click.echo(
                f"Configured {model} ({model_path}) with thinking_budget={thinking_budget}"
            )
        else:
            lm = dspy.LM(model_path, api_key=api_key, max_tokens=MAX_TOKENS)
            click.echo(f"Configured {model} ({model_path})")

        dspy.configure(lm=lm)
    except Exception as e:
        click.echo(f"Error configuring model {model}: {e}")
        raise click.Abort()


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
    default="simba",
    type=click.Choice(["bootstrap", "mipro", "simba"]),
    help="Optimization method",
)
@click.option("--num-demos", default=5, help="Number of demonstrations")
@click.option("--limit", default=50, help="Training examples limit")
@click.option("--threads", default=8, help="Number of threads for optimization")
@click.option(
    "--model",
    default=DEFAULT_MODEL,
    type=click.Choice(list(MODELS.keys()) + ["custom"]),
    help="Model to use",
)
@click.option("--debug", is_flag=True, help="Show debug information about optimization")
def optimize(train_data, output, system, method, num_demos, limit, threads, model, debug):
    """Optimize memory system using DSPy."""
    configure_lm(model)

    # Load data and create model
    train = load_locomo_dataset(train_data)[:limit]
    memory_model = get_memory_system(system)

    # Use LOCOMO metric
    metric = LOCOMOMetric()

    # Optimize
    if method == "bootstrap":
        optimizer = BootstrapFewShot(metric=metric, max_bootstrapped_demos=num_demos)
    elif method == "mipro":
        optimizer = MIPROv2(metric=metric, num_candidates=10, init_temperature=0.7)
    else:  # simba
        optimizer = SIMBA(
            metric=metric,
            max_demos=num_demos,
            num_threads=threads,
            max_steps=12,  # Increased from 6 for better optimization
            num_candidates=16,  # Increased from 8 for more exploration
            bsize=min(len(train), 16),  # Increased batch size
        )

    optimized = optimizer.compile(memory_model, trainset=train)

    if debug:
        click.echo("\n=== OPTIMIZATION DEBUG ===")
        click.echo(f"Base model: {memory_model}")
        click.echo(f"Optimized model: {optimized}")
        
        # Check for demonstrations in optimized model
        for attr_name in dir(optimized):
            attr = getattr(optimized, attr_name)
            if hasattr(attr, 'demos') and attr.demos:
                click.echo(f"Found {len(attr.demos)} demos in {attr_name}")
            elif hasattr(attr, 'signature') and hasattr(attr.signature, 'instructions'):
                click.echo(f"Instructions in {attr_name}: {attr.signature.instructions[:100]}...")

    # Save with metadata - include both system and model
    output_path = output.replace(".json", f"_{system}_{model}.json")
    
    # Enhanced save: use both program save and JSON save
    program_dir = output_path.replace(".json", "_program")
    try:
        # Primary save with full program serialization
        optimized.save(program_dir, save_program=True)
        click.echo(f"Saved optimized program to {program_dir}/")
    except Exception as e:
        click.echo(f"Warning: Program save failed: {e}")
    
    # Fallback JSON save
    optimized.save(output_path)
    click.echo(f"Saved optimized {system} model ({model}) to {output_path}")
    
    if debug:
        click.echo(f"=== SAVED FILES ===")
        if Path(program_dir).exists():
            click.echo(f"Program directory: {program_dir}")
            for f in Path(program_dir).iterdir():
                click.echo(f"  - {f.name}")
        click.echo(f"JSON file: {output_path} ({Path(output_path).stat().st_size} bytes)")


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
@click.option(
    "--model",
    default=DEFAULT_MODEL,
    type=click.Choice(list(MODELS.keys()) + ["custom"]),
    help="Model to use",
)
def evaluate(test_data, system, model_path, limit, threads, output, model):
    """Evaluate memory system using LOCOMO LLM judge."""
    configure_lm(model)

    # Load model with enhanced loader
    memory_model = load_optimized_model(system, model_path) if model_path else get_memory_system(system)

    # Load test data
    test = load_locomo_dataset(test_data)[:limit]

    # Evaluate with LOCOMO metric
    evaluator = Evaluate(
        devset=test, metric=LOCOMOMetric(), num_threads=threads, display_progress=True
    )

    score = evaluator(memory_model)

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
@click.option(
    "--model",
    default=DEFAULT_MODEL,
    type=click.Choice(list(MODELS.keys()) + ["custom"]),
    help="Model to use",
)
def compare(test_data, systems, limit, threads, output_dir, model):
    """Compare multiple memory systems with and without optimization."""
    configure_lm(model)

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

        # Optimized model - check for model-specific file first, then fallback
        opt_path = f"optimized_model_{system_name}_{model}.json"
        if not Path(opt_path).exists():
            opt_path = f"optimized_model_{system_name}.json"  # Fallback to old naming
        opt_score = None

        if Path(opt_path).exists():
            click.echo(f"\nOptimized model ({opt_path}):")
            opt_model = load_optimized_model(system_name, opt_path)
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
        # Convert from percentage (100.0) to decimal (1.0) for proper formatting
        base_decimal = scores["base_score"] / 100.0
        opt_decimal = (
            scores["optimized_score"] / 100.0 if scores["optimized_score"] else None
        )
        delta_decimal = scores["improvement"] / 100.0 if scores["improvement"] else None

        base = f"{base_decimal:.1%}"
        opt = f"{opt_decimal:.1%}" if opt_decimal is not None else "N/A"
        delta = f"{delta_decimal:+.1%}" if delta_decimal is not None else "N/A"
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
@click.option(
    "--model",
    default=DEFAULT_MODEL,
    type=click.Choice(list(MODELS.keys()) + ["custom"]),
    help="Model to use",
)
def benchmark(test_data, systems, configs, limit, threads, output, model):
    """Run comprehensive benchmark across systems and configurations."""
    configure_lm(model)

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

            memory_model = get_memory_system(system_name)

            if config == "optimized":
                opt_path = f"optimized_model_{system_name}_{model}.json"
                if not Path(opt_path).exists():
                    opt_path = (
                        f"optimized_model_{system_name}.json"  # Fallback to old naming
                    )
                if Path(opt_path).exists():
                    memory_model = load_optimized_model(system_name, opt_path)
                else:
                    click.echo(f"  Skipping - no optimized model found")
                    continue

            score = evaluator(memory_model)

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
@click.option(
    "--model",
    default=DEFAULT_MODEL,
    type=click.Choice(list(MODELS.keys()) + ["custom"]),
    help="Model to use",
)
def ask(question, conversation, system, model_path, model):
    """Interactive Q&A with chosen memory system."""
    configure_lm(model)

    # Load model with enhanced loader 
    memory_model = load_optimized_model(system, model_path) if model_path else get_memory_system(system)

    # Get conversation if not provided
    if not conversation:
        conversation = click.prompt("Enter conversation context")

    # Get answer
    pred = memory_model(conversation=conversation, question=question)
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
