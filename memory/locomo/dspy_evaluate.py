"""
Paper-accurate DSPy evaluation script for LOCOMO QA.
Based on the official LOCOMO paper methodology.
"""

import argparse
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

import dspy
from tqdm import tqdm
from locomo.dspy_dataset import load_locomo_dataset
from locomo.dspy_modules import create_module
from locomo.dspy_metrics import LocomoPaperMetrics


def evaluate_single_example(example: dspy.Example, module: dspy.Module, verbose: bool = False) -> Tuple[dspy.Prediction, float]:
    """Evaluate a single example and return prediction and score."""
    try:
        # Pass category to the module for paper-accurate handling
        pred = module(
            conversation=example.conversation,
            question=example.question,
            category=example.category,
        )
        
        # Calculate score for this example
        paper_metric = LocomoPaperMetrics.get_metric("locomo_paper")
        score = paper_metric(example, pred)
        
        if verbose:
            print(f"Question: {example.question}")
            print(f"Ground Truth: {example.answer}")
            print(f"Category: {example.category}")
            print(f"Prediction: {pred.answer}")
            if hasattr(pred, "reasoning") and pred.reasoning:
                print(f"Reasoning: {pred.reasoning[:200]}...")
            if hasattr(pred, "question_type"):
                print(f"Question Type: {pred.question_type}")
            print(f"Paper-accurate Score: {score:.3f}")
            print("-" * 50)
        
        return pred, score
        
    except Exception as e:
        if verbose:
            print(f"❌ Error processing example: {e}")
        return dspy.Prediction(answer=""), 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Paper-accurate DSPy evaluation for LOCOMO"
    )
    parser.add_argument(
        "--data-path", default="./data/locomo10.json", help="Path to LOCOMO dataset"
    )
    parser.add_argument(
        "--module-type",
        default="paper_accurate",
        choices=["paper_accurate", "memory_aware", "simple"],
    )
    parser.add_argument(
        "--limit", type=int, default=20, help="Number of examples to evaluate"
    )
    parser.add_argument(
        "--model", default="openai/gpt-4o-mini", help="Language model to use"
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        type=int,
        default=None,
        help="Specific categories to test (1-5)",
    )
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    parser.add_argument("--max-workers", type=int, default=4, help="Number of parallel workers for evaluation")

    args = parser.parse_args()

    # Setup language model
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found. Please set your OpenAI API key.")
        return

    lm = dspy.LM(args.model, api_key=api_key)
    dspy.configure(lm=lm)

    # Load dataset
    print("📚 Loading LOCOMO dataset...")
    dataset = load_locomo_dataset(args.data_path)

    # Filter by categories if specified
    if args.categories:
        examples = dataset.get_category_split(
            args.categories, limit_per_category=args.limit // len(args.categories)
        )
        print(f"🎯 Evaluating categories {args.categories}: {len(examples)} examples")
    else:
        examples = dataset.get_examples(limit=args.limit)
        print(f"📊 Evaluating {len(examples)} examples across all categories")

    # Create module
    print(f"🤖 Creating {args.module_type} module...")
    module = create_module(args.module_type)

    # Evaluate examples
    print(f"🔍 Running paper-accurate evaluation...")

    predictions = []
    category_examples = {i: [] for i in range(1, 6)}
    category_predictions = {i: [] for i in range(1, 6)}

    if args.max_workers == 1:
        # Sequential processing with progress bar
        for example in tqdm(examples, desc="Evaluating", unit="examples"):
            pred, score = evaluate_single_example(example, module, args.verbose)
            predictions.append(pred)
            
            # Group by category for detailed analysis
            category_examples[example.category].append(example)
            category_predictions[example.category].append(pred)
    else:
        # Parallel processing with progress bar
        print(f"Using {args.max_workers} parallel workers")
        
        # Note: DSPy modules need to be recreated in each thread due to internal state
        def evaluate_with_module(example):
            # Create a new module instance for this thread
            thread_module = create_module(args.module_type)
            return evaluate_single_example(example, thread_module, False)  # Disable verbose in parallel mode
        
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            # Submit all tasks
            future_to_example = {
                executor.submit(evaluate_with_module, example): example 
                for example in examples
            }
            
            # Collect results with progress bar
            for future in tqdm(as_completed(future_to_example), 
                             total=len(examples), 
                             desc="Evaluating", 
                             unit="examples"):
                example = future_to_example[future]
                pred, score = future.result()
                predictions.append(pred)
                
                # Group by category for detailed analysis
                category_examples[example.category].append(example)
                category_predictions[example.category].append(pred)

    # Overall evaluation using paper-accurate metrics
    print(f"\n📊 Paper-Accurate LOCOMO Evaluation Results:")
    print("=" * 50)

    results = LocomoPaperMetrics.evaluate_with_paper_metrics(examples, predictions)

    # Overall results
    print(f"Overall Score: {results['overall_score']:.3f}")
    print(f"Total Examples: {results['total_examples']}")
    print()

    # Category-specific results
    print("Category-Specific Results:")
    category_names = {
        1: "Multi-hop Reasoning",
        2: "Temporal Reasoning",
        3: "Single-hop Factual",
        4: "Unanswerable Questions",
        5: "Ambiguous Questions",
    }

    for cat in range(1, 6):
        if f"category_{cat}_score" in results:
            score = results[f"category_{cat}_score"]
            count = results[f"category_{cat}_count"]
            print(
                f"  Category {cat} ({category_names[cat]}): {score:.3f} ({count} examples)"
            )
    print()

    # Memory distance results
    print("Memory Distance Results:")
    for dist_type in ["immediate", "recent", "medium", "distant"]:
        score_key = f"memory_{dist_type}_score"
        count_key = f"memory_{dist_type}_count"
        if score_key in results:
            score = results[score_key]
            count = results[count_key]
            print(f"  {dist_type.capitalize()}: {score:.3f} ({count} examples)")

    # Show some example predictions by category
    if args.verbose:
        print(f"\n🔍 Example Predictions by Category:")
        print("=" * 50)

        for cat in range(1, 6):
            if category_examples[cat]:
                print(f"\nCategory {cat} ({category_names[cat]}):")
                example = category_examples[cat][0]
                pred = category_predictions[cat][0]

                print(f"  Q: {example.question}")
                print(f"  Ground Truth: {example.answer}")
                print(f"  Prediction: {pred.answer}")
                if hasattr(pred, "reasoning") and pred.reasoning:
                    print(f"  Reasoning: {pred.reasoning[:150]}...")


if __name__ == "__main__":
    main()
