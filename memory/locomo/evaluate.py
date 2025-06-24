"""
Paper-accurate DSPy evaluation script for LOCOMO QA.
Based on the official LOCOMO paper methodology.
"""

import argparse
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
import json
from datetime import datetime
import pickle

import dspy
from tqdm import tqdm
from locomo.dataset import load_locomo_dataset
from locomo.modules import create_module
from locomo.metrics import LocomoPaperMetrics


def evaluate_single_example(
    example: dspy.Example, module: dspy.Module, verbose: bool = False
) -> Tuple[dspy.Prediction, float]:
    """Evaluate a single example and return prediction and score."""
    try:
        # Pass category to the module for paper-accurate handling
        # Check if module accepts category parameter
        if hasattr(module, 'forward') and 'category' in module.forward.__code__.co_varnames:
            pred = module(
                conversation=example.conversation,
                question=example.question,
                category=example.category,
            )
        else:
            pred = module(
                conversation=example.conversation,
                question=example.question,
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
        "--limit",
        type=int,
        default=None,
        help="Number of examples to evaluate (default: all)",
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
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of parallel workers for evaluation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/locomo_dspy_runs",
        help="Directory to save evaluation outputs and traces",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Name for this experiment (default: timestamp)",
    )

    args = parser.parse_args()

    # Setup output directory and experiment tracking
    experiment_name = args.experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Saving outputs to: {output_dir}")
    
    # Setup language model with trace tracking
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found. Please set your OpenAI API key.")
        return

    lm = dspy.LM(args.model, api_key=api_key)
    # Enable trace tracking for DSPy
    dspy.configure(lm=lm, trace=[])
    
    # Save experiment configuration
    config = {
        "experiment_name": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
        "model": args.model,
        "module_type": args.module_type,
        "categories": args.categories,
        "limit": args.limit,
        "max_workers": args.max_workers,
    }
    
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

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
    scores = []
    all_results = []  # Store all results for logging
    category_examples = {i: [] for i in range(1, 6)}
    category_predictions = {i: [] for i in range(1, 6)}

    if args.max_workers == 1:
        # Sequential processing with progress bar
        for i, example in enumerate(tqdm(examples, desc="Evaluating", unit="examples")):
            pred, score = evaluate_single_example(example, module, args.verbose)
            predictions.append(pred)
            scores.append(score)
            
            # Store full result for logging
            result = {
                "example_id": i,
                "sample_id": example.sample_id,
                "qa_id": example.qa_id,
                "question": example.question,
                "ground_truth": example.answer,
                "prediction": pred.answer,
                "category": example.category,
                "score": score,
                "reasoning": getattr(pred, "reasoning", None),
                "evidence": example.evidence,
            }
            all_results.append(result)

            # Group by category for detailed analysis
            category_examples[example.category].append(example)
            category_predictions[example.category].append(pred)
    else:
        # Parallel processing with progress bar
        print(f"Using {args.max_workers} parallel workers")

        # Note: DSPy modules need to be recreated in each thread due to internal state
        def evaluate_with_module(example_with_idx):
            idx, example = example_with_idx
            # Create a new module instance for this thread
            thread_module = create_module(args.module_type)
            pred, score = evaluate_single_example(
                example, thread_module, False
            )  # Disable verbose in parallel mode
            return idx, example, pred, score

        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            # Submit all tasks with indices
            indexed_examples = list(enumerate(examples))
            future_to_data = {
                executor.submit(evaluate_with_module, item): item
                for item in indexed_examples
            }

            # Collect results with progress bar
            results_list = []
            for future in tqdm(
                as_completed(future_to_data),
                total=len(examples),
                desc="Evaluating",
                unit="examples",
            ):
                idx, example, pred, score = future.result()
                results_list.append((idx, example, pred, score))
                
        # Sort by index to maintain order
        results_list.sort(key=lambda x: x[0])
        
        # Process sorted results
        for idx, example, pred, score in results_list:
            predictions.append(pred)
            scores.append(score)
            
            # Store full result for logging
            result = {
                "example_id": idx,
                "sample_id": example.sample_id,
                "qa_id": example.qa_id,
                "question": example.question,
                "ground_truth": example.answer,
                "prediction": pred.answer,
                "category": example.category,
                "score": score,
                "reasoning": getattr(pred, "reasoning", None),
                "evidence": example.evidence,
            }
            all_results.append(result)

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
    
    # Save all outputs and traces
    print(f"\n💾 Saving evaluation outputs...")
    
    # Save detailed results as JSON
    results_data = {
        "experiment_name": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "results": results,
        "detailed_results": all_results,
        "scores": scores,
        "mean_score": sum(scores) / len(scores) if scores else 0,
    }
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(results_data, f, indent=2)
    
    # Save predictions separately for easy access
    predictions_data = []
    for i, (example, pred) in enumerate(zip(examples, predictions)):
        pred_dict = {
            "example_id": i,
            "question": example.question,
            "ground_truth": example.answer,
            "prediction": pred.answer,
            "category": example.category,
            "score": scores[i] if i < len(scores) else None,
        }
        if hasattr(pred, "reasoning"):
            pred_dict["reasoning"] = pred.reasoning
        predictions_data.append(pred_dict)
    
    with open(output_dir / "predictions.json", "w") as f:
        json.dump(predictions_data, f, indent=2)
    
    # Save DSPy traces if available (with error handling)
    try:
        if hasattr(dspy.settings, "trace") and dspy.settings.trace:
            # Convert traces to a serializable format
            serializable_traces = []
            for trace in dspy.settings.trace:
                try:
                    # Extract key information without complex objects
                    trace_info = {
                        "timestamp": str(trace) if hasattr(trace, '__str__') else "unknown",
                        "type": type(trace).__name__,
                    }
                    serializable_traces.append(trace_info)
                except:
                    continue
            
            with open(output_dir / "dspy_traces.json", "w") as f:
                json.dump(serializable_traces, f, indent=2)
    except Exception as e:
        print(f"⚠️  Warning: Could not save DSPy traces: {e}")
    
    # Save examples and predictions as DSPy dataset for future use
    try:
        # Convert to serializable format
        serializable_examples = []
        for ex in examples:
            try:
                if hasattr(ex, 'toDict'):
                    serializable_examples.append(ex.toDict())
                else:
                    # Manual serialization
                    serializable_examples.append({
                        "conversation": ex.conversation,
                        "question": ex.question,
                        "answer": ex.answer,
                        "category": getattr(ex, 'category', None),
                        "sample_id": getattr(ex, 'sample_id', None),
                        "qa_id": getattr(ex, 'qa_id', None),
                        "evidence": getattr(ex, 'evidence', []),
                    })
            except Exception as e:
                print(f"⚠️  Warning: Could not serialize example: {e}")
                continue
        
        serializable_predictions = []
        for pred in predictions:
            try:
                if hasattr(pred, 'toDict'):
                    serializable_predictions.append(pred.toDict())
                else:
                    # Manual serialization
                    pred_dict = {"answer": getattr(pred, 'answer', '')}
                    if hasattr(pred, 'reasoning'):
                        pred_dict["reasoning"] = pred.reasoning
                    serializable_predictions.append(pred_dict)
            except Exception as e:
                print(f"⚠️  Warning: Could not serialize prediction: {e}")
                continue
        
        dspy_dataset = {
            "examples": serializable_examples,
            "predictions": serializable_predictions,
            "module_type": args.module_type,
        }
        
        with open(output_dir / "dspy_dataset.json", "w") as f:
            json.dump(dspy_dataset, f, indent=2)
    except Exception as e:
        print(f"⚠️  Warning: Could not save DSPy dataset: {e}")
    
    # Create a summary file
    summary = {
        "total_examples": len(examples),
        "mean_score": sum(scores) / len(scores) if scores else 0,
        "category_breakdown": {
            f"category_{cat}": {
                "count": results.get(f"category_{cat}_count", 0),
                "score": results.get(f"category_{cat}_score", 0),
            }
            for cat in range(1, 6)
        },
        "output_files": [
            "config.json",
            "results.json", 
            "predictions.json",
            "dspy_traces.json",
            "dspy_dataset.json",
        ],
    }
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ All outputs saved to: {output_dir}")
    print(f"   - config.json: Experiment configuration")
    print(f"   - results.json: Detailed evaluation results")
    print(f"   - predictions.json: All predictions with scores")
    print(f"   - dspy_traces.json: DSPy execution traces")
    print(f"   - dspy_dataset.json: Examples and predictions in DSPy format")
    print(f"   - summary.json: Quick summary of results")


if __name__ == "__main__":
    main()
