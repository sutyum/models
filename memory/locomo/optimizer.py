"""
DSPy optimization script for LOCOMO conversational QA using MIPRO.
"""
import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Optional

import dspy
from dspy.evaluate import Evaluate
from dspy.teleprompt import MIPROv2 as MIPRO

from locomo.dataset import load_locomo_dataset
from locomo.modules import create_module
from locomo.metrics import LocomoPaperMetrics


def setup_language_model(model_name: str = "openai/gpt-4o-mini", api_key: Optional[str] = None):
    """Setup the language model for DSPy."""
    if api_key is None:
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
    
    lm = dspy.LM(model_name, api_key=api_key)
    dspy.configure(lm=lm, track_usage=True)
    return lm


def run_baseline_evaluation(dataset, module, metric_name: str = "f1", 
                          test_set: Optional[List] = None, limit: int = 50):
    """Run baseline evaluation before optimization."""
    print("🔍 Running baseline evaluation...")
    
    if test_set is None:
        test_set = dataset.get_examples(limit=limit)
    
    metric_fn = LocomoPaperMetrics.get_metric(metric_name)
    
    # Use DSPy's Evaluate utility
    evaluator = Evaluate(
        devset=test_set,
        num_threads=1,
        display_progress=True,
        display_table=5
    )
    
    baseline_score = evaluator(module, metric=metric_fn)
    
    print(f"📊 Baseline {metric_name} score: {baseline_score:.3f}")
    return baseline_score


def optimize_with_mipro(module, train_set: List, val_set: List, metric_name: str = "f1", 
                       num_trials: int = 50, max_bootstrapped_demos: int = 4,
                       max_labeled_demos: int = 16, init_temperature: float = 1.0):
    """Optimize the module using MIPRO."""
    print("🚀 Starting MIPRO optimization...")
    
    metric_fn = LocomoPaperMetrics.get_metric(metric_name)
    
    # Initialize MIPRO optimizer with correct parameters
    mipro_optimizer = MIPRO(
        metric=metric_fn,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
        init_temperature=init_temperature,
        auto="light",  # Use auto mode instead of num_trials
        verbose=True,
        track_stats=True
    )
    
    # Run optimization
    print(f"📈 Optimizing in 'light' mode...")
    optimized_module = mipro_optimizer.compile(
        student=module,
        trainset=train_set,
        valset=val_set
    )
    
    print("✅ MIPRO optimization completed!")
    return optimized_module


def detailed_evaluation(examples: List, module, metric_names: List[str] = None):
    """Run detailed evaluation with multiple metrics."""
    if metric_names is None:
        metric_names = ["locomo_paper"]
    
    print("📋 Running detailed evaluation...")
    
    predictions = []
    for example in examples:
        try:
            # Check if module accepts category parameter
            if hasattr(module, 'forward') and 'category' in module.forward.__code__.co_varnames:
                pred = module(conversation=example.conversation, question=example.question, category=example.category)
            else:
                pred = module(conversation=example.conversation, question=example.question)
            predictions.append(pred)
        except Exception as e:
            print(f"Error processing example: {e}")
            # Create dummy prediction
            predictions.append(dspy.Prediction(answer=""))
    
    # Use paper-accurate evaluation
    results = LocomoPaperMetrics.evaluate_with_paper_metrics(examples, predictions)
    
    print(f"Overall Score: {results['overall_score']:.3f}")
    print(f"Total Examples: {results['total_examples']}")
    
    # Print category breakdowns if available
    for cat in range(1, 6):
        score_key = f"category_{cat}_score"
        count_key = f"category_{cat}_count"
        if score_key in results and count_key in results:
            print(f"Category {cat}: {results[score_key]:.3f} ({results[count_key]} examples)")
    
    return results, predictions


def save_results(results: Dict, predictions: List, output_path: str):
    """Save evaluation results and predictions."""
    output_data = {
        "evaluation_results": results,
        "predictions": [
            {
                "answer": pred.answer if hasattr(pred, 'answer') else "",
                "reasoning": pred.reasoning if hasattr(pred, 'reasoning') else ""
            }
            for pred in predictions
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"💾 Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="DSPy optimization for LOCOMO QA")
    parser.add_argument(
        "--data-path", 
        default="./data/locomo10.json",
        help="Path to LOCOMO dataset"
    )
    parser.add_argument(
        "--module-type",
        default="paper_accurate",
        choices=["paper_accurate", "memory_aware", "simple"],
        help="Type of DSPy module to use"
    )
    parser.add_argument(
        "--metric",
        default="locomo_paper",
        choices=["locomo_paper", "multi_hop", "temporal", "unanswerable", "ambiguous"],
        help="Evaluation metric to optimize"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of examples to use (default: all)"
    )
    parser.add_argument(
        "--auto-mode",
        default="light",
        choices=["light", "medium", "heavy"],
        help="MIPRO auto optimization mode"
    )
    parser.add_argument(
        "--max-demos",
        type=int,
        default=8,
        help="Maximum demonstrations for optimization"
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-4o-mini",
        help="Language model to use"
    )
    parser.add_argument(
        "--output-dir",
        default="./outputs/dspy_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--skip-optimization",
        action="store_true",
        help="Skip optimization and only run baseline evaluation"
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        type=int,
        default=None,
        help="Specific categories to evaluate (1-5)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup language model
    print("🔧 Setting up language model...")
    lm = setup_language_model(args.model)
    
    # Load dataset
    print("📚 Loading LOCOMO dataset...")
    dataset = load_locomo_dataset(args.data_path)
    
    # Print dataset stats
    stats = dataset.get_stats()
    print("Dataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Get examples (filter by categories if specified)
    if args.categories:
        examples = dataset.get_category_split(
            args.categories, 
            limit_per_category=args.limit // len(args.categories) if args.limit else None
        )
        print(f"🎯 Using categories {args.categories}: {len(examples)} examples")
    else:
        examples = dataset.get_examples(limit=args.limit)
        print(f"📊 Using {len(examples)} examples")
    
    # Split data
    train_ratio, val_ratio, test_ratio = 0.6, 0.2, 0.2
    n_total = len(examples)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_set = examples[:n_train]
    val_set = examples[n_train:n_train + n_val]
    test_set = examples[n_train + n_val:]
    
    print(f"📊 Data split: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")
    
    # Create module
    print(f"🤖 Creating {args.module_type} module...")
    module = create_module(args.module_type)
    
    # Baseline evaluation
    baseline_score = run_baseline_evaluation(dataset, module, args.metric, test_set)
    
    if not args.skip_optimization:
        # Optimize with MIPRO
        optimized_module = optimize_with_mipro(
            module=module,
            train_set=train_set,
            val_set=val_set,
            metric_name=args.metric,
            max_labeled_demos=args.max_demos
        )
        
        # Evaluate optimized module
        print("🎯 Evaluating optimized module...")
        optimized_score = run_baseline_evaluation(dataset, optimized_module, args.metric, test_set)
        
        improvement = optimized_score - baseline_score
        print(f"📈 Improvement: {improvement:+.3f} ({improvement/baseline_score*100:+.1f}%)")
        
        # Detailed evaluation
        results, predictions = detailed_evaluation(test_set, optimized_module)
        
        # Save results
        save_results(
            results, 
            predictions, 
            output_dir / f"optimized_results_{args.module_type}_{args.metric}.json"
        )
        
        # Save the optimized module
        optimized_module.save(output_dir / f"optimized_module_{args.module_type}.json")
        
    else:
        # Just detailed baseline evaluation
        results, predictions = detailed_evaluation(test_set, module)
        save_results(
            results, 
            predictions, 
            output_dir / f"baseline_results_{args.module_type}_{args.metric}.json"
        )
    
    print("🎉 Evaluation completed!")


if __name__ == "__main__":
    main()