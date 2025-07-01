#!/usr/bin/env python3
"""
Evaluate optimized memory QA prompt on LOCOMO benchmark.
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Any
import argparse
from datetime import datetime

import dspy
from dspy.evaluate import Evaluate

from locomo.dataset import load_locomo_dataset
from locomo.evaluate import evaluate_predictions, LOCOMOMetric


def load_optimized_prompt(filepath: str) -> dspy.Module:
    """Load the optimized prompt from JSON file."""
    with open(filepath, "r") as f:
        optimized_data = json.load(f)

    # Extract the compiled program from the optimizer results
    if "memory.evolver.predict" in optimized_data:
        compiled_module = optimized_data["memory.evolver.predict"]
        print(
            f"Found optimized module with {len(compiled_module.get('demos', []))} demos"
        )
        return compiled_module
    else:
        raise ValueError("No optimized module found in the file")


class OptimizedMemoryQA(dspy.Module):
    """Memory QA module using optimized prompt."""

    def __init__(self, optimized_data: Dict[str, Any]):
        super().__init__()
        self.optimized_data = optimized_data
        self.predict = dspy.ChainOfThought(
            "current_state, new_information -> updated_state"
        )

        # Set up the optimized demos if available
        if "demos" in optimized_data:
            # Convert demos to proper format
            demos = []
            for demo in optimized_data["demos"]:
                example = dspy.Example(
                    current_state=demo.get("current_state", ""),
                    new_information=demo.get("new_information", ""),
                    updated_state=demo.get("updated_state", ""),
                ).with_inputs("current_state", "new_information")
                demos.append(example)
            self.predict.demos = demos

    def forward(self, current_state: str, new_information: str) -> dspy.Prediction:
        """Generate updated state given current state and new information."""
        # For LOCOMO, we need to generate an answer to the question
        # The new_information contains the question
        pred = self.predict(
            current_state=current_state, new_information=new_information
        )

        # Extract the answer from the updated state
        # For LOCOMO, the answer is typically a short response
        return dspy.Prediction(
            answer=pred.updated_state,
            updated_state=pred.updated_state,
            rationale=getattr(pred, "rationale", ""),
        )


def evaluate_on_locomo(
    optimized_prompt_path: str,
    test_data_path: str = "data/locomo_test.json",
    output_path: str = None,
) -> Dict[str, Any]:
    """
    Evaluate optimized prompt on LOCOMO test set.

    Args:
        optimized_prompt_path: Path to optimized prompt JSON file
        test_data_path: Path to LOCOMO test dataset
        output_path: Path to save evaluation results

    Returns:
        Evaluation results dictionary
    """
    print(f"Loading optimized prompt from: {optimized_prompt_path}")

    # Load optimized prompt
    try:
        optimized_data = load_optimized_prompt(optimized_prompt_path)
        model = OptimizedMemoryQA(optimized_data)
    except Exception as e:
        print(f"Error loading optimized prompt: {e}")
        return {"error": str(e)}

    # Load test dataset
    print(f"Loading LOCOMO test data from: {test_data_path}")
    try:
        test_examples = load_locomo_dataset(test_data_path)
        print(f"Loaded {len(test_examples)} test examples")
    except Exception as e:
        print(f"Error loading test data: {e}")
        return {"error": str(e)}

    # Set up DSPy evaluator
    evaluator = Evaluate(
        devset=test_examples,
        metric=LOCOMOMetric(),
        num_threads=1,
        display_progress=True,
        display_table=10,
    )

    print("Starting evaluation...")
    start_time = datetime.now()

    try:
        # Run evaluation
        results = evaluator(model)

        evaluation_time = datetime.now() - start_time

        # Prepare detailed results
        detailed_results = {
            "timestamp": datetime.now().isoformat(),
            "evaluation_time_seconds": evaluation_time.total_seconds(),
            "model_path": optimized_prompt_path,
            "test_data_path": test_data_path,
            "num_examples": len(test_examples),
            "overall_score": results,
            "per_example_results": [],
        }

        # Generate predictions for detailed analysis
        print("Generating detailed predictions...")
        predictions = []
        for example in test_examples:
            try:
                pred = model(
                    current_state=example.current_state,
                    new_information=example.new_information,
                )
                predictions.append(
                    {
                        "answer": pred.updated_state,
                        "reasoning": getattr(pred, "rationale", ""),
                        "example_id": getattr(example, "id", "unknown"),
                    }
                )
            except Exception as e:
                print(f"Error generating prediction: {e}")
                predictions.append(
                    {
                        "answer": "",
                        "reasoning": f"Error: {e}",
                        "example_id": getattr(example, "id", "unknown"),
                    }
                )

        # Get detailed evaluation results
        detailed_eval = evaluate_predictions(predictions, test_examples)
        detailed_results.update(detailed_eval)

        print(f"\nEvaluation completed in {evaluation_time}")
        print(f"Overall Score: {results:.3f}")

        # Print category-specific results if available
        for cat in range(1, 6):
            cat_score_key = f"category_{cat}_score"
            cat_count_key = f"category_{cat}_count"
            if (
                cat_score_key in detailed_results
                and detailed_results[cat_count_key] > 0
            ):
                score = detailed_results[cat_score_key]
                count = detailed_results[cat_count_key]
                print(f"Category {cat}: {score:.3f} ({count} examples)")

        # Save results if output path provided
        if output_path:
            print(f"Saving results to: {output_path}")
            with open(output_path, "w") as f:
                json.dump(detailed_results, f, indent=2)

        return detailed_results

    except Exception as e:
        print(f"Error during evaluation: {e}")
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate optimized memory QA prompt on LOCOMO"
    )
    parser.add_argument(
        "--optimized-prompt",
        default="optimized_memory_qa.json",
        help="Path to optimized prompt JSON file",
    )
    parser.add_argument(
        "--test-data",
        default="data/locomo_test.json",
        help="Path to LOCOMO test dataset",
    )
    parser.add_argument(
        "--output", default=None, help="Path to save evaluation results"
    )
    parser.add_argument(
        "--model", default="gpt-3.5-turbo", help="Model to use for evaluation"
    )

    args = parser.parse_args()

    # Configure DSPy
    import os

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        return 1

    lm = dspy.LM(model=args.model, max_tokens=1000)
    dspy.configure(lm=lm)

    # Generate output filename if not provided
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"evaluation_results_{timestamp}.json"

    # Run evaluation
    results = evaluate_on_locomo(
        optimized_prompt_path=args.optimized_prompt,
        test_data_path=args.test_data,
        output_path=args.output,
    )

    if "error" in results:
        print(f"Evaluation failed: {results['error']}")
        return 1

    print("\nEvaluation Summary:")
    print(f"- Overall Score: {results.get('overall_llm_judge_score', 'N/A'):.3f}")
    print(f"- Total Examples: {results.get('total_examples', 'N/A')}")
    print(f"- Results saved to: {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
