"""
LOCOMO Evaluation - LLM-as-Judge DSPy evaluation module.
"""
import json
from typing import List, Dict, Any
from pathlib import Path

import dspy
from dspy.evaluate import Evaluate

from locomo.dataset import load_locomo_dataset
from locomo.llm_judge import create_locomo_judge


class LOCOMOMetric:
    """LOCOMO evaluation metric using LLM-as-Judge."""
    
    def __init__(self):
        self.judge = create_locomo_judge()
    
    def __call__(self, example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
        """Evaluate prediction using LLM-as-Judge."""
        try:
            evaluation = self.judge.evaluate_answer(
                question=example.question,
                ground_truth=example.answer,
                generated_answer=pred.answer
            )
            return 1.0 if evaluation["is_correct"] else 0.0
        except Exception as e:
            print(f"Error in LOCOMO evaluation: {e}")
            # Fallback to simple string matching
            pred_answer = pred.answer.lower().strip()
            ground_truth = example.answer.lower().strip()
            return 1.0 if pred_answer == ground_truth else 0.0


def evaluate_predictions(predictions: List[Dict[str, Any]], 
                        examples: List[dspy.Example]) -> Dict[str, Any]:
    """
    Evaluate predictions using LLM-as-Judge.
    
    Args:
        predictions: List of {"answer": str, ...} dictionaries
        examples: List of DSPy examples with ground truth
        
    Returns:
        Evaluation results dictionary
    """
    judge = create_locomo_judge()
    total_score = 0
    category_scores = {cat: [] for cat in range(1, 6)}
    detailed_results = []
    
    for pred, example in zip(predictions, examples):
        try:
            evaluation = judge.evaluate_answer(
                question=example.question,
                ground_truth=example.answer,
                generated_answer=pred["answer"]
            )
            
            score = 1.0 if evaluation["is_correct"] else 0.0
            total_score += score
            category_scores[example.category].append(score)
            
            detailed_results.append({
                "question": example.question,
                "ground_truth": example.answer,
                "prediction": pred["answer"],
                "category": example.category,
                "score": score,
                "judge_reasoning": evaluation["reasoning"],
                "judge_judgment": evaluation["judgment"]
            })
            
        except Exception as e:
            print(f"Error evaluating prediction: {e}")
            score = 0.0
            total_score += score
            category_scores[example.category].append(score)
            
            detailed_results.append({
                "question": example.question,
                "ground_truth": example.answer,
                "prediction": pred["answer"],
                "category": example.category,
                "score": score,
                "judge_reasoning": f"Error: {e}",
                "judge_judgment": "ERROR"
            })
    
    # Calculate overall and category-specific scores
    overall_score = total_score / len(predictions) if predictions else 0.0
    
    results = {
        "overall_llm_judge_score": overall_score,
        "total_examples": len(predictions),
        "detailed_results": detailed_results
    }
    
    # Add category-specific results
    for cat in range(1, 6):
        if category_scores[cat]:
            cat_score = sum(category_scores[cat]) / len(category_scores[cat])
            results[f"category_{cat}_score"] = cat_score
            results[f"category_{cat}_count"] = len(category_scores[cat])
        else:
            results[f"category_{cat}_score"] = 0.0
            results[f"category_{cat}_count"] = 0
    
    return results


def create_dspy_evaluator(metric_name: str = "locomo_llm_judge") -> Evaluate:
    """Create DSPy evaluator with LOCOMO metric."""
    return Evaluate(
        devset=[],  # Will be set when calling evaluate
        metric=LOCOMOMetric(),
        num_threads=1,
        display_progress=True,
        display_table=5
    )