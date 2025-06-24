"""
DSPy metrics for LOCOMO conversational QA evaluation.
"""
import dspy
from typing import Optional, Dict, Any
from locomo.evaluation import f1_score, exact_match_score, normalize_answer


def f1_metric(example: dspy.Example, pred: dspy.Prediction, trace: Optional[Any] = None) -> float:
    """
    F1 score metric for LOCOMO QA.
    
    Args:
        example: Ground truth example
        pred: Model prediction
        trace: DSPy trace (for optimization)
    
    Returns:
        F1 score between 0 and 1
    """
    if not hasattr(pred, 'answer') or not hasattr(example, 'answer'):
        return 0.0
    
    # Handle different question categories
    if hasattr(example, 'category'):
        category = example.category
        
        # Category 5 (adversarial) - check for "not mentioned" responses
        if category == 5:
            pred_answer = pred.answer.lower()
            if "no information available" in pred_answer or "not mentioned" in pred_answer:
                # This is the correct response for adversarial questions
                return 1.0 if trace is None else True
            else:
                return 0.0 if trace is None else False
        
        # Multi-hop questions (category 1) - use specialized F1
        elif category == 1:
            return f1(pred.answer, example.answer)
        
        # Other categories - use standard F1
        else:
            return f1_score(pred.answer, example.answer)
    
    # Default F1 score
    return f1_score(pred.answer, example.answer)


def exact_match_metric(example: dspy.Example, pred: dspy.Prediction, trace: Optional[Any] = None) -> float:
    """
    Exact match metric for LOCOMO QA.
    """
    if not hasattr(pred, 'answer') or not hasattr(example, 'answer'):
        return 0.0
    
    match = exact_match_score(pred.answer, example.answer)
    
    if trace is None:
        return float(match)
    else:
        return match


def category_aware_metric(example: dspy.Example, pred: dspy.Prediction, trace: Optional[Any] = None) -> float:
    """
    Category-aware metric that uses different scoring for different question types.
    """
    if not hasattr(pred, 'answer') or not hasattr(example, 'answer'):
        return 0.0
    
    category = getattr(example, 'category', 3)  # Default to category 3
    
    if category == 5:  # Adversarial
        return adversarial_metric(example, pred, trace)
    elif category == 2:  # Temporal
        return temporal_metric(example, pred, trace)
    else:  # General QA
        return f1_metric(example, pred, trace)


def adversarial_metric(example: dspy.Example, pred: dspy.Prediction, trace: Optional[Any] = None) -> float:
    """
    Metric for adversarial questions (category 5).
    These questions should result in "not mentioned" or similar responses.
    """
    pred_answer = pred.answer.lower().strip()
    
    # Check if the model correctly identifies that information is not available
    not_mentioned_phrases = [
        "no information available",
        "not mentioned",
        "not available",
        "cannot be determined",
        "not stated",
        "not provided"
    ]
    
    is_correct = any(phrase in pred_answer for phrase in not_mentioned_phrases)
    
    if trace is None:
        return 1.0 if is_correct else 0.0
    else:
        return is_correct


def temporal_metric(example: dspy.Example, pred: dspy.Prediction, trace: Optional[Any] = None) -> float:
    """
    Metric for temporal questions (category 2).
    """
    # For temporal questions, we can be more lenient with date formats
    pred_answer = normalize_answer(pred.answer)
    true_answer = normalize_answer(example.answer)
    
    # Check for partial date matches (year, month, etc.)
    score = f1_score(pred.answer, example.answer)
    
    # Bonus for exact temporal match
    if pred_answer == true_answer:
        score = 1.0
    
    if trace is None:
        return score
    else:
        return score > 0.5


def comprehensive_metric(example: dspy.Example, pred: dspy.Prediction, trace: Optional[Any] = None) -> float:
    """
    Comprehensive metric that combines multiple aspects of answer quality.
    """
    if not hasattr(pred, 'answer') or not hasattr(example, 'answer'):
        return 0.0
    
    # Base F1 score
    f1 = f1_metric(example, pred, trace=None)
    
    # Answer length penalty (very short or very long answers are penalized)
    answer_len = len(pred.answer.split())
    if answer_len < 1:
        length_penalty = 0.0
    elif answer_len > 20:
        length_penalty = 0.8
    else:
        length_penalty = 1.0
    
    # Category-specific bonus
    category_score = category_aware_metric(example, pred, trace=None)
    
    # Combined score
    final_score = (f1 * 0.6 + category_score * 0.4) * length_penalty
    
    if trace is None:
        return final_score
    else:
        return final_score > 0.6


def f1(prediction: str, ground_truth: str) -> float:
    """
    Multi-answer F1 score for category 1 questions.
    Handles comma-separated answers.
    """
    predictions = [p.strip() for p in prediction.split(",")]
    ground_truths = [g.strip() for g in ground_truth.split(",")]
    
    scores = []
    for gt in ground_truths:
        gt_scores = [f1_score(pred, gt) for pred in predictions]
        scores.append(max(gt_scores) if gt_scores else 0.0)
    
    return sum(scores) / len(scores) if scores else 0.0


class LocomoMetrics:
    """Collection of metrics for LOCOMO evaluation."""
    
    @staticmethod
    def get_metric(metric_name: str):
        """Get a metric by name."""
        metrics_map = {
            "f1": f1_metric,
            "exact_match": exact_match_metric,
            "category_aware": category_aware_metric,
            "comprehensive": comprehensive_metric,
            "adversarial": adversarial_metric,
            "temporal": temporal_metric
        }
        
        if metric_name not in metrics_map:
            raise ValueError(f"Unknown metric: {metric_name}. Available: {list(metrics_map.keys())}")
        
        return metrics_map[metric_name]
    
    @staticmethod
    def evaluate_predictions(examples: list, predictions: list, metric_name: str = "f1") -> Dict[str, float]:
        """
        Evaluate a list of predictions against examples.
        
        Args:
            examples: List of ground truth examples
            predictions: List of model predictions
            metric_name: Name of metric to use
            
        Returns:
            Dictionary with evaluation results
        """
        metric_fn = LocomoMetrics.get_metric(metric_name)
        
        scores = []
        category_scores = {1: [], 2: [], 3: [], 4: [], 5: []}
        
        for example, pred in zip(examples, predictions):
            score = metric_fn(example, pred)
            scores.append(score)
            
            if hasattr(example, 'category'):
                category_scores[example.category].append(score)
        
        # Calculate overall and per-category metrics
        results = {
            "overall_score": sum(scores) / len(scores) if scores else 0.0,
            "total_examples": len(scores)
        }
        
        for cat, cat_scores in category_scores.items():
            if cat_scores:
                results[f"category_{cat}_score"] = sum(cat_scores) / len(cat_scores)
                results[f"category_{cat}_count"] = len(cat_scores)
        
        return results


if __name__ == "__main__":
    # Example usage
    example = dspy.Example(
        question="What did Alice eat?",
        answer="pasta",
        category=3
    )
    
    pred = dspy.Prediction(answer="spaghetti")
    
    print(f"F1 Score: {f1_metric(example, pred)}")
    print(f"Exact Match: {exact_match_metric(example, pred)}")
    print(f"Category-aware: {category_aware_metric(example, pred)}")
    
    # Test adversarial example
    adv_example = dspy.Example(
        question="What did Charlie wear?",
        answer="not mentioned",
        category=5
    )
    
    adv_pred = dspy.Prediction(answer="No information available in the conversation")
    print(f"Adversarial Score: {adversarial_metric(adv_example, adv_pred)}")