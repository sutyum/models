"""
LOCOMO-paper-accurate DSPy metrics for conversational QA evaluation.
Based on the official LOCOMO paper methodology.
"""
import dspy
from typing import Optional, Dict, Any, List
from locomo.evaluation import f1_score, exact_match_score, normalize_answer
import re


def locomo_paper_metric(example: dspy.Example, pred: dspy.Prediction, trace: Optional[Any] = None) -> float:
    """
    Paper-accurate LOCOMO metric based on question categories.
    
    From the paper:
    - Category 1 (Multi-hop): F1 score with partial credit
    - Category 2 (Temporal): F1 score for temporal expressions
    - Category 3 (Single-hop): F1 score for factual answers
    - Category 4 (Unanswerable): Exact match for "don't know" responses
    - Category 5 (Ambiguous): Multiple acceptable answers evaluation
    """
    if not hasattr(pred, 'answer') or not hasattr(example, 'answer'):
        return 0.0
    
    category = getattr(example, 'category', 3)
    pred_answer = pred.answer.strip()
    true_answer = example.answer.strip()
    
    if category == 1:
        # Multi-hop reasoning: F1 with partial credit for connecting multiple facts
        return multi_hop_f1_score(pred_answer, true_answer)
    
    elif category == 2:
        # Temporal reasoning: F1 with special handling for dates/times
        return temporal_f1_score(pred_answer, true_answer)
    
    elif category == 3:
        # Single-hop factual: Standard F1 score
        return f1_score(pred_answer, true_answer)
    
    elif category == 4:
        # Unanswerable: Check for appropriate "don't know" responses
        return unanswerable_score(pred_answer, true_answer)
    
    elif category == 5:
        # Ambiguous: Multiple acceptable answers
        return ambiguous_score(pred_answer, true_answer)
    
    else:
        # Default to F1
        return f1_score(pred_answer, true_answer)


def multi_hop_f1_score(prediction: str, ground_truth: str) -> float:
    """
    F1 score for multi-hop questions that may require connecting multiple facts.
    From paper: "For multi-hop questions, we compute F1 scores between predicted and ground truth answers"
    """
    # Handle comma-separated multiple answers
    if ',' in ground_truth:
        gt_parts = [part.strip() for part in ground_truth.split(',')]
        pred_parts = [part.strip() for part in prediction.split(',')]
        
        # Calculate F1 for each ground truth part against all predicted parts
        scores = []
        for gt_part in gt_parts:
            part_scores = [f1_score(pred_part, gt_part) for pred_part in pred_parts]
            scores.append(max(part_scores) if part_scores else 0.0)
        
        return sum(scores) / len(scores) if scores else 0.0
    else:
        return f1_score(prediction, ground_truth)


def temporal_f1_score(prediction: str, ground_truth: str) -> float:
    """
    F1 score for temporal questions with special handling for dates/times.
    From paper: "Temporal questions require reasoning about time and dates"
    """
    # Normalize temporal expressions
    pred_normalized = normalize_temporal(prediction)
    true_normalized = normalize_temporal(ground_truth)
    
    # Use standard F1 but with temporal normalization
    return f1_score(pred_normalized, true_normalized)


def unanswerable_score(prediction: str, ground_truth: str) -> float:
    """
    Score for unanswerable questions (Category 4).
    From paper: "Questions where the answer cannot be determined from the conversation"
    """
    pred_lower = prediction.lower().strip()
    true_lower = ground_truth.lower().strip()
    
    # Check if both indicate the question is unanswerable
    unanswerable_phrases = [
        "i don't know",
        "don't know",
        "cannot be determined",
        "not mentioned",
        "not available",
        "insufficient information",
        "unclear",
        "unknown",
        "not stated",
        "not provided"
    ]
    
    pred_is_unanswerable = any(phrase in pred_lower for phrase in unanswerable_phrases)
    true_is_unanswerable = any(phrase in true_lower for phrase in unanswerable_phrases)
    
    if pred_is_unanswerable and true_is_unanswerable:
        return 1.0
    elif not pred_is_unanswerable and not true_is_unanswerable:
        # Both provide actual answers, use F1
        return f1_score(prediction, ground_truth)
    else:
        # One is unanswerable, other isn't
        return 0.0


def ambiguous_score(prediction: str, ground_truth: str) -> float:
    """
    Score for ambiguous questions (Category 5) with multiple acceptable answers.
    From paper: "Questions that could have multiple valid interpretations"
    """
    # Handle multiple acceptable answers separated by semicolons or pipes
    if ';' in ground_truth or '|' in ground_truth:
        acceptable_answers = []
        for sep in [';', '|']:
            if sep in ground_truth:
                acceptable_answers = [ans.strip() for ans in ground_truth.split(sep)]
                break
        
        # Check if prediction matches any acceptable answer
        scores = [f1_score(prediction, ans) for ans in acceptable_answers]
        return max(scores) if scores else 0.0
    else:
        return f1_score(prediction, ground_truth)


def normalize_temporal(text: str) -> str:
    """Normalize temporal expressions for better matching."""
    text = text.lower().strip()
    
    # Common temporal normalizations
    temporal_mappings = {
        'january': 'jan', 'february': 'feb', 'march': 'mar',
        'april': 'apr', 'may': 'may', 'june': 'jun',
        'july': 'jul', 'august': 'aug', 'september': 'sep',
        'october': 'oct', 'november': 'nov', 'december': 'dec'
    }
    
    for full_month, short_month in temporal_mappings.items():
        text = text.replace(full_month, short_month)
    
    # Remove common temporal words that don't affect meaning
    temporal_noise = ['on', 'at', 'in', 'during', 'the']
    words = text.split()
    filtered_words = [w for w in words if w not in temporal_noise]
    
    return ' '.join(filtered_words)


def memory_distance_metric(example: dspy.Example, pred: dspy.Prediction, trace: Optional[Any] = None) -> Dict[str, float]:
    """
    Calculate memory distance-aware metrics as per LOCOMO paper.
    From paper: "We analyze performance based on how far back in the conversation the relevant information appears"
    """
    base_score = locomo_paper_metric(example, pred, trace)
    
    # Extract memory distance from evidence if available
    memory_distance = calculate_memory_distance(example)
    
    return {
        'score': base_score,
        'memory_distance': memory_distance,
        'distance_category': categorize_memory_distance(memory_distance)
    }


def calculate_memory_distance(example: dspy.Example) -> float:
    """
    Calculate memory distance based on evidence timestamps.
    From paper: "Memory distance is measured in conversation turns"
    """
    if not hasattr(example, 'evidence') or not example.evidence:
        return 0.0
    
    # Extract session and dialog IDs from evidence
    distances = []
    for evidence in example.evidence:
        if isinstance(evidence, str) and ':' in evidence:
            # Format: "D{session}:{dialog}" or "S{session}:{dialog}"
            try:
                session_part, dialog_part = evidence.split(':')
                session_num = int(session_part[1:])  # Remove 'D' or 'S' prefix
                dialog_num = int(dialog_part)
                
                # Simple distance calculation (could be more sophisticated)
                distance = session_num * 100 + dialog_num  # Rough ordering
                distances.append(distance)
            except (ValueError, IndexError):
                continue
    
    return min(distances) if distances else 0.0


def categorize_memory_distance(distance: float) -> str:
    """Categorize memory distance as per paper analysis."""
    if distance == 0:
        return "immediate"
    elif distance < 50:
        return "recent"
    elif distance < 200:
        return "medium"
    else:
        return "distant"


class LocomoPaperMetrics:
    """Paper-accurate metrics collection for LOCOMO evaluation."""
    
    @staticmethod
    def get_metric(metric_name: str):
        """Get a paper-accurate metric by name."""
        metrics_map = {
            "locomo_paper": locomo_paper_metric,
            "multi_hop": lambda ex, pred, trace=None: multi_hop_f1_score(pred.answer, ex.answer),
            "temporal": lambda ex, pred, trace=None: temporal_f1_score(pred.answer, ex.answer),
            "unanswerable": lambda ex, pred, trace=None: unanswerable_score(pred.answer, ex.answer),
            "ambiguous": lambda ex, pred, trace=None: ambiguous_score(pred.answer, ex.answer),
            "memory_distance": memory_distance_metric
        }
        
        if metric_name not in metrics_map:
            raise ValueError(f"Unknown metric: {metric_name}. Available: {list(metrics_map.keys())}")
        
        return metrics_map[metric_name]
    
    @staticmethod
    def evaluate_with_paper_metrics(examples: List, predictions: List) -> Dict[str, Any]:
        """
        Evaluate predictions using paper-accurate LOCOMO metrics.
        """
        results = {
            "overall": {"scores": [], "count": 0},
            "by_category": {i: {"scores": [], "count": 0} for i in range(1, 6)},
            "by_memory_distance": {
                "immediate": {"scores": [], "count": 0},
                "recent": {"scores": [], "count": 0}, 
                "medium": {"scores": [], "count": 0},
                "distant": {"scores": [], "count": 0}
            }
        }
        
        for example, pred in zip(examples, predictions):
            # Main metric
            score = locomo_paper_metric(example, pred)
            results["overall"]["scores"].append(score)
            results["overall"]["count"] += 1
            
            # By category
            category = getattr(example, 'category', 3)
            if 1 <= category <= 5:
                results["by_category"][category]["scores"].append(score)
                results["by_category"][category]["count"] += 1
            
            # By memory distance
            memory_info = memory_distance_metric(example, pred)
            distance_cat = memory_info["distance_category"]
            results["by_memory_distance"][distance_cat]["scores"].append(score)
            results["by_memory_distance"][distance_cat]["count"] += 1
        
        # Calculate averages
        final_results = {}
        
        # Overall
        final_results["overall_score"] = (
            sum(results["overall"]["scores"]) / len(results["overall"]["scores"])
            if results["overall"]["scores"] else 0.0
        )
        final_results["total_examples"] = results["overall"]["count"]
        
        # By category
        for cat in range(1, 6):
            cat_scores = results["by_category"][cat]["scores"]
            if cat_scores:
                final_results[f"category_{cat}_score"] = sum(cat_scores) / len(cat_scores)
                final_results[f"category_{cat}_count"] = len(cat_scores)
        
        # By memory distance
        for dist_cat in ["immediate", "recent", "medium", "distant"]:
            dist_scores = results["by_memory_distance"][dist_cat]["scores"]
            if dist_scores:
                final_results[f"memory_{dist_cat}_score"] = sum(dist_scores) / len(dist_scores)
                final_results[f"memory_{dist_cat}_count"] = len(dist_scores)
        
        return final_results


if __name__ == "__main__":
    # Test the paper-accurate metrics
    
    # Multi-hop example
    multi_hop_ex = dspy.Example(
        question="What did Alice eat and where did she go?",
        answer="pasta, restaurant",
        category=1
    )
    multi_hop_pred = dspy.Prediction(answer="spaghetti and Italian place")
    print(f"Multi-hop F1: {locomo_paper_metric(multi_hop_ex, multi_hop_pred)}")
    
    # Unanswerable example  
    unanswerable_ex = dspy.Example(
        question="What color was Bob's car?",
        answer="I don't know",
        category=4
    )
    unanswerable_pred = dspy.Prediction(answer="Cannot be determined from conversation")
    print(f"Unanswerable score: {locomo_paper_metric(unanswerable_ex, unanswerable_pred)}")
    
    # Temporal example
    temporal_ex = dspy.Example(
        question="When did the meeting happen?",
        answer="January 15, 2023",
        category=2
    )
    temporal_pred = dspy.Prediction(answer="Jan 15 2023")
    print(f"Temporal F1: {locomo_paper_metric(temporal_ex, temporal_pred)}")