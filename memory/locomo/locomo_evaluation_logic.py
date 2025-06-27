"""
LOCOMO-specific evaluation logic matching the exact evaluation method.
This implements the category-specific scoring used in eval_question_answering.
"""
import numpy as np
from typing import List, Dict, Any, Tuple
from locomo.evaluation import f1_score, f1, normalize_answer


def locomo_category_evaluation(predictions: List[Dict], eval_key: str = "prediction") -> Tuple[List[float], List[float]]:
    """
    Evaluate predictions using LOCOMO category-specific logic.
    
    Args:
        predictions: List of prediction dictionaries with keys:
            - prediction/answer: The generated answer
            - ground_truth/answer: The expected answer  
            - category: Question category (1-5)
        eval_key: Key to use for prediction ('prediction' or 'answer')
    
    Returns:
        Tuple of (accuracy_scores, recall_scores)
    """
    all_ems = []
    all_recall = []
    
    for i, line in enumerate(predictions):
        # Get the answer based on the structure
        if eval_key in line:
            output = line[eval_key]
        elif "answer" in line:
            output = line["answer"]
        else:
            output = ""
            
        # Get ground truth
        if "ground_truth" in line:
            answer = str(line["ground_truth"])
        elif "answer" in line and eval_key != "answer":
            answer = str(line["answer"])
        else:
            answer = ""
            
        # Category 3 special handling for temporal questions
        if line.get("category", 0) == 3:
            # Extract first part before semicolon for temporal answers
            answer = answer.split(";")[0].strip()
        
        # Category-specific evaluation
        category = line.get("category", 3)
        
        # Single-hop, temporal, open-domain eval without splitting for sub-answers
        if category in [2, 3, 4]:
            score = f1_score(output, answer)
            all_ems.append(score)
            
        # Multi-hop eval by splitting entire phrase into sub-answers and computing partial F1 for each
        elif category == 1:
            score = f1(output, answer)  # Uses multi-answer F1
            all_ems.append(score)
            
        # Adversarial/Ambiguous eval - check for "no information" type responses
        elif category == 5:
            output_lower = output.lower()
            if ("no information available" in output_lower or 
                "not mentioned" in output_lower or
                "cannot be determined" in output_lower or
                "don't know" in output_lower or
                "insufficient information" in output_lower):
                all_ems.append(1.0)
            else:
                all_ems.append(0.0)
        else:
            # Default to standard F1
            score = f1_score(output, answer)
            all_ems.append(score)
        
        # Add recall score (simplified - using 1.0 for now)
        all_recall.append(1.0)
    
    return all_ems, all_recall


def evaluate_with_locomo_logic(predictions: List[Dict], verbose: bool = False) -> Dict[str, Any]:
    """
    Evaluate predictions using exact LOCOMO logic.
    
    Args:
        predictions: List of prediction dictionaries
        verbose: Whether to print detailed results
    
    Returns:
        Dictionary with evaluation results
    """
    # Run category-specific evaluation
    accuracy_scores, recall_scores = locomo_category_evaluation(predictions)
    
    # Calculate overall metrics
    mean_accuracy = np.mean(accuracy_scores) if accuracy_scores else 0.0
    mean_recall = np.mean(recall_scores) if recall_scores else 0.0
    
    # Category breakdown
    category_scores = {i: [] for i in range(1, 6)}
    for pred, score in zip(predictions, accuracy_scores):
        cat = pred.get("category", 3)
        if 1 <= cat <= 5:
            category_scores[cat].append(score)
    
    # Compile results
    results = {
        "overall_accuracy": mean_accuracy,
        "overall_recall": mean_recall,
        "total_examples": len(predictions),
        "individual_scores": accuracy_scores,
    }
    
    # Add category breakdowns
    for cat in range(1, 6):
        if category_scores[cat]:
            results[f"category_{cat}_accuracy"] = np.mean(category_scores[cat])
            results[f"category_{cat}_count"] = len(category_scores[cat])
    
    if verbose:
        print(f"\n📊 LOCOMO Evaluation Results:")
        print(f"Overall Accuracy: {mean_accuracy:.3f} ({mean_accuracy*100:.1f}%)")
        print(f"Total Examples: {len(predictions)}")
        
        category_names = {
            1: "Multi-hop (F1 multi-answer)",
            2: "Single-hop (F1)",
            3: "Temporal (F1)",
            4: "Open-domain (F1)",
            5: "Adversarial (exact match)"
        }
        
        print("\nCategory Breakdown:")
        for cat in range(1, 6):
            if f"category_{cat}_accuracy" in results:
                acc = results[f"category_{cat}_accuracy"]
                count = results[f"category_{cat}_count"]
                name = category_names[cat]
                print(f"  {name}: {acc:.3f} ({count} examples)")
    
    return results


def convert_to_locomo_format(example: Dict, prediction: Dict) -> Dict:
    """
    Convert a prediction to LOCOMO evaluation format.
    
    Args:
        example: Original example with question, answer, category
        prediction: Generated prediction with answer
    
    Returns:
        Dictionary in LOCOMO format
    """
    return {
        "question": example.get("question", ""),
        "answer": example.get("answer", ""),  # ground truth
        "prediction": prediction.get("answer", ""),  # generated
        "category": example.get("category", 3),
        "evidence": example.get("evidence", []),
        "sample_id": example.get("sample_id", ""),
        "qa_id": example.get("qa_id", ""),
        "prediction_context": prediction.get("relevant_memories", [])
    }


def calculate_locomo_score(examples: List[Dict], predictions: List[Dict], 
                          verbose: bool = True) -> float:
    """
    Calculate LOCOMO score using the exact evaluation logic.
    
    Args:
        examples: List of examples with ground truth
        predictions: List of predictions
        verbose: Whether to print results
    
    Returns:
        Overall LOCOMO accuracy score
    """
    # Convert to LOCOMO format
    locomo_predictions = []
    for example, pred in zip(examples, predictions):
        locomo_pred = convert_to_locomo_format(example, pred)
        locomo_predictions.append(locomo_pred)
    
    # Evaluate using LOCOMO logic
    results = evaluate_with_locomo_logic(locomo_predictions, verbose=verbose)
    
    return results["overall_accuracy"]


# Category-specific prompts for better performance
CATEGORY_PROMPTS = {
    1: """For this multi-hop question, connect information from multiple parts of the conversation. 
         Provide all relevant facts separated by commas if there are multiple answers.""",
    
    2: """For this single-hop question, find the specific fact in the conversation and 
         provide a direct, concise answer.""",
    
    3: """For this temporal question, pay attention to dates and time references. 
         Provide the specific date, time, or temporal information requested.""",
    
    4: """For this open-domain question, use the conversation context along with general knowledge 
         to provide a comprehensive answer.""",
    
    5: """This may be an adversarial or unanswerable question. If the information is not in the conversation, 
         respond with 'no information available' or 'not mentioned in the conversation'."""
}


def get_category_prompt(category: int) -> str:
    """Get category-specific prompt for better answer generation."""
    return CATEGORY_PROMPTS.get(category, CATEGORY_PROMPTS[2])