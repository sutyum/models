"""
Unified evaluation module for testing MemorySystem with LOCOMO benchmark
========================================================================
Tests the memory system with and without optimized prompts.
"""

import os
import json
import time
import dspy
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import argparse

from memory_system import MemorySystem
from locomo.dataset import load_locomo_dataset


class MemoryQAWithOptimization(dspy.Module):
    """Memory QA module that can use optimized prompts."""
    
    def __init__(self, memory_system: MemorySystem, optimized_prompt_file: Optional[str] = None):
        super().__init__()
        self.memory_system = memory_system
        self.optimized_demos = []
        
        if optimized_prompt_file and os.path.exists(optimized_prompt_file):
            with open(optimized_prompt_file, 'r') as f:
                prompt_data = json.load(f)
                self.optimized_demos = prompt_data.get('demos', [])
    
    def forward(self, conversation: str, question: str) -> str:
        """Process conversation and answer question using memory system."""
        # Clear memory state for each new conversation
        self.memory_system.state = ""
        
        # Update memory with conversation context
        if conversation and conversation.strip():
            self.memory_system.update(conversation)
        
        # If we have optimized demos, use them to guide the answer
        if self.optimized_demos:
            # Create a prompt with few-shot examples
            demo_prompt = "Here are some examples of how to answer questions based on conversations:\n\n"
            for demo in self.optimized_demos[:3]:  # Use top 3 demos
                demo_prompt += f"Conversation: {demo['conversation']}\n"
                demo_prompt += f"Question: {demo['question']}\n"
                demo_prompt += f"Answer: {demo['answer']}\n\n"
            
            # Include demo context in the query
            enhanced_question = f"{demo_prompt}Now, based on the current conversation, {question}"
            answer = self.memory_system.query(enhanced_question)
        else:
            # Direct query without optimization
            answer = self.memory_system.query(question)
        
        return answer


def evaluate_memory_system(
    test_data_path: str,
    model_name: str = "gpt-3.5-turbo",
    optimized_prompt_file: Optional[str] = None,
    output_file: Optional[str] = None,
    max_examples: Optional[int] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """Evaluate memory system on LOCOMO dataset."""
    
    # Load test data
    print(f"Loading test data from {test_data_path}...")
    test_examples = load_locomo_dataset(test_data_path)
    
    if max_examples:
        test_examples = test_examples[:max_examples]
    
    # Initialize language model
    if model_name.startswith("gpt"):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI models")
        lm = dspy.LM(model_name, api_key=api_key)
    else:
        # Assume Together AI model
        api_key = os.environ.get("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError("TOGETHER_API_KEY environment variable is required for Together AI models")
        lm = dspy.LM(model_name, api_key=api_key)
    
    dspy.configure(lm=lm)
    
    # Initialize memory system and QA module
    memory_system = MemorySystem(persist=False, resource_limit=20000)
    qa_module = MemoryQAWithOptimization(memory_system, optimized_prompt_file)
    
    # Evaluation metrics
    results = {
        "model": model_name,
        "optimized_prompt_file": optimized_prompt_file,
        "total_examples": len(test_examples),
        "predictions": []
    }
    
    category_scores = {f"category_{i}": [] for i in range(1, 6)}
    overall_scores = []
    
    print(f"\nEvaluating {len(test_examples)} examples...")
    print(f"Using {'optimized' if optimized_prompt_file else 'base'} prompts")
    
    start_time = time.time()
    
    # Process each example
    for example in tqdm(test_examples, desc="Evaluating"):
        try:
            # Get conversation and question from example
            conversation = example.get('conversation', '')
            question = example.get('question', '')
            expected_answer = example.get('answer', '')
            
            # Generate answer
            predicted_answer = qa_module(conversation=conversation, question=question)
            
            # Calculate similarity score (simple word overlap for now)
            # In practice, you'd use the LLM judge here
            predicted_words = set(predicted_answer.lower().split())
            expected_words = set(expected_answer.lower().split())
            overlap = len(predicted_words & expected_words)
            score = overlap / max(len(expected_words), 1)
            
            # Store result
            result = {
                "conversation": conversation,
                "question": question,
                "expected_answer": expected_answer,
                "predicted_answer": predicted_answer,
                "score": score,
                "metadata": example.get('metadata', {})
            }
            results["predictions"].append(result)
            
            # Track scores by category
            category = example.get('metadata', {}).get('category', 0)
            if 1 <= category <= 5:
                category_scores[f"category_{category}"].append(score)
            overall_scores.append(score)
            
            if verbose:
                print(f"\nQuestion: {question}")
                print(f"Expected: {expected_answer}")
                print(f"Predicted: {predicted_answer}")
                print(f"Score: {score:.3f}")
        
        except Exception as e:
            print(f"\nError processing example: {e}")
            results["predictions"].append({
                "conversation": conversation,
                "question": question,
                "error": str(e),
                "score": 0
            })
            overall_scores.append(0)
    
    # Calculate final metrics
    evaluation_time = time.time() - start_time
    
    results["evaluation_time_seconds"] = evaluation_time
    results["overall_score"] = sum(overall_scores) / len(overall_scores) if overall_scores else 0
    
    # Category-wise scores
    for category, scores in category_scores.items():
        if scores:
            results[f"{category}_score"] = sum(scores) / len(scores)
            results[f"{category}_count"] = len(scores)
        else:
            results[f"{category}_score"] = 0
            results[f"{category}_count"] = 0
    
    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Evaluation Summary")
    print(f"{'='*50}")
    print(f"Model: {model_name}")
    print(f"Optimization: {'Yes' if optimized_prompt_file else 'No'}")
    print(f"Overall Score: {results['overall_score']:.3f}")
    print(f"Total Examples: {results['total_examples']}")
    print(f"Evaluation Time: {evaluation_time:.2f}s")
    
    print(f"\nCategory Breakdown:")
    for i in range(1, 6):
        cat_score = results.get(f'category_{i}_score', 0)
        cat_count = results.get(f'category_{i}_count', 0)
        if cat_count > 0:
            print(f"  Category {i}: {cat_score:.3f} ({cat_count} examples)")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Memory System on LOCOMO Benchmark")
    parser.add_argument("--test-data", type=str, required=True, help="Path to test data")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="Model to use")
    parser.add_argument("--optimized-prompt", type=str, help="Path to optimized prompt file")
    parser.add_argument("--output", type=str, help="Output file for results")
    parser.add_argument("--max-examples", type=int, help="Maximum number of examples to evaluate")
    parser.add_argument("--verbose", action="store_true", help="Print detailed results")
    
    args = parser.parse_args()
    
    evaluate_memory_system(
        test_data_path=args.test_data,
        model_name=args.model,
        optimized_prompt_file=args.optimized_prompt,
        output_file=args.output,
        max_examples=args.max_examples,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()