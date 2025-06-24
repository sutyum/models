"""
Simple DSPy evaluation script for LOCOMO QA.
"""
import argparse
import os
from pathlib import Path

import dspy
from locomo.dspy_dataset import load_locomo_dataset
from locomo.dspy_modules import create_locomo_qa_module
from locomo.dspy_metrics import LocomoMetrics


def main():
    parser = argparse.ArgumentParser(description="Simple DSPy evaluation for LOCOMO")
    parser.add_argument("--data-path", default="./data/locomo10.json", help="Path to LOCOMO dataset")
    parser.add_argument("--module-type", default="chain_of_thought", 
                       choices=["basic", "chain_of_thought", "category_aware", "multi_step"])
    parser.add_argument("--limit", type=int, default=10, help="Number of examples to evaluate")
    parser.add_argument("--model", default="openai/gpt-4o-mini", help="Language model to use")
    
    args = parser.parse_args()
    
    # Setup language model
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not found. Please set your OpenAI API key.")
        return
    
    lm = dspy.LM(args.model, api_key=api_key)
    dspy.configure(lm=lm)
    
    # Load dataset
    print("📚 Loading LOCOMO dataset...")
    dataset = load_locomo_dataset(args.data_path)
    examples = dataset.get_examples(limit=args.limit)
    
    # Create module
    print(f"🤖 Creating {args.module_type} module...")
    module = create_locomo_qa_module(args.module_type)
    
    # Evaluate examples
    print(f"🔍 Evaluating {len(examples)} examples...")
    
    predictions = []
    for i, example in enumerate(examples):
        print(f"\n--- Example {i+1}/{len(examples)} ---")
        print(f"Question: {example.question}")
        print(f"Ground Truth: {example.answer}")
        print(f"Category: {example.category}")
        
        try:
            pred = module(conversation=example.conversation, question=example.question)
            predictions.append(pred)
            print(f"Prediction: {pred.answer}")
            
            if hasattr(pred, 'reasoning'):
                print(f"Reasoning: {pred.reasoning[:200]}...")
            
            # Calculate F1 score for this example
            f1_metric = LocomoMetrics.get_metric("f1")
            score = f1_metric(example, pred)
            print(f"F1 Score: {score:.3f}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            predictions.append(dspy.Prediction(answer=""))
    
    # Overall evaluation
    print(f"\n📊 Overall Evaluation:")
    results = LocomoMetrics.evaluate_predictions(examples, predictions, "f1")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()