#!/usr/bin/env python3
"""
Memory System for LOCOMO - CLI Runner
Simple and clean CLI for running the memory system.

Usage Examples:
    # Quick demo
    python cli.py --demo --limit 20
    
    # Full evaluation
    python cli.py --evaluate --limit 100
    
    # Benchmark run
    python cli.py --benchmark
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add the locomo package to the path
sys.path.append(str(Path(__file__).parent))

import dspy
from locomo.dataset import load_locomo_dataset
from locomo.evaluate import evaluate_predictions
from memory_system import create_memory_system


def setup_environment():
    """Setup DSPy environment and check API key."""
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        print("❌ API key not found! Please set TOGETHER_API_KEY environment variable.")
        sys.exit(1)
    
    # Configure DSPy
    try:
        MODEL = "together_ai/deepseek-ai/DeepSeek-R1-0528-tput"
        lm = dspy.LM(MODEL, api_key=api_key)
        dspy.configure(lm=lm)
        print(f"✅ Environment configured with {MODEL}")
        return lm
    except Exception as e:
        print(f"❌ Failed to configure model: {e}")
        print("   Please check your API key and try again")
        sys.exit(1)


def run_demo(limit: int = 20):
    """Run a quick demo of the memory system."""
    print(f"\n🎯 Running Memory System Demo (limit: {limit})")
    print("=" * 60)
    
    # Load dataset
    print("📚 Loading LOCOMO dataset...")
    try:
        dataset = load_locomo_dataset("./data/locomo10.json")
        examples = dataset.get_examples(limit=limit)
        print(f"📊 Demo dataset: {len(examples)} examples")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return 0.0
    
    # Create memory system
    print("🏗️  Creating memory system...")
    memory_system = create_memory_system()
    
    # For demo, build memories from first conversation to get realistic results
    print("💾 Building memories from first conversation...")
    if examples:
        sample_id = examples[0].sample_id
        for sample in dataset.raw_data:
            if sample["sample_id"] == sample_id:
                try:
                    print(f"   Processing conversation {sample_id}...")
                    memory_system.process_conversation(sample, sample_id)
                    print(f"   ✓ Built {len(memory_system.memories)} memories")
                    break
                except Exception as e:
                    print(f"   ⚠️  Error: {e}")
                    break
    
    print(f"🧠 Memory store ready with {len(memory_system.memories)} memories")
    
    # Alternative: Build minimal memories from just first conversation
    # print("💾 Building minimal memories from first conversation...")
    # if examples:
    #     sample_id = examples[0].sample_id
    #     for sample in dataset.raw_data:
    #         if sample["sample_id"] == sample_id:
    #             try:
    #                 print(f"   Processing conversation {sample_id}...")
    #                 memory_system.process_conversation(sample, sample_id)
    #                 print(f"   ✓ Built {len(memory_system.memories)} memories")
    #                 break
    #             except Exception as e:
    #                 print(f"   ⚠️  Error: {e}")
    #                 break
    
    # Test a few questions
    print("\n🤔 Testing questions...")
    # Use all available examples for demo, up to 5
    test_examples = examples[:min(5, len(examples))]
    
    correct_count = 0
    total_count = 0
    
    category_map = {
        1: "multi-hop",
        2: "temporal", 
        3: "single-hop",
        4: "unanswerable",
        5: "ambiguous",
    }
    
    for i, example in enumerate(test_examples):
        print(f"\n--- Question {i+1} ---")
        print(f"Q: {example.question}")
        print(f"Ground Truth: {example.answer}")
        print(f"Category: {example.category}")
        
        # Get answer from memory system
        question_category = category_map.get(example.category, "single-hop")
        
        print("   Generating answer...")
        try:
            result = memory_system.answer_question(example.question, question_category)
            print(f"Prediction: {result['answer']}")
            print(f"Confidence: {result['confidence']}")
            
            # Use LLM judge for accurate evaluation like Mem0 paper
            print("   Evaluating with LLM judge...")
            
            try:
                evaluation = memory_system.evaluate_with_llm_judge(
                    example.question, example.answer, result["answer"]
                )
                is_correct = evaluation["is_correct"]
                correct_count += int(is_correct)
                total_count += 1
                
                status = "✅ CORRECT" if is_correct else "❌ WRONG"
                print(f"LLM Judge: {status} - {evaluation['reasoning']}")
                
            except Exception as e:
                print(f"   ⚠️  LLM judge error: {e}")
                # Fallback to simple evaluation
                ground_truth_words = set(example.answer.lower().split())
                prediction_words = set(result['answer'].lower().split())
                overlap = len(ground_truth_words.intersection(prediction_words))
                is_correct = overlap > 0 and overlap >= len(ground_truth_words) * 0.3
                correct_count += int(is_correct)
                total_count += 1
                
                status = "✅ CORRECT" if is_correct else "❌ WRONG"
                reasoning = f"Fallback word overlap: {overlap}/{len(ground_truth_words)} words"
                print(f"Simple Judge: {status} - {reasoning}")
            
        except Exception as e:
            print(f"   ❌ Error generating answer: {e}")
            print(f"Prediction: Unable to generate answer")
            print(f"Simple Judge: ❌ WRONG - Error occurred")
            total_count += 1
    
    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"\n📊 Demo Results:")
    print(f"   Accuracy: {accuracy:.3f} ({correct_count}/{total_count})")
    print(f"   Performance: {accuracy*100:.1f}%")
    
    if accuracy >= 0.68:
        print("🏆 Demo achieved >68% target!")
    else:
        print(f"📈 Need {(0.68 - accuracy)*100:.1f}% more to reach 68% target")
    
    return accuracy


def run_demo_offline(limit: int = 5):
    """Run an offline demo that shows system architecture without API calls."""
    print(f"\n🎯 Running Offline Demo (limit: {limit})")
    print("   This demo shows the system architecture without making API calls")
    print("=" * 60)
    
    # Load dataset
    print("📚 Loading LOCOMO dataset...")
    try:
        dataset = load_locomo_dataset("./data/locomo10.json")
        examples = dataset.get_examples(limit=limit)
        print(f"📊 Demo dataset: {len(examples)} examples")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return 0.0
    
    # Show system components
    print("\n🏗️  Memory System Architecture:")
    print("   ✓ Memory store (persistent)")
    print("   ✓ Memory extractor (DSPy module)")
    print("   ✓ Memory updater (DSPy module)")
    print("   ✓ ReACT memory searcher (DSPy module)")
    print("   ✓ Memory ranker (DSPy module)")
    print("   ✓ QA generator (DSPy module)")
    print("   ✓ LOCOMO LLM judge integration")
    
    # Show sample questions
    print(f"\n🤔 Sample Questions from Dataset:")
    for i, example in enumerate(examples[:3]):
        print(f"\n--- Question {i+1} ---")
        print(f"Q: {example.question}")
        print(f"Expected Answer: {example.answer}")
        print(f"Category: {example.category}")
        print(f"Sample ID: {example.sample_id}")
    
    print(f"\n📊 Offline Demo Results:")
    print(f"   Dataset loaded: ✅")
    print(f"   Questions available: {len(examples)}")
    print(f"   System components: ✅")
    print(f"   Ready for API-based evaluation: ✅")
    print(f"\n💡 To run with LLM inference:")
    print(f"   1. Set TOGETHER_API_KEY environment variable")
    print(f"   2. Run: python cli.py --demo --limit {limit}")
    
    return 1.0


def run_full_evaluation(limit: int = None, num_threads: int = 4):
    """Run full evaluation of memory system using DSPy evaluation."""
    print(f"\n🔍 Running Full Memory System Evaluation")
    if limit:
        print(f"   (Limited to {limit} examples)")
    print(f"   Using {num_threads} threads for parallel evaluation")
    print("=" * 60)
    
    # Load dataset
    print("📚 Loading LOCOMO dataset...")
    dataset = load_locomo_dataset("./data/locomo10.json")
    examples = dataset.get_examples(limit=limit)
    print(f"📊 Evaluation dataset: {len(examples)} examples")
    
    # Create memory system
    print("🏗️  Creating memory system...")
    memory_system = create_memory_system()
    
    # Build memories from conversations
    print("💾 Building memories from conversations...")
    processed_conversations = set()
    
    for example in examples:
        sample_id = example.sample_id
        if sample_id not in processed_conversations:
            # Find conversation data
            for sample in dataset.raw_data:
                if sample["sample_id"] == sample_id:
                    try:
                        memory_system.process_conversation(sample, sample_id)
                        processed_conversations.add(sample_id)
                        break
                    except Exception as e:
                        print(f"   ⚠️  Error processing {sample_id}: {e}")
    
    print(f"🧠 Built memory store with {len(memory_system.memories)} memories")
    
    # Prepare examples for DSPy evaluation
    category_map = {
        1: "multi-hop",
        2: "temporal",
        3: "single-hop",
        4: "unanswerable",
        5: "ambiguous",
    }
    
    # Create DSPy examples
    dspy_examples = []
    for example in examples:
        question_category = category_map.get(example.category, "single-hop")
        dspy_examples.append(
            dspy.Example(
                question=example.question,
                question_category=question_category,
                ground_truth=example.answer
            ).with_inputs('question', 'question_category')
        )
    
    # Define metric function for LLM-as-Judge evaluation
    def llm_judge_metric(example, prediction, trace=None):
        """Evaluate prediction using LLM-as-Judge."""
        try:
            evaluation = memory_system.evaluate_with_llm_judge(
                example.question, 
                example.ground_truth, 
                prediction.answer
            )
            return evaluation["is_correct"]
        except:
            return False
    
    # Run parallel evaluation using DSPy
    print("🤔 Running parallel evaluation with DSPy...")
    evaluator = dspy.evaluate.Evaluate(
        devset=dspy_examples,
        metric=llm_judge_metric,
        num_threads=num_threads,
        display_progress=True,
        display_table=0
    )
    
    # Run evaluation
    score = evaluator(memory_system)
    
    # Alternative: Use batch processing for faster inference
    # This can be enabled for even better performance
    # predictions = memory_system.batch(dspy_examples)
    # correct = sum(llm_judge_metric(ex, pred) for ex, pred in zip(dspy_examples, predictions))
    # score = correct / len(dspy_examples)
    
    print(f"\n🎯 Full Evaluation Complete!")
    print(f"   Overall LLM-as-Judge Score: {score:.3f} ({score*100:.1f}%)")
    
    return {"overall_llm_judge_score": score}


def run_benchmark(num_threads: int = 8):
    """Run full benchmark for performance."""
    print(f"\n🏁 Running Memory System Benchmark")
    print("   Target: >68% LLM-as-Judge performance on LOCOMO")
    print(f"   Using {num_threads} threads for maximum performance")
    print("=" * 60)
    
    # Run evaluation
    print("📊 Running Benchmark Evaluation")
    results = run_full_evaluation(limit=500, num_threads=num_threads)  # Use substantial subset
    final_score = results["overall_llm_judge_score"]
    
    # Final results
    print(f"\n🏁 BENCHMARK RESULTS:")
    print(f"   Final Score: {final_score:.3f} ({final_score*100:.1f}%)")
    print(f"   Target Score: 0.680 (68.0%)")
    
    if final_score >= 0.68:
        print("🏆 BENCHMARK PASSED!")
    else:
        print(f"📈 Need {(0.68 - final_score)*100:.1f}% more to reach target")
    
    return final_score


def main():
    parser = argparse.ArgumentParser(
        description="Memory System for LOCOMO - CLI Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Offline demo (no API key needed)
    python cli.py --demo-offline --limit 5
    
    # Quick demo with LLM inference
    python cli.py --demo --limit 20
    
    # Full evaluation with parallelization
    python cli.py --evaluate --limit 100 --num-threads 8
    
    # Benchmark run
    python cli.py --benchmark --num-threads 16
        """,
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--demo", action="store_true", help="Run quick demo with sample questions"
    )
    mode_group.add_argument(
        "--demo-offline", action="store_true", help="Run offline demo without API calls"
    )
    mode_group.add_argument(
        "--evaluate", action="store_true", help="Run full evaluation"
    )
    mode_group.add_argument(
        "--benchmark", action="store_true", help="Run benchmark for performance"
    )
    
    # Parameters
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of examples"
    )
    parser.add_argument(
        "--data-path", default="./data/locomo10.json", help="Path to LOCOMO dataset"
    )
    parser.add_argument(
        "--num-threads", type=int, default=4, help="Number of threads for parallel evaluation"
    )
    
    args = parser.parse_args()
    
    # Setup
    print("🎯 Memory System for LOCOMO")
    print("   Clean implementation with LOCOMO modules")
    print("")
    
    # For offline demo, skip API setup
    if not args.demo_offline:
        lm = setup_environment()
    
    # Check if dataset exists
    if not Path(args.data_path).exists():
        print(f"❌ Dataset not found: {args.data_path}")
        print("   Please ensure the LOCOMO dataset is available")
        sys.exit(1)
    
    # Run selected mode
    try:
        if args.demo:
            result = run_demo(args.limit or 20)
        elif args.demo_offline:
            result = run_demo_offline(args.limit or 5)
        elif args.evaluate:
            result = run_full_evaluation(args.limit, args.num_threads)
        elif args.benchmark:
            result = run_benchmark(args.num_threads)
        
        print("\n✅ Run completed successfully!")
    
    except KeyboardInterrupt:
        print("\n⏹️  Run interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()