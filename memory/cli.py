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
        sys.exit(1)


def run_full_evaluation(limit: int = None, num_threads: int = 4):
    """Run full evaluation of memory system using DSPy evaluation."""
    print(f"\n🔍 Running Full Memory System Evaluation")
    if limit:
        print(f"   (Limited to {limit} examples)")
    print(f"   Using {num_threads} threads for parallel evaluation")

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
                ground_truth=example.answer,
            ).with_inputs("question", "question_category")
        )

    # Define metric function for LLM-as-Judge evaluation
    def llm_judge_metric(example, prediction, trace=None):
        """Evaluate prediction using LLM-as-Judge."""
        try:
            evaluation = memory_system.evaluate_with_llm_judge(
                example.question, example.ground_truth, prediction.answer
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
        display_table=0,
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


def run_quick_benchmark(limit: int = 20, num_threads: int = 8):
    """Run a quick benchmark to compare with Mem0 paper results."""
    print(f"\n🏁 Running Quick Memory System Benchmark")
    print(f"   Using {limit} examples with {num_threads} threads")

    # Run smaller evaluation for speed
    print("📊 Running Quick Benchmark Evaluation")
    results = run_full_evaluation(limit=limit, num_threads=num_threads)
    final_score = results["overall_llm_judge_score"]

    # Compare with Mem0 results
    mem0_single_hop = 0.6713  # 67.13% from paper
    mem0_multi_hop = 0.5115  # 51.15% from paper
    mem0_overall_avg = (mem0_single_hop + mem0_multi_hop) / 2  # ~59.14%

    # Final results
    print(f"\n🏁 QUICK BENCHMARK RESULTS:")
    print(f"   Our Score: {final_score:.3f} ({final_score*100:.1f}%)")
    print(f"   Mem0 Single-hop: {mem0_single_hop:.3f} ({mem0_single_hop*100:.1f}%)")
    print(f"   Mem0 Multi-hop: {mem0_multi_hop:.3f} ({mem0_multi_hop*100:.1f}%)")
    print(f"   Mem0 Avg: {mem0_overall_avg:.3f} ({mem0_overall_avg*100:.1f}%)")

    return final_score


def run_benchmark(num_threads: int = 16):
    """Run full benchmark for performance."""
    print(f"\n🏁 Running Memory System Benchmark")
    print(f"   Using {num_threads} threads for maximum performance")
    print("=" * 60)

    # Run evaluation
    print("📊 Running Benchmark Evaluation")
    results = run_full_evaluation(num_threads=num_threads)  # Use substantial subset
    final_score = results["overall_llm_judge_score"]

    # Final results
    print(f"\n🏁 BENCHMARK RESULTS:")
    print(f"   Final Score: {final_score:.3f} ({final_score*100:.1f}%)")

    return final_score


def main():
    parser = argparse.ArgumentParser(
        description="Memory System for LOCOMO - CLI Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Benchmark run
    python cli.py --benchmark --num-threads 16
        """,
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--quick-benchmark",
        action="store_true",
        help="Run quick benchmark vs Mem0 paper",
    )
    mode_group.add_argument(
        "--benchmark", action="store_true", help="Run full benchmark for performance"
    )

    # Parameters
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of examples"
    )
    parser.add_argument(
        "--data-path", default="./data/locomo10.json", help="Path to LOCOMO dataset"
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="Number of threads for parallel evaluation",
    )

    args = parser.parse_args()

    # Setup
    print("🎯 Memory System")

    # Check if dataset exists
    if not Path(args.data_path).exists():
        print(f"❌ Dataset not found: {args.data_path}")
        print("   Please ensure the LOCOMO dataset is available")
        sys.exit(1)

    # Run selected mode
    try:
        if args.quick_benchmark:
            result = run_quick_benchmark(args.limit or 20, args.num_threads)
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
