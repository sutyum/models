import json
import time
import os
import dspy
from memory_system import MemorySystem
from locomo.evaluate import LOCOMOMetric, evaluate_predictions
from optimize import MemoryQASignature, OptimizedMemoryQA, prepare_training_data
from collections import defaultdict


class SimplePredict(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(MemoryQASignature)

    def forward(self, conversation: str, question: str) -> dspy.Prediction:
        return self.predict(conversation=conversation, question=question)


class EvaluationResults:
    def __init__(self, name: str):
        self.name = name
        self.total_correct = 0
        self.total_examples = 0
        self.category_results = defaultdict(lambda: {"correct": 0, "total": 0})
        self.latencies = []
        self.predictions = []

    def add_result(self, example, prediction, score, latency, category=None):
        self.total_examples += 1
        if score > 0:
            self.total_correct += 1

        if category:
            self.category_results[category]["total"] += 1
            if score > 0:
                self.category_results[category]["correct"] += 1

        self.latencies.append(latency)
        self.predictions.append(
            {
                "question": example.question,
                "predicted": prediction.answer,
                "ground_truth": example.answer,
                "correct": score > 0,
                "latency": latency,
            }
        )

    def get_summary(self):
        overall_accuracy = (
            self.total_correct / self.total_examples if self.total_examples > 0 else 0
        )
        avg_latency = sum(self.latencies) / len(self.latencies) if self.latencies else 0

        category_accuracies = {}
        for cat, results in self.category_results.items():
            if results["total"] > 0:
                category_accuracies[cat] = results["correct"] / results["total"]

        return {
            "name": self.name,
            "overall_accuracy": overall_accuracy,
            "total_correct": self.total_correct,
            "total_examples": self.total_examples,
            "average_latency_seconds": avg_latency,
            "category_accuracies": category_accuracies,
        }


def evaluate_model(
    model: dspy.Module,
    test_data: list[dspy.Example],
    model_name: str,
    metric: LOCOMOMetric,
    categories: dict[int, str] = None,
) -> EvaluationResults:
    results = EvaluationResults(model_name)

    print(f"\nEvaluating {model_name}...")

    for idx, example in enumerate(test_data):
        start_time = time.time()

        try:
            prediction = model(
                conversation=example.conversation, question=example.question
            )
            score = metric(example, prediction)
            latency = time.time() - start_time

            category = (
                categories.get(example.get("category", 0), "unknown")
                if categories
                else None
            )
            results.add_result(example, prediction, score, latency, category)

            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{len(test_data)} examples...")

        except Exception as e:
            print(f"  Error processing example {idx}: {str(e)}")
            latency = time.time() - start_time
            results.add_result(example, dspy.Prediction(answer="Error"), 0, latency)

    return results


def compare_systems(
    test_file: str = "data/locomo_test.json",
    num_test: int = 100,
    optimized_program_path: str = None,
    use_categories: bool = True,
):
    # Configure DSPy with language model
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError("TOGETHER_API_KEY environment variable not set")
    
    MODEL = "together_ai/deepseek-ai/DeepSeek-R1-0528-tput"
    lm = dspy.LM(MODEL, api_key=api_key, max_tokens=20_000)
    dspy.configure(lm=lm)
    
    print("Loading test dataset...")
    test_data = prepare_training_data(test_file, num_test)

    categories = (
        {
            1: "multi_hop",
            2: "temporal",
            3: "single_hop",
            4: "unanswerable",
            5: "ambiguous",
        }
        if use_categories
        else None
    )

    # Categories are already included in test_data from prepare_training_data

    metric = LOCOMOMetric()

    print("Initializing models...")
    simple_model = SimplePredict()

    memory_system = MemorySystem(persist=False, resource_limit=20000)
    memory_model = OptimizedMemoryQA(memory_system)

    if optimized_program_path:
        print(f"Loading optimized program from {optimized_program_path}...")
        memory_model.load(optimized_program_path)

    simple_results = evaluate_model(
        simple_model, test_data, "Simple Predict", metric, categories
    )
    memory_results = evaluate_model(
        memory_model, test_data, "Memory System", metric, categories
    )

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    for results in [simple_results, memory_results]:
        summary = results.get_summary()
        print(f"\n{summary['name']}:")
        print(
            f"  Overall Accuracy: {summary['overall_accuracy']:.2%} ({summary['total_correct']}/{summary['total_examples']})"
        )
        print(f"  Average Latency: {summary['average_latency_seconds']:.2f}s")

        if summary["category_accuracies"]:
            print("  Category Accuracies:")
            for cat, acc in summary["category_accuracies"].items():
                print(f"    {cat}: {acc:.2%}")

    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)

    simple_summary = simple_results.get_summary()
    memory_summary = memory_results.get_summary()

    accuracy_gap = (
        memory_summary["overall_accuracy"] - simple_summary["overall_accuracy"]
    )
    print(f"Accuracy Improvement: {accuracy_gap:+.2%}")

    latency_diff = (
        memory_summary["average_latency_seconds"]
        - simple_summary["average_latency_seconds"]
    )
    print(f"Latency Difference: {latency_diff:+.2f}s")

    if categories:
        print("\nCategory-wise Improvements:")
        for cat in categories.values():
            if (
                cat in simple_summary["category_accuracies"]
                and cat in memory_summary["category_accuracies"]
            ):
                improvement = (
                    memory_summary["category_accuracies"][cat]
                    - simple_summary["category_accuracies"][cat]
                )
                print(f"  {cat}: {improvement:+.2%}")

    results_dict = {
        "simple_predict": simple_summary,
        "memory_system": memory_summary,
        "comparison": {
            "accuracy_improvement": accuracy_gap,
            "latency_difference": latency_diff,
        },
    }

    with open("evaluation_results.json", "w") as f:
        json.dump(results_dict, f, indent=2)

    print("\nDetailed results saved to evaluation_results.json")

    return results_dict


def analyze_failures(
    test_file: str = "data/locomo_test.json", num_examples: int = 20
):
    # Configure DSPy with language model
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError("TOGETHER_API_KEY environment variable not set")
    
    MODEL = "together_ai/deepseek-ai/DeepSeek-R1-0528-tput"
    lm = dspy.LM(MODEL, api_key=api_key, max_tokens=20_000)
    dspy.configure(lm=lm)
    
    print("Analyzing failure cases...")
    test_data = prepare_training_data(test_file, num_examples)

    metric = LOCOMOMetric()

    simple_model = SimplePredict()
    memory_model = OptimizedMemoryQA(MemorySystem(persist=False))

    failure_analysis = []

    for example in test_data:
        simple_pred = simple_model(
            conversation=example.conversation, question=example.question
        )
        memory_pred = memory_model(
            conversation=example.conversation, question=example.question
        )

        simple_score = metric(example, simple_pred)
        memory_score = metric(example, memory_pred)

        if simple_score == 0 and memory_score > 0:
            failure_analysis.append(
                {
                    "type": "memory_system_helped",
                    "question": example.question,
                    "ground_truth": example.answer,
                    "simple_answer": simple_pred.answer,
                    "memory_answer": memory_pred.answer,
                }
            )
        elif simple_score > 0 and memory_score == 0:
            failure_analysis.append(
                {
                    "type": "memory_system_hurt",
                    "question": example.question,
                    "ground_truth": example.answer,
                    "simple_answer": simple_pred.answer,
                    "memory_answer": memory_pred.answer,
                }
            )

    print(f"\nFound {len(failure_analysis)} interesting cases")

    with open("failure_analysis.json", "w") as f:
        json.dump(failure_analysis, f, indent=2)

    print("Failure analysis saved to failure_analysis.json")

    return failure_analysis


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate and compare memory systems")
    parser.add_argument(
        "--mode",
        choices=["compare", "analyze"],
        default="compare",
        help="Mode to run: compare systems or analyze failures",
    )
    parser.add_argument(
        "--num-test", type=int, default=100, help="Number of test examples"
    )
    parser.add_argument(
        "--optimized-path",
        type=str,
        default=None,
        help="Path to optimized program (if available)",
    )
    parser.add_argument(
        "--no-categories", action="store_true", help="Disable category-wise evaluation"
    )

    args = parser.parse_args()

    if args.mode == "compare":
        compare_systems(
            num_test=args.num_test,
            optimized_program_path=args.optimized_path,
            use_categories=not args.no_categories,
        )
    else:
        analyze_failures(num_examples=args.num_test)
