#!/usr/bin/env python3
"""
Compare evaluation results between baseline and optimized models.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import pandas as pd


def load_results(filepath: str) -> Dict[str, Any]:
    """Load evaluation results from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def compare_results(baseline_path: str, optimized_path: str) -> Dict[str, Any]:
    """
    Compare baseline and optimized results.

    Args:
        baseline_path: Path to baseline results JSON
        optimized_path: Path to optimized results JSON

    Returns:
        Comparison results dictionary
    """
    try:
        baseline = load_results(baseline_path)
        optimized = load_results(optimized_path)
    except FileNotFoundError as e:
        print(f"Error loading results: {e}")
        return {"error": str(e)}

    comparison = {
        "baseline_file": baseline_path,
        "optimized_file": optimized_path,
        "baseline_score": baseline.get("overall_llm_judge_score", 0.0),
        "optimized_score": optimized.get("overall_llm_judge_score", 0.0),
        "improvement": 0.0,
        "category_comparisons": {},
        "summary": {},
    }

    # Calculate overall improvement
    if comparison["baseline_score"] > 0:
        improvement = (
            (comparison["optimized_score"] - comparison["baseline_score"])
            / comparison["baseline_score"]
        ) * 100
        comparison["improvement"] = improvement

    # Compare category-specific results
    for cat in range(1, 6):
        baseline_key = f"category_{cat}_score"
        baseline_count_key = f"category_{cat}_count"

        if (
            baseline_key in baseline
            and baseline_key in optimized
            and baseline.get(baseline_count_key, 0) > 0
            and optimized.get(baseline_count_key, 0) > 0
        ):
            baseline_cat_score = baseline[baseline_key]
            optimized_cat_score = optimized[baseline_key]

            cat_improvement = 0.0
            if baseline_cat_score > 0:
                cat_improvement = (
                    (optimized_cat_score - baseline_cat_score) / baseline_cat_score
                ) * 100

            comparison["category_comparisons"][f"category_{cat}"] = {
                "baseline_score": baseline_cat_score,
                "optimized_score": optimized_cat_score,
                "improvement": cat_improvement,
                "count": baseline.get(baseline_count_key, 0),
            }

    # Generate summary
    comparison["summary"] = {
        "overall_improvement": f"{comparison['improvement']:.2f}%",
        "baseline_total": baseline.get("total_examples", 0),
        "optimized_total": optimized.get("total_examples", 0),
        "categories_improved": sum(
            1
            for cat_data in comparison["category_comparisons"].values()
            if cat_data["improvement"] > 0
        ),
        "categories_degraded": sum(
            1
            for cat_data in comparison["category_comparisons"].values()
            if cat_data["improvement"] < 0
        ),
    }

    return comparison


def create_comparison_plots(comparison: Dict[str, Any], output_dir: str = "plots"):
    """Create visualization plots for the comparison."""
    Path(output_dir).mkdir(exist_ok=True)

    # Overall comparison bar chart
    plt.figure(figsize=(10, 6))

    scores = [comparison["baseline_score"], comparison["optimized_score"]]
    labels = ["Baseline", "Optimized"]
    colors = ["#ff7f7f", "#7f7fff"]

    bars = plt.bar(labels, scores, color=colors, alpha=0.7)
    plt.title("Overall Performance Comparison", fontsize=16)
    plt.ylabel("LLM Judge Score", fontsize=12)
    plt.ylim(0, 1.0)

    # Add value labels on bars
    for bar, score in zip(bars, scores):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=12,
        )

    # Add improvement annotation
    improvement = comparison["improvement"]
    color = "green" if improvement > 0 else "red"
    plt.text(
        0.5,
        max(scores) + 0.1,
        f"Improvement: {improvement:.2f}%",
        ha="center",
        va="center",
        fontsize=14,
        color=color,
        weight="bold",
    )

    plt.tight_layout()
    plt.savefig(f"{output_dir}/overall_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Category-wise comparison
    if comparison["category_comparisons"]:
        categories = list(comparison["category_comparisons"].keys())
        baseline_cat_scores = [
            comparison["category_comparisons"][cat]["baseline_score"]
            for cat in categories
        ]
        optimized_cat_scores = [
            comparison["category_comparisons"][cat]["optimized_score"]
            for cat in categories
        ]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Category scores comparison
        x = range(len(categories))
        width = 0.35

        ax1.bar(
            [i - width / 2 for i in x],
            baseline_cat_scores,
            width,
            label="Baseline",
            color="#ff7f7f",
            alpha=0.7,
        )
        ax1.bar(
            [i + width / 2 for i in x],
            optimized_cat_scores,
            width,
            label="Optimized",
            color="#7f7fff",
            alpha=0.7,
        )

        ax1.set_xlabel("Categories")
        ax1.set_ylabel("LLM Judge Score")
        ax1.set_title("Category-wise Performance Comparison")
        ax1.set_xticks(x)
        ax1.set_xticklabels([cat.replace("category_", "Cat ") for cat in categories])
        ax1.legend()
        ax1.set_ylim(0, 1.0)

        # Improvement percentages
        improvements = [
            comparison["category_comparisons"][cat]["improvement"] for cat in categories
        ]
        colors = ["green" if imp > 0 else "red" for imp in improvements]

        bars = ax2.bar(range(len(categories)), improvements, color=colors, alpha=0.7)
        ax2.set_xlabel("Categories")
        ax2.set_ylabel("Improvement (%)")
        ax2.set_title("Category-wise Improvement")
        ax2.set_xticks(range(len(categories)))
        ax2.set_xticklabels([cat.replace("category_", "Cat ") for cat in categories])
        ax2.axhline(y=0, color="black", linestyle="-", alpha=0.3)

        # Add value labels
        for bar, imp in zip(bars, improvements):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (1 if bar.get_height() > 0 else -3),
                f"{imp:.1f}%",
                ha="center",
                va="bottom" if bar.get_height() > 0 else "top",
                fontsize=10,
            )

        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/category_comparison.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    print(f"Plots saved to {output_dir}/")


def print_comparison_report(comparison: Dict[str, Any]):
    """Print a detailed comparison report."""
    print("\n" + "=" * 60)
    print("LOCOMO EVALUATION COMPARISON REPORT")
    print("=" * 60)

    print(f"\nBaseline Results: {comparison['baseline_file']}")
    print(f"Optimized Results: {comparison['optimized_file']}")

    print(f"\nOVERALL PERFORMANCE:")
    print(f"  Baseline Score:  {comparison['baseline_score']:.4f}")
    print(f"  Optimized Score: {comparison['optimized_score']:.4f}")
    print(f"  Improvement:     {comparison['improvement']:+.2f}%")

    if comparison["improvement"] > 0:
        print("  ✅ IMPROVEMENT DETECTED")
    elif comparison["improvement"] < 0:
        print("  ❌ PERFORMANCE DEGRADATION")
    else:
        print("  ➖ NO CHANGE")

    print(f"\nCATEGORY-WISE BREAKDOWN:")
    if comparison["category_comparisons"]:
        for cat, data in comparison["category_comparisons"].items():
            cat_name = cat.replace("category_", "Category ")
            status = (
                "✅"
                if data["improvement"] > 0
                else "❌"
                if data["improvement"] < 0
                else "➖"
            )
            print(
                f"  {cat_name}: {data['baseline_score']:.3f} → {data['optimized_score']:.3f} "
                f"({data['improvement']:+.1f}%) {status} ({data['count']} examples)"
            )
    else:
        print("  No category-wise data available")

    print(f"\nSUMMARY:")
    summary = comparison["summary"]
    print(f"  Overall Improvement: {summary['overall_improvement']}")
    print(f"  Categories Improved: {summary['categories_improved']}")
    print(f"  Categories Degraded: {summary['categories_degraded']}")
    print(f"  Total Examples: {summary['baseline_total']}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare baseline and optimized evaluation results"
    )
    parser.add_argument("baseline", help="Path to baseline results JSON file")
    parser.add_argument("optimized", help="Path to optimized results JSON file")
    parser.add_argument(
        "--output",
        default="comparison_results.json",
        help="Path to save comparison results",
    )
    parser.add_argument(
        "--plots", action="store_true", help="Generate comparison plots"
    )
    parser.add_argument("--plot-dir", default="plots", help="Directory to save plots")

    args = parser.parse_args()

    # Perform comparison
    comparison = compare_results(args.baseline, args.optimized)

    if "error" in comparison:
        print(f"Comparison failed: {comparison['error']}")
        return 1

    # Print report
    print_comparison_report(comparison)

    # Save detailed results
    with open(args.output, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nDetailed comparison saved to: {args.output}")

    # Generate plots if requested
    if args.plots:
        try:
            create_comparison_plots(comparison, args.plot_dir)
        except ImportError:
            print("Warning: matplotlib not available. Skipping plot generation.")
        except Exception as e:
            print(f"Error generating plots: {e}")

    return 0


if __name__ == "__main__":
    exit(main())
