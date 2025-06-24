import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import os, json
import asyncio
import argparse
from locomo.evaluation import eval_question_answering
from locomo.evaluation_stats import analyze_aggr_acc
from locomo.async_utils import evaluate_model_async
from google import genai
from google.genai import types
import numpy as np

MODEL = "gemini-2.5-flash-lite-preview-06-17"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-file", default="./outputs/results_async.json", type=str)
    parser.add_argument("--model", type=str, default=MODEL)
    parser.add_argument("--data-file", type=str, default="./data/locomo10.json")
    parser.add_argument(
        "--max-concurrent", type=int, default=20, help="Maximum concurrent API calls"
    )
    parser.add_argument(
        "--batch-size", type=int, default=50, help="Batch size for processing QA items"
    )
    parser.add_argument("--use-rag", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    return args


async def main():
    # Get arguments
    args = parse_args()

    print(f"******************  Evaluating Model {args.model} (Async) ***************")
    print(f"Max concurrent requests: {args.max_concurrent}")

    # Get API key
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")

    # Run async evaluation
    results = await evaluate_model_async(
        model_name=args.model,
        data_file=args.data_file,
        out_file=args.out_file,
        api_key=api_key,
        max_concurrent=args.max_concurrent,
        batch_size=args.batch_size,
    )

    # Run evaluation on the results
    prediction_key = f"{args.model}_prediction"
    model_key = args.model

    # Calculate F1 scores
    for result in results:
        qa_items = result["qa"]
        exact_matches, lengths, recall = eval_question_answering(
            qa_items, prediction_key
        )

        for i in range(len(qa_items)):
            qa_items[i][model_key + "_f1"] = round(exact_matches[i], 3)

    # Save results with scores
    with open(args.out_file, "w") as f:
        json.dump(results, f, indent=2)

    # Generate statistics
    analyze_aggr_acc(
        args.data_file,
        args.out_file,
        args.out_file.replace(".json", "_stats.json"),
        model_key,
        model_key + "_f1",
        rag=args.use_rag,
    )

    print("Evaluation and analysis complete!")


if __name__ == "__main__":
    asyncio.run(main())
