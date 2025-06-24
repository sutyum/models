import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import os, json
from tqdm import tqdm
import argparse
import asyncio
from locomo.evaluation import eval_question_answering
from locomo.evaluation_stats import analyze_aggr_acc
from locomo.gemini_utils import get_gemini_answers, get_gemini_answers_async
from locomo.utils import RateLimiter
from google import genai
from google.genai import types
import numpy as np

MODEL = "gemini-2.5-flash-lite-preview-06-17"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-file", default="./outputs/results.json", type=str)
    parser.add_argument("--model", type=str, default=MODEL)
    parser.add_argument("--data-file", type=str, default="./data/locomo10.json")
    parser.add_argument("--use-rag", action="store_true")
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--rag-mode", type=str, default="")
    parser.add_argument("--emb-dir", type=str, default="")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retriever", type=str, default="contriever")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--async", action="store_true", 
                       help="Use async processing for parallel API calls")
    parser.add_argument("--max-concurrent", type=int, default=5,
                       help="Maximum concurrent API calls (only for async mode)")
    args = parser.parse_args()
    return args


def process_and_save_results(samples, out_samples, args, gemini_model, prediction_key, model_key):
    """Common processing logic for both sync and async versions"""
    for data in tqdm(samples, desc="Processing samples"):
        out_data = {"sample_id": data["sample_id"]}
        if data["sample_id"] in out_samples:
            out_data["qa"] = out_samples[data["sample_id"]]["qa"].copy()
        else:
            out_data["qa"] = data["qa"].copy()

        if "gemini" in args.model:
            # get answers for each sample
            answers = get_gemini_answers(
                gemini_model, data, out_data, prediction_key, args
            )
        else:
            raise NotImplementedError

        # evaluate individual QA samples and save the score
        exact_matches, lengths, recall = eval_question_answering(
            answers["qa"], prediction_key
        )
        for i in range(0, len(answers["qa"])):
            answers["qa"][i][model_key + "_f1"] = round(exact_matches[i], 3)
            if args.use_rag and len(recall) > 0:
                answers["qa"][i][model_key + "_recall"] = round(recall[i], 3)

        out_samples[data["sample_id"]] = answers

    return out_samples


async def process_and_save_results_async(samples, out_samples, args, gemini_model, prediction_key, model_key):
    """Async version of processing logic"""
    rate_limiter = RateLimiter(max_concurrent=args.max_concurrent, delay_between_calls=0.5)
    
    # Calculate total QA items for progress tracking
    total_qa = sum(len(sample["qa"]) for sample in samples)
    pbar = tqdm(total=total_qa, desc="Processing QA items (async)")
    
    def update_progress(n=1):
        pbar.update(n)
    
    for data in samples:
        out_data = {"sample_id": data["sample_id"]}
        if data["sample_id"] in out_samples:
            out_data["qa"] = out_samples[data["sample_id"]]["qa"].copy()
        else:
            out_data["qa"] = data["qa"].copy()

        if "gemini" in args.model:
            # get answers for each sample
            answers = await get_gemini_answers_async(
                gemini_model, data, out_data, prediction_key, args, 
                rate_limiter, progress_callback=update_progress
            )
        else:
            raise NotImplementedError

        # evaluate individual QA samples and save the score
        exact_matches, lengths, recall = eval_question_answering(
            answers["qa"], prediction_key
        )
        for i in range(0, len(answers["qa"])):
            answers["qa"][i][model_key + "_f1"] = round(exact_matches[i], 3)
            if args.use_rag and len(recall) > 0:
                answers["qa"][i][model_key + "_recall"] = round(recall[i], 3)

        out_samples[data["sample_id"]] = answers
    
    pbar.close()
    return out_samples


async def main_async():
    """Async version of main function"""
    args = parse_args()
    
    print(f"******************  Evaluating Model {args.model} (Async) ***************")
    print(f"Max concurrent requests: {args.max_concurrent}")
    
    if "gemini" in args.model:
        # Initialize Gemini API
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        gemini_model = genai.Client(api_key=api_key)
    else:
        raise NotImplementedError

    # load conversations
    samples = json.load(open(args.data_file))
    prediction_key = (
        "%s_prediction" % args.model
        if not args.use_rag
        else "%s_%s_top_%s_prediction" % (args.model, args.rag_mode, args.top_k)
    )
    model_key = (
        "%s" % args.model
        if not args.use_rag
        else "%s_%s_top_%s" % (args.model, args.rag_mode, args.top_k)
    )
    
    # load the output file if it exists to check for overwriting
    if os.path.exists(args.out_file):
        out_samples = {d["sample_id"]: d for d in json.load(open(args.out_file))}
    else:
        out_samples = {}

    # Process samples asynchronously
    out_samples = await process_and_save_results_async(
        samples, out_samples, args, gemini_model, prediction_key, model_key
    )

    # Save results
    with open(args.out_file, "w") as f:
        json.dump(list(out_samples.values()), f, indent=2)

    # Generate statistics
    analyze_aggr_acc(
        args.data_file,
        args.out_file,
        args.out_file.replace(".json", "_stats.json"),
        model_key,
        model_key + "_f1",
        rag=args.use_rag,
    )


def main():
    """Main function that routes to sync or async version"""
    args = parse_args()
    
    if args.async:
        # Run async version
        asyncio.run(main_async())
        return
    
    print(f"******************  Evaluating Model {args.model} ***************")
    
    if "gemini" in args.model:
        # Initialize Gemini API
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        gemini_model = genai.Client(api_key=api_key)
    else:
        raise NotImplementedError

    # load conversations
    samples = json.load(open(args.data_file))
    prediction_key = (
        "%s_prediction" % args.model
        if not args.use_rag
        else "%s_%s_top_%s_prediction" % (args.model, args.rag_mode, args.top_k)
    )
    model_key = (
        "%s" % args.model
        if not args.use_rag
        else "%s_%s_top_%s" % (args.model, args.rag_mode, args.top_k)
    )
    
    # load the output file if it exists to check for overwriting
    if os.path.exists(args.out_file):
        out_samples = {d["sample_id"]: d for d in json.load(open(args.out_file))}
    else:
        out_samples = {}

    # Process samples synchronously
    out_samples = process_and_save_results(
        samples, out_samples, args, gemini_model, prediction_key, model_key
    )

    # Save results
    with open(args.out_file, "w") as f:
        json.dump(list(out_samples.values()), f, indent=2)

    # Generate statistics
    analyze_aggr_acc(
        args.data_file,
        args.out_file,
        args.out_file.replace(".json", "_stats.json"),
        model_key,
        model_key + "_f1",
        rag=args.use_rag,
    )


if __name__ == "__main__":
    main()