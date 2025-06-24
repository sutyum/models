import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import os, json
from tqdm import tqdm
import argparse
from locomo.evaluation import eval_question_answering
from locomo.evaluation_stats import analyze_aggr_acc
from locomo.gemini_utils import get_gemini_answers
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
    args = parser.parse_args()
    return args


def main():
    # get arguments
    args = parse_args()

    print("******************  Evaluating Model %s ***************" % args.model)

    if "gemini" in args.model:
        # Initialize Gemini API
        client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
        gemini_model = client
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

    for data in samples:
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

    with open(args.out_file, "w") as f:
        json.dump(list(out_samples.values()), f, indent=2)

    analyze_aggr_acc(
        args.data_file,
        args.out_file,
        args.out_file.replace(".json", "_stats.json"),
        model_key,
        model_key + "_f1",
        rag=args.use_rag,
    )
    # encoder=tiktoken.encoding_for_model(args.model))


main()
