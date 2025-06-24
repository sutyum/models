import sys
from pathlib import Path
import random
import os, json
import asyncio
from typing import Dict, List, Optional, Tuple
import time
from locomo.async_utils import run_gemini_async, RateLimiter
from locomo.gemini_utils import (
    QA_PROMPT, QA_PROMPT_CAT_5, QA_PROMPT_BATCH, CONV_START_PROMPT,
    get_cat_5_answer, get_input_context, process_ouput
)
from google.genai import types


async def get_gemini_answers_async(
    client, 
    in_data: Dict, 
    out_data: Dict, 
    prediction_key: str, 
    args,
    rate_limiter: RateLimiter,
    progress_callback=None
) -> Dict:
    """Async version of get_gemini_answers that processes QA items concurrently"""
    
    assert len(in_data["qa"]) == len(out_data["qa"]), (
        len(in_data["qa"]),
        len(out_data["qa"]),
    )
    
    # Start instruction prompt
    speakers_names = list(
        set([d["speaker"] for d in in_data["conversation"]["session_1"]])
    )
    start_prompt = CONV_START_PROMPT.format(speakers_names[0], speakers_names[1])
    start_tokens = 100
    
    # Get conversation context once (it's the same for all questions)
    question_prompt = QA_PROMPT_BATCH + "\n".join(["Placeholder"])
    num_question_tokens = 200
    query_conv = get_input_context(
        in_data["conversation"], num_question_tokens + start_tokens, client, args
    )
    query_conv = start_prompt + query_conv
    
    # Prepare all QA tasks
    qa_tasks = []
    
    for i, qa in enumerate(in_data["qa"]):
        # Skip if already processed and not overwriting
        if prediction_key in out_data["qa"][i] and not args.overwrite:
            continue
        
        # Create task for this QA item
        task = process_single_qa_async(
            client, qa, i, query_conv, prediction_key, args, rate_limiter
        )
        qa_tasks.append((i, task))
    
    # Process all QA items concurrently
    if qa_tasks:
        print(f"Processing {len(qa_tasks)} QA items concurrently...")
        
        # Execute all tasks
        results = await asyncio.gather(
            *[task for _, task in qa_tasks],
            return_exceptions=True
        )
        
        # Update out_data with results
        for (idx, _), result in zip(qa_tasks, results):
            if isinstance(result, Exception):
                print(f"Error processing QA {idx}: {result}")
                out_data["qa"][idx][prediction_key] = ""
            else:
                out_data["qa"][idx][prediction_key] = result
            
            if progress_callback:
                progress_callback(1)
    
    return out_data


async def process_single_qa_async(
    client,
    qa: Dict,
    qa_index: int,
    query_conv: str,
    prediction_key: str,
    args,
    rate_limiter: RateLimiter
) -> str:
    """Process a single QA item asynchronously"""
    
    async with rate_limiter:
        # Prepare question based on category
        if qa["category"] == 2:
            question = qa["question"] + " Use DATE of CONVERSATION to answer with an approximate date."
        elif qa["category"] == 5:
            # Handle category 5 (adversarial)
            question = qa["question"] + " Select the correct answer: (a) {} (b) {}. "
            if random.random() < 0.5:
                question = question.format(
                    "Not mentioned in the conversation", qa["answer"]
                )
                answer_key = {
                    "a": "Not mentioned in the conversation",
                    "b": qa["answer"],
                }
            else:
                question = question.format(
                    qa["answer"], "Not mentioned in the conversation"
                )
                answer_key = {
                    "b": "Not mentioned in the conversation",
                    "a": qa["answer"],
                }
            
            # Use category 5 specific prompt
            query = query_conv + "\n\n" + QA_PROMPT_CAT_5.format(question)
            answer = await run_gemini_async(client, query, args.model)
            
            if answer is None:
                return ""
            
            return get_cat_5_answer(answer, answer_key)
        else:
            question = qa["question"]
        
        # Create query
        query = query_conv + "\n\n" + QA_PROMPT.format(question)
        
        # Get answer
        answer = await run_gemini_async(client, query, args.model)
        
        if answer is None:
            return ""
        
        return answer.strip()


async def get_gemini_answers_batch_async(
    client,
    samples: List[Dict],
    prediction_key: str,
    model_key: str,
    args,
    max_concurrent: int = 5
) -> List[Dict]:
    """Process multiple samples concurrently"""
    
    # Initialize rate limiter
    rate_limiter = RateLimiter(max_concurrent=max_concurrent, delay_between_calls=0.5)
    
    # Load existing results if any
    if os.path.exists(args.out_file):
        with open(args.out_file) as f:
            out_samples = {d["sample_id"]: d for d in json.load(f)}
    else:
        out_samples = {}
    
    # Process samples concurrently
    tasks = []
    
    for data in samples:
        out_data = {"sample_id": data["sample_id"]}
        if data["sample_id"] in out_samples:
            out_data["qa"] = out_samples[data["sample_id"]]["qa"].copy()
        else:
            out_data["qa"] = data["qa"].copy()
        
        # Create task for this sample
        task = get_gemini_answers_async(
            client, data, out_data, prediction_key, args, rate_limiter
        )
        tasks.append((data["sample_id"], out_data, task))
    
    # Process all samples
    print(f"Processing {len(tasks)} samples concurrently...")
    results = []
    
    for sample_id, out_data, task in tasks:
        try:
            updated_data = await task
            results.append(updated_data)
        except Exception as e:
            print(f"Error processing sample {sample_id}: {e}")
            results.append(out_data)
    
    return results