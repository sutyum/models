import asyncio
import concurrent.futures
from typing import List, Dict, Any, Optional
import time
from google import genai
from google.genai import types
import json
from tqdm import tqdm

# Global thread pool executor
executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)

async def run_gemini_async(client, content: str, model_name: str = "gemini-2.0-flash", max_tokens: int = 0) -> Optional[str]:
    """Async wrapper for run_gemini using thread pool executor"""
    loop = asyncio.get_event_loop()
    
    def _run_gemini():
        max_retries = 3
        retry_delay = 35
        
        for attempt in range(max_retries):
            try:
                contents = [
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(text=content),
                        ],
                    ),
                ]
                
                config = types.GenerateContentConfig(
                    response_mime_type="text/plain",
                )
                
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config
                )
                
                if hasattr(response, 'text'):
                    return response.text
                elif hasattr(response, 'candidates') and response.candidates:
                    return response.candidates[0].content.parts[0].text
                else:
                    return None
                    
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    if attempt < max_retries - 1:
                        print(f"Rate limit hit, waiting {retry_delay} seconds before retry...")
                        time.sleep(retry_delay)
                        continue
                print(f"{type(e).__name__}: {e}")
                return None
    
    return await loop.run_in_executor(executor, _run_gemini)


class RateLimiter:
    """Simple rate limiter using asyncio.Semaphore"""
    def __init__(self, max_concurrent: int = 5, delay_between_calls: float = 0.5):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.delay = delay_between_calls
        self.last_call = 0
    
    async def __aenter__(self):
        await self.semaphore.acquire()
        # Ensure minimum delay between calls
        current_time = time.time()
        time_since_last = current_time - self.last_call
        if time_since_last < self.delay:
            await asyncio.sleep(self.delay - time_since_last)
        self.last_call = time.time()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.semaphore.release()


async def process_qa_batch_async(
    client,
    qa_items: List[Dict],
    in_data: Dict,
    model_name: str,
    prediction_key: str,
    rate_limiter: RateLimiter,
    progress_bar=None
) -> List[Dict]:
    """Process a batch of Q&A items asynchronously"""
    
    async def process_single_qa(qa_item, index):
        async with rate_limiter:
            # Extract question processing logic from get_gemini_answers
            qa = qa_item
            
            # Build the prompt based on category
            if qa["category"] == 2:
                question = qa["question"] + " Use DATE of CONVERSATION to answer with an approximate date."
            elif qa["category"] == 5:
                # Simplified category 5 handling for async
                question = qa["question"]
            else:
                question = qa["question"]
            
            # Get conversation context (simplified)
            speakers = list(set([d["speaker"] for d in in_data["conversation"]["session_1"]]))
            context = f"Conversation between {speakers[0]} and {speakers[1]}:\n\n"
            
            # Add conversation content (simplified - you'd want to use the full context logic)
            for session_key in sorted([k for k in in_data["conversation"].keys() if k.startswith("session_")]):
                if session_key in in_data["conversation"]:
                    for dialog in in_data["conversation"][session_key]:
                        context += f'{dialog["speaker"]}: "{dialog["text"]}"\n'
            
            # Create the full prompt
            prompt = context + "\n\nQuestion: " + question + " Short answer:"
            
            # Make the API call
            answer = await run_gemini_async(client, prompt, model_name)
            
            if answer is None:
                qa_item[prediction_key] = ""
            else:
                qa_item[prediction_key] = answer.strip()
            
            if progress_bar:
                progress_bar.update(1)
            
            return qa_item
    
    # Process all QA items concurrently
    tasks = [process_single_qa(qa, i) for i, qa in enumerate(qa_items)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle any exceptions
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Error processing QA item {i}: {result}")
            qa_items[i][prediction_key] = ""
            processed_results.append(qa_items[i])
        else:
            processed_results.append(result)
    
    return processed_results


async def evaluate_model_async(
    model_name: str,
    data_file: str,
    out_file: str,
    api_key: str,
    max_concurrent: int = 5,
    batch_size: int = 50
):
    """Main async evaluation function"""
    # Initialize client
    client = genai.Client(api_key=api_key)
    
    # Load data
    with open(data_file) as f:
        samples = json.load(f)
    
    # Initialize rate limiter
    rate_limiter = RateLimiter(max_concurrent=max_concurrent, delay_between_calls=0.5)
    
    prediction_key = f"{model_name}_prediction"
    model_key = model_name
    
    # Load existing output if it exists
    try:
        with open(out_file) as f:
            out_samples = {d["sample_id"]: d for d in json.load(f)}
    except FileNotFoundError:
        out_samples = {}
    
    # Process all samples
    all_results = []
    
    # Calculate total QA items for progress bar
    total_qa = sum(len(sample["qa"]) for sample in samples)
    
    with tqdm(total=total_qa, desc="Processing QA items") as pbar:
        for sample in samples:
            sample_id = sample["sample_id"]
            
            # Prepare output data
            if sample_id in out_samples:
                out_qa = out_samples[sample_id]["qa"].copy()
            else:
                out_qa = sample["qa"].copy()
            
            # Filter QA items that need processing
            qa_to_process = []
            qa_indices = []
            
            for i, qa in enumerate(out_qa):
                if prediction_key not in qa:
                    qa_to_process.append(qa)
                    qa_indices.append(i)
                else:
                    pbar.update(1)  # Already processed
            
            if qa_to_process:
                # Process QA items in batches
                for i in range(0, len(qa_to_process), batch_size):
                    batch = qa_to_process[i:i+batch_size]
                    
                    # Process batch asynchronously
                    processed_batch = await process_qa_batch_async(
                        client,
                        batch,
                        sample,
                        model_name,
                        prediction_key,
                        rate_limiter,
                        pbar
                    )
                    
                    # Update results
                    for j, processed_qa in enumerate(processed_batch):
                        out_qa[qa_indices[i+j]] = processed_qa
            
            all_results.append({
                "sample_id": sample_id,
                "qa": out_qa
            })
    
    # Save results
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Evaluation complete. Results saved to {out_file}")
    
    return all_results