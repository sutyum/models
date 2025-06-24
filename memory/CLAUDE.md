# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a memory evaluation project for Large Language Models (LLMs), specifically focused on evaluating question-answering capabilities across different contexts. The project uses DSPy for LLM interactions and includes evaluation utilities for assessing model performance.

## Development Commands

### Environment Setup
```bash
# Using uv (detected in the project)
uv venv
source .venv/bin/activate
uv pip install -e .

# Install dependencies
uv pip install -r pyproject.toml
```

### Running the Main Script
```bash
python main.py
```

### Running Evaluation

#### Synchronous Evaluation
```bash
# Set Google API key (required for Gemini models)
export GOOGLE_API_KEY="your-api-key"

# Using uv run with default settings
uv run python locomo/evaluate_qa.py

# Using uv run with custom parameters
uv run python locomo/evaluate_qa.py \
  --out-file ./outputs/results.json \
  --model gemini-2.5-flash-lite-preview-06-17 \
  --data-file ./data/locomo10.json \
  [--use-rag] \
  [--batch-size N] \
  [--rag-mode <mode>] \
  [--top-k N] \
  [--overwrite]
```

#### Asynchronous Evaluation (Faster with Parallelization)
```bash
# Run async evaluation with concurrent API calls
uv run python locomo/evaluate_qa_async.py \
  --out-file ./outputs/results_async.json \
  --model gemini-2.5-flash-lite-preview-06-17 \
  --data-file ./data/locomo10.json \
  --max-concurrent 5 \
  --batch-size 50

# Adjust max-concurrent based on your API rate limits
# Higher values = faster processing but may hit rate limits
```

## Architecture Overview

### Core Components

1. **main.py**: Entry point that configures DSPy with Ollama (llama3.2 model) running locally on port 11434.

2. **locomo/**: Main evaluation package containing:
   - **evaluate_qa.py**: Main evaluation script for question-answering tasks. Handles model initialization, data loading, and evaluation orchestration.
   - **evaluate_qa_async.py**: Asynchronous version that processes multiple QA items concurrently for faster evaluation.
   - **evaluation.py**: Core evaluation metrics implementation including:
     - F1 score calculation with stemming
     - Exact match scoring
     - BERT score integration
     - Rouge-L scoring
     - Category-specific evaluation (single-hop, multi-hop, temporal, adversarial)
   - **evaluation_stats.py**: Aggregates evaluation results and generates statistics by:
     - Question category (1-5)
     - Memory/context length requirements
     - Recall accuracy for RAG-based approaches
   - **async_utils.py**: Asynchronous utilities for parallel API calls:
     - `run_gemini_async`: Thread-pool based async wrapper for Gemini API calls
     - `RateLimiter`: Semaphore-based rate limiting for concurrent requests
     - `evaluate_model_async`: Main async evaluation orchestrator
   - **gemini_utils_async.py**: Async version of gemini_utils with concurrent QA processing

### Data Flow

1. Input data is expected in JSON format with samples containing:
   - `sample_id`: Unique identifier
   - `qa`: Question-answer pairs with categories
   - `conversation`: Dialog history with sessions
   - `evidence`: Supporting evidence references

2. The evaluation pipeline:
   - Loads conversation data
   - Generates model predictions
   - Calculates per-sample F1 scores
   - Aggregates statistics by category and memory requirements
   - Outputs results to JSON files

### Key Dependencies

- **google-genai**: Google's Generative AI SDK for Gemini models
- **dspy**: LLM framework for model interactions
- **bert-score**: For semantic similarity evaluation
- **nltk**: For text processing and stemming
- **numpy**: Numerical computations
- **tqdm**: Progress bars
- **regex**: Regular expression operations
- **rouge**: ROUGE metric calculation
- **argparse**, **pathlib**: CLI and file handling

## Notes

- The project appears to be evaluating conversational memory capabilities of LLMs
- Question categories (1-5) represent different types of queries:
  - Category 1: Multi-hop questions
  - Category 2-4: Single-hop, temporal, open-domain
  - Category 5: Adversarial questions
- Currently configured to use local Ollama with llama3.2 model in main.py
- Evaluation script uses Google's Gemini models via the google-genai SDK
- Supports both standard and RAG-enhanced evaluation modes
- Includes rate limit handling with automatic retries for API calls
- Uses the new Google Genai API format with types.Content and types.GenerateContentConfig