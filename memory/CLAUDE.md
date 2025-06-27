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

```bash
# Set Google API key (required for Gemini models)
export GOOGLE_API_KEY="your-api-key"

# Basic usage (async by default with 10 concurrent requests)
uv run python locomo/evaluate_qa.py

# Quick testing with limited samples
uv run python locomo/evaluate_qa.py --max-samples 20

# Synchronous processing (slower but more conservative)
uv run python locomo/evaluate_qa.py --sync

# Custom async settings
uv run python locomo/evaluate_qa.py \
  --max-concurrent 15 \
  --max-samples 50 \
  --out-file ./outputs/custom_results.json

# Full parameter example
uv run python locomo/evaluate_qa.py \
  --out-file ./outputs/results.json \
  --model gemini-2.5-flash-lite-preview-06-17 \
  --data-file ./data/locomo10.json \
  --max-samples 100 \
  --max-concurrent 10 \
  [--sync] \
  [--use-rag] \
  [--batch-size N] \
  [--overwrite]
```

#### Performance Notes
- **Async mode (default)**: ~5-10x faster with concurrent API calls
- **Sync mode**: Safer for API rate limits but much slower
- **max-concurrent**: Higher values = faster but may hit rate limits
- **max-samples**: Limits conversation samples (dataset has 10 total)
- **max-qa-items**: Limits total QA items across all samples (useful for testing)

### DSPy Integration

The project includes a complete DSPy wrapper for using LOCOMO as a benchmark for optimizing conversational QA systems.

#### Quick DSPy Evaluation
```bash
# Set OpenAI API key for DSPy
export OPENAI_API_KEY="your-openai-key"

# Simple evaluation with 10 examples
uv run python locomo/dspy_evaluate.py --limit 10

# Test different module types
uv run python locomo/dspy_evaluate.py --module-type category_aware --limit 5
```

#### DSPy Optimization with MIPRO
```bash
# Basic optimization (50 trials, ~100 examples)
uv run python locomo/dspy_optimizer.py

# Quick test optimization
uv run python locomo/dspy_optimizer.py \
  --limit-examples 50 \
  --num-trials 20 \
  --module-type chain_of_thought

# Optimize for specific categories
uv run python locomo/dspy_optimizer.py \
  --categories 1 3 5 \
  --limit-examples 100 \
  --metric category_aware

# Skip optimization, just evaluate baseline
uv run python locomo/dspy_optimizer.py \
  --skip-optimization \
  --limit-examples 50
```

#### DSPy Module Types
- **basic**: Simple prediction without reasoning
- **chain_of_thought**: Step-by-step reasoning before answering
- **category_aware**: Different strategies per question category
- **multi_step**: Context extraction followed by answer generation

#### DSPy Metrics
- **f1**: F1 score with category-specific handling
- **exact_match**: Exact string matching
- **category_aware**: Category-specific evaluation strategies
- **comprehensive**: Multi-aspect scoring with penalties

## Architecture Overview

### Core Components

1. **main.py**: Entry point that configures DSPy with Ollama (llama3.2 model) running locally on port 11434.

2. **locomo/**: Main evaluation package containing:
   - **evaluate_qa.py**: Unified evaluation script supporting both sync/async processing:
     - Async mode (default): Concurrent API calls for faster evaluation
     - Sync mode: Sequential processing for conservative rate limiting
     - Configurable sample limits for testing
   - **utils.py**: Core utilities for API interactions:
     - `run_gemini`: Synchronous Gemini API calls with rate limiting
     - `run_gemini_async`: Thread-pool based async wrapper
     - `RateLimiter`: Semaphore-based concurrent request control
   - **gemini_utils.py**: Gemini-specific evaluation logic:
     - `get_gemini_answers`: Synchronous QA processing
     - `get_gemini_answers_async`: Concurrent QA processing
     - Category-specific prompt handling and answer extraction
   - **evaluation.py**: Core evaluation metrics implementation:
     - F1 score calculation with stemming
     - Exact match scoring
     - BERT score integration
     - Rouge-L scoring
     - Category-specific evaluation (single-hop, multi-hop, temporal, adversarial)
   - **evaluation_stats.py**: Aggregates evaluation results and generates statistics by:
     - Question category (1-5)
     - Memory/context length requirements
     - Recall accuracy for RAG-based approaches
   - **DSPy Integration**:
     - **dspy_dataset.py**: Converts LOCOMO data to DSPy Examples with conversation formatting
     - **dspy_modules.py**: DSPy signatures and modules for conversational QA
     - **dspy_metrics.py**: Category-aware metrics for DSPy evaluation and optimization
     - **dspy_optimizer.py**: MIPRO optimization pipeline for systematic prompt improvement
     - **dspy_evaluate.py**: Simple evaluation script for testing DSPy modules

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
- **dspy**: LLM framework for model interactions (original main.py)
- **dspy-ai**: Advanced DSPy framework for prompt optimization and evaluation
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
- DO NOT GENERATE NEW FILES OR FOLDERS
