# LOCOMO Memory Benchmark

Lean implementation of the LOCOMO (Long-Context Multi-turn Conversational QA) benchmark using DSPy and Mem0-inspired memory architecture.

## Setup

```bash
export TOGETHER_API_KEY="your_api_key_here"
uv install
```

## Usage

```bash
# Quick demo
python run_sota_locomo.py --demo --limit 20

# Full evaluation
python run_sota_locomo.py --evaluate --limit 100

# Benchmark run
python run_sota_locomo.py --benchmark
```

## Architecture

- **Memory System**: DSPy-based memory extraction and ReACT search (no embeddings)
- **Evaluation**: LLM-as-Judge methodology from Mem0 paper
- **Target**: >68% performance on LOCOMO benchmark

## Files

- `locomo/` - Core implementation
- `data/locomo10.json` - LOCOMO dataset
- `run_sota_locomo.py` - Main runner