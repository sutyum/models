# Memory System for Conversational QA

DSPy-based adaptive memory system for multi-hop conversational QA using the LOCOMO benchmark.

## Setup

```bash
export TOGETHER_API_KEY="your_api_key_here"
uv install

# Start mlflow server for tracing
uv run mlflow server --backend-store-uri sqlite:///mydb.sqlite
```

## Scripts

Generate Dataset Splits
```bash
# Generate train, validation, and test splits for the LOCOMO dataset
python locomo/split_data.py
```

Prompt Optimization
```bash
# Optimize prompts using SIMBA (fast)
python optimize.py --num-examples 30 --max-demos 3

# For faster testing
python optimize.py --num-examples 15 --max-demos 2
```

Model Training (GRPO)
```bash
# Train with reinforcement learning
python train.py --num-train 600 --num-train-steps 500
```

Evaluation
```bash
# Compare Simple Predict vs Memory System
python evaluate.py --num-test 100
```

## Memory System

- **Adaptive Memory**: Token-aware storage with 20K limit
- **Multi-hop Reasoning**: Dynamic memory updates and queries  
- **LLM-as-Judge**: Evaluation using LOCOMO metrics
