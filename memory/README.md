# Memory System for Conversational QA

DSPy-based adaptive memory system for multi-hop conversational QA using the LOCOMO benchmark.

## Setup

```bash
export TOGETHER_API_KEY="your_api_key_here"
uv install
```

## Scripts

### 1. Prompt Optimization
```bash
# Optimize prompts using MIPROv2
python optimize.py --num-train 50 --num-val 20
```

### 2. Model Training (GRPO)
```bash
# Train with reinforcement learning
python train.py --num-train 600 --num-train-steps 500
```

### 3. Evaluation
```bash
# Compare Simple Predict vs Memory System
python evaluate.py --num-test 100
```

## Memory System

- **Adaptive Memory**: Token-aware storage with 20K limit
- **Multi-hop Reasoning**: Dynamic memory updates and queries  
- **LLM-as-Judge**: Evaluation using LOCOMO metrics