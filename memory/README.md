# Memory System for Conversational QA

DSPy-based adaptive memory system for multi-hop conversational QA using the LOCOMO benchmark.

## Overview

This project implements a token-aware memory system that:
- Handles multi-hop reasoning with adaptive memory (20K token limit)
- Supports prompt optimization via SIMBA/MIPROv2
- Enables model fine-tuning with GRPO (Gradient-based Reward Policy Optimization)
- Provides LLM-as-Judge evaluation using LOCOMO metrics

## Setup

### Basic Setup

```bash
# Install dependencies
uv install

# Set API key (for Together AI or OpenAI)
export TOGETHER_API_KEY="your_api_key_here"

# Optional: Start MLflow for experiment tracking
uv run mlflow server --backend-store-uri sqlite:///mydb.sqlite
```

### Data Preparation

```bash
# Generate train, validation, and test splits
python -m locomo.split_data
```

## Quick Start

### 1. Evaluation (Compare Baselines)

```bash
# Compare Simple Predict vs Memory System
python evaluate.py --num-test 100
```

### 2. Prompt Optimization (Recommended)

```bash
# Optimize prompts using SIMBA (fast)
python optimize.py --num-examples 30 --max-demos 3

# For faster testing
python optimize.py --num-examples 15 --max-demos 2
```

### 3. Model Fine-tuning with GRPO (Advanced)

See the [GRPO Training](#grpo-training) section below.

## GRPO Training

### Warning
GRPO is experimental and requires significant computational resources (2+ GPUs).

### GPU Requirements
- Minimum: 2 GPUs (1 for inference, 1 for training)
- Recommended: 3+ GPUs for better performance

### GRPO Setup

1. **Configure GPUs** (arbor.yaml is auto-created):
   ```yaml
   inference:
     gpu_ids: '0'
   training:
     gpu_ids: '1,2'
   ```

2. **Start Arbor server** (in a separate terminal):
   ```bash
   python -m arbor.cli serve --arbor-config arbor.yaml
   ```

3. **Run training**:
   ```bash
   python train.py --mode train \
       --num-train 600 \
       --num-val 100 \
       --num-train-steps 500 \
       --model "Qwen/Qwen2.5-3B"
   ```

### GRPO Parameters
- `--num-train`: Training examples (default: 600)
- `--num-val`: Validation examples (default: 100)
- `--num-train-steps`: Total training steps (default: 500)
- `--model`: Model to fine-tune (default: "Qwen/Qwen2.5-3B")
- `--inference-gpus`: GPU IDs for inference (default: "0")
- `--training-gpus`: GPU IDs for training (default: "1,2")

### GRPO Configuration Details
- Batch Size: 2 per device with 4 gradient accumulation steps
- Learning Rate: 2e-5 with constant warmup
- KL Penalty (β): 0.04
- LoRA: r=64, α=16
- Precision: bfloat16
- Max Prompt/Completion: 4096/512 tokens

### Evaluate GRPO Model
```bash
python train.py --mode evaluate --num-test 100
```

## Arbor Deployment on RunPod

For cloud deployment of the Arbor server, see [arbor/README.md](arbor/README.md).

### Quick RunPod Deployment

```bash
# Set RunPod API key
export RUNPOD_API_KEY=your_api_key_here

# Deploy with default settings
python arbor/deploy.py

# Deploy with specific GPU
python arbor/deploy.py --gpu-type A100

# List pods
python arbor/deploy.py --list

# Terminate pod
python arbor/deploy.py --terminate <pod-id>
```

## Project Structure

```
memory/
├── memory_system.py      # Core memory system implementation
├── optimize.py          # SIMBA/MIPROv2 prompt optimization
├── train.py            # GRPO training script
├── evaluate.py         # Evaluation script
├── locomo/             # LOCOMO dataset and metrics
│   ├── dataset.py
│   ├── evaluate.py
│   ├── llm_judge.py
│   └── split_data.py
├── arbor/              # Arbor deployment
│   ├── deploy.py       # RunPod deployment script
│   ├── startup.sh      # Container startup
│   └── requirements.txt
└── data/               # Dataset files
```

## Results & Outputs

- **Optimized prompts**: `optimized_memory_qa.json`
- **GRPO model weights**: `./grpo_checkpoints/`
- **GRPO config**: `grpo_optimized_memory_qa.json`
- **Evaluation metrics**: Printed to console and MLflow (if enabled)

## Troubleshooting

### Common Issues

1. **"No Arbor server found"**: Ensure Arbor server is running before GRPO training
2. **GPU OOM**: Reduce batch size or use fewer training examples
3. **Slow GRPO training**: Expected; consider using prompt optimization instead
4. **API rate limits**: Reduce concurrent requests or add delays
