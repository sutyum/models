# GRPO Training for Memory System

This guide explains how to use GRPO (Gradient-based Reward Policy Optimization) to fine-tune a language model for the memory-augmented QA task.

## ⚠️ WARNING
This feature is experimental and in proof-of-concept stage. It requires significant computational resources and the Arbor RL server.

## GPU Requirements
   - At least 2 GPUs (1 for inference, 1 for training)
   - Recommended: 3+ GPUs for better performance

## Setup

1. **Split your data** (if not already done):
   ```bash
   python -m locomo.split_data
   ```

2. **Create Arbor config** (automatically created when running train.py):
   ```yaml
   # arbor.yaml
   inference:
     gpu_ids: '0'
   
   training:
     gpu_ids: '1,2'
   ```

3. **Start Arbor server** (in a separate terminal):
   ```bash
   python -m arbor.cli serve --arbor-config arbor.yaml
   ```

## Training

### Basic Training
```bash
python train.py --mode train
```

### Custom Configuration
```bash
python train.py --mode train \
    --num-train 600 \
    --num-val 100 \
    --num-train-steps 500 \
    --model "Qwen/Qwen2.5-3B" \
    --inference-gpus "0" \
    --training-gpus "1,2"
```

### Parameters
- `--num-train`: Number of training examples (default: 600)
- `--num-val`: Number of validation examples (default: 100)
- `--num-train-steps`: Total GRPO training steps (default: 500)
- `--model`: Model to fine-tune (default: "Qwen/Qwen2.5-3B")
- `--inference-gpus`: GPU IDs for inference (default: "0")
- `--training-gpus`: GPU IDs for training (default: "1,2")

## Evaluation

Evaluate the GRPO-trained model:
```bash
python train.py --mode evaluate --num-test 100
```

## Training Details

### GRPO Configuration
- **Batch Size**: 2 per device with 4 gradient accumulation steps
- **Learning Rate**: 2e-5 with constant warmup scheduler
- **KL Penalty (β)**: 0.04
- **LoRA**: Enabled with r=64, α=16
- **Precision**: bfloat16
- **Max Prompt Length**: 4096 tokens
- **Max Completion Length**: 512 tokens

### Memory System Integration
The training script:
1. Loads LOCOMO conversation data
2. Initializes the memory system with 20K token limit
3. Uses GRPO to optimize the memory-augmented QA module
4. Evaluates using the LOCOMOMetric (LLM-as-Judge)

## Expected Results

Based on the tutorial, GRPO training can improve performance but typically:
- Takes 18+ hours for significant improvements
- May be less cost-effective than prompt optimization (MIPROv2/SIMBA)
- Works best with smaller models that can be fine-tuned locally

## Troubleshooting

1. **"No Arbor server found"**: Make sure the Arbor server is running
2. **GPU OOM**: Reduce batch size or use fewer training examples
3. **Slow training**: This is expected; GRPO is computationally intensive

## Alternative Approaches

For better cost/quality trade-offs, consider:
- **MIPROv2**: `python optimize.py` (prompt optimization)
- **Simple Evaluation**: `python evaluate.py` (compare baselines)

## Notes

- The trained model weights are saved in `./grpo_checkpoints/`
- The optimized program configuration is saved to `grpo_optimized_memory_qa.json`
- Training logs and metrics are reported during training
- Use a separate evaluation LM for unbiased metric computation
