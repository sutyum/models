"""
GRPO Training Script for Memory System
======================================
This script uses DSPy's GRPO (Gradient-based Reward Policy Optimization) to fine-tune
a language model for the memory-augmented QA task on LOCOMO dataset.

WARNING: This is experimental and requires the Arbor RL server.
"""

import os
import json
import dspy
import wandb
from datetime import datetime
from dspy.teleprompt.grpo import GRPO
from dspy.clients.lm_local_arbor import ArborProvider
from memory_system import MemorySystem
from optimize import MemoryQASignature, OptimizedMemoryQA
from locomo.dataset import LocomoDataset
from locomo.evaluate import LOCOMOMetric


def prepare_training_data_from_dataset(
    dataset: LocomoDataset, num_examples: int = 100
) -> list[dspy.Example]:
    """Prepare training data using LocomoDataset."""
    examples = dataset.get_examples(limit=num_examples)
    return examples


def upload_model_to_hf(
    model_path: str,
    repo_id: str,
    model_name: str,
    train_config: dict,
    num_train_steps: int,
):
    """
    Upload the trained model to Hugging Face Hub.

    Args:
        model_path: Path to the saved model JSON file
        repo_id: Hugging Face repository ID
        model_name: Base model name that was fine-tuned
        train_config: Training configuration dictionary
        num_train_steps: Number of training steps completed
    """
    from huggingface_hub import HfApi, create_repo
    import tempfile
    import shutil

    # Create repository if it doesn't exist
    api = HfApi()
    try:
        create_repo(repo_id, exist_ok=True)
    except Exception as e:
        print(f"Repository might already exist: {e}")

    # Create a temporary directory for the model files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Copy the model file
        model_filename = "grpo_memory_qa_model.json"
        shutil.copy(model_path, f"{temp_dir}/{model_filename}")

        # Create README.md
        readme_content = f"""---
base_model: {model_name}
library_name: dspy
tags:
- grpo
- memory-system
- conversational-qa
- locomo
- question-answering
license: apache-2.0
---

# Memory System GRPO Model

This model was trained using GRPO (Gradient-based Reward Policy Optimization) on the LOCOMO conversational QA dataset.

## Model Details

- **Base Model**: {model_name}
- **Training Method**: GRPO (DSPy)
- **Dataset**: LOCOMO conversational QA
- **Training Steps**: {num_train_steps}
- **Memory System**: Token-aware adaptive memory with {train_config.get('max_prompt_length', 4096)} token budget

## Training Configuration

- **Batch Size**: {train_config.get('per_device_train_batch_size', 2)} per device
- **Gradient Accumulation Steps**: {train_config.get('gradient_accumulation_steps', 4)}
- **Learning Rate**: {train_config.get('learning_rate', 2e-5)}
- **LoRA**: r={train_config.get('lora_r', 64)}, alpha={train_config.get('lora_alpha', 16)}
- **KL Penalty (β)**: {train_config.get('beta', 0.04)}

## Usage

```python
import dspy
from memory_system import MemorySystem
from optimize import OptimizedMemoryQA

# Load the model
memory_system = MemorySystem(persist=False, resource_limit=20_000)
model = OptimizedMemoryQA(memory_system)
model.load("{model_filename}")

# Use for conversational QA
result = model(
    conversation="...",  # Multi-session conversation text
    question="..."       # Question about the conversation
)
print(result.answer)
```

## Performance

This model uses an adaptive memory system that:
- Maintains conversation history within token limits
- Performs multi-hop reasoning over stored memories
- Updates memory state dynamically based on new information

## Citation

If you use this model, please cite:

```bibtex
@misc{{memory-system-grpo,
  title={{Memory System GRPO Model}},
  author={{Generated via DSPy GRPO}},
  year={{2024}},
  url={{https://huggingface.co/{repo_id}}}
}}
```
"""

        with open(f"{temp_dir}/README.md", "w") as f:
            f.write(readme_content)

        # Create model card metadata
        config_content = f"""{{
  "model_name": "{model_name}",
  "training_method": "GRPO",
  "dataset": "LOCOMO",
  "num_train_steps": {num_train_steps},
  "training_config": {json.dumps(train_config, indent=2)}
}}"""

        with open(f"{temp_dir}/training_config.json", "w") as f:
            f.write(config_content)

        # Upload all files
        api.upload_folder(
            folder_path=temp_dir,
            repo_id=repo_id,
            commit_message=f"Upload GRPO-trained memory system model ({num_train_steps} steps)",
        )

    print(f"Model uploaded successfully to: https://huggingface.co/{repo_id}")


def setup_arbor_lm(
    model_name: str = "Qwen/Qwen2.5-3B", port: int = 7453, temperature: float = 0.7
) -> dspy.LM:
    """
    Setup Arbor-served local language model for GRPO training.

    Args:
        model_name: Name of the model to train
        port: Port where Arbor server is running
        temperature: Sampling temperature

    Returns:
        Configured DSPy LM instance
    """
    local_lm = dspy.LM(
        model=f"openai/arbor:{model_name}",
        provider=ArborProvider(),
        temperature=temperature,
        api_base=f"http://localhost:{port}/v1/",
        api_key="arbor",
        max_tokens=20_000,
    )

    return local_lm


def create_arbor_config(
    inference_gpus: str = "0",
    training_gpus: str = "1,2",
    config_path: str = "arbor.yaml",
):
    """Create Arbor configuration file if it doesn't exist."""
    if not os.path.exists(config_path):
        config = {
            "inference": {"gpu_ids": inference_gpus},
            "training": {"gpu_ids": training_gpus},
        }

        import yaml

        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        print(f"Created Arbor config at {config_path}")
        print(
            "Please start Arbor server with: python -m arbor.cli serve --arbor-config arbor.yaml"
        )
        return False
    return True


def grpo_train_memory_system(
    train_file: str = "data/locomo_train.json",
    val_file: str = "data/locomo_val.json",
    num_train: int = 600,
    num_val: int = 100,
    num_train_steps: int = 500,
    model_name: str = "Qwen/Qwen2.5-3B",
    output_dir: str = "./grpo_checkpoints",
    use_evaluation_lm: bool = True,
    wandb_project: str = "memory-system-grpo",
    hf_repo_id: str = None,
    push_to_hub: bool = True,
):
    """
    Train the memory system using GRPO.

    Args:
        train_file: Path to training data
        val_file: Path to validation data
        num_train: Number of training examples
        num_val: Number of validation examples
        num_train_steps: Total number of GRPO training steps
        model_name: Model to fine-tune
        output_dir: Directory to save checkpoints
        use_evaluation_lm: Whether to use a separate LM for evaluation
        wandb_project: Wandb project name for logging
        hf_repo_id: Hugging Face repository ID for model storage
        push_to_hub: Whether to push the trained model to Hugging Face Hub
    """
    # Initialize Wandb
    if hf_repo_id is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hf_repo_id = f"memory-system-grpo-{timestamp}"

    wandb.init(
        project=wandb_project,
        name=f"grpo-{model_name.replace('/', '-')}-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "model_name": model_name,
            "num_train": num_train,
            "num_val": num_val,
            "num_train_steps": num_train_steps,
            "train_file": train_file,
            "val_file": val_file,
            "hf_repo_id": hf_repo_id,
        },
    )

    # Check if Arbor config exists
    if not create_arbor_config():
        wandb.finish()
        return

    print(f"Loading training data from {train_file}...")
    train_dataset = LocomoDataset(train_file)
    trainset = prepare_training_data_from_dataset(train_dataset, num_train)

    print(f"Loading validation data from {val_file}...")
    val_dataset = LocomoDataset(val_file)
    valset = prepare_training_data_from_dataset(val_dataset, num_val)

    print(f"Loaded {len(trainset)} training and {len(valset)} validation examples")

    # Log dataset statistics to Wandb
    train_stats = train_dataset.get_stats()
    val_stats = val_dataset.get_stats()

    wandb.log(
        {
            "dataset/train_total": train_stats["total_examples"],
            "dataset/train_samples": train_stats["total_samples"],
            "dataset/val_total": val_stats["total_examples"],
            "dataset/val_samples": val_stats["total_samples"],
            "dataset/train_categories": train_stats["category_distribution"],
            "dataset/val_categories": val_stats["category_distribution"],
        }
    )

    # Setup the training LM (Arbor-served)
    print("Setting up Arbor LM for training...")
    train_lm = setup_arbor_lm(model_name=model_name)

    # Setup evaluation LM if requested
    if use_evaluation_lm:
        print("Setting up evaluation LM...")
        eval_api_key = os.getenv("TOGETHER_API_KEY")
        if not eval_api_key:
            raise ValueError("TOGETHER_API_KEY required for evaluation LM")

        eval_lm = dspy.LM(
            "together_ai/deepseek-ai/DeepSeek-R1-0528-tput",
            api_key=eval_api_key,
            max_tokens=20_000,
        )
    else:
        eval_lm = train_lm

    # Configure DSPy with the training LM
    dspy.configure(lm=train_lm)

    # Create the memory QA program
    print("Initializing memory QA system...")
    memory_system = MemorySystem(persist=False, resource_limit=20_000)
    program = OptimizedMemoryQA(memory_system)
    program.set_lm(train_lm)

    # Setup evaluation metric
    metric = LOCOMOMetric()
    metric.set_lm(eval_lm)  # Use separate LM for evaluation

    # GRPO training configuration with Wandb and HF integration
    train_kwargs = {
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "temperature": 0.7,
        "beta": 0.04,  # KL penalty coefficient
        "learning_rate": 2e-5,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "bf16": True,  # Use bfloat16 for training
        "lr_scheduler_type": "constant_with_warmup",
        "warmup_steps": 50,
        "max_prompt_length": 4096,  # Limit prompt length
        "max_completion_length": 512,  # Limit completion length
        "scale_rewards": True,
        "max_grad_norm": 0.5,
        "lora": True,  # Use LoRA for efficient fine-tuning
        "lora_r": 64,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "output_dir": output_dir,
        "save_steps": 50,
        "logging_steps": 10,
        "report_to": ["wandb"] if wandb.run else [],
        "push_to_hub": push_to_hub,
        "hub_model_id": hf_repo_id,
        "hub_strategy": "checkpoint",
    }

    # Log training config to Wandb
    wandb.config.update(train_kwargs)

    # Create GRPO compiler
    print("Setting up GRPO compiler...")
    compiler = GRPO(
        metric=metric,
        multitask=True,  # Support multiple tasks/modules
        num_dspy_examples_per_grpo_step=6,  # Examples per gradient step
        num_samples_per_input=8,  # Samples for reward estimation
        exclude_demos=True,  # Don't include demos in training
        num_train_steps=num_train_steps,
        num_threads=24,
        use_train_as_val=False,
        num_steps_for_val=10,  # Validation frequency
        train_kwargs=train_kwargs,
        report_train_scores=True,
    )

    # Start GRPO training
    print("\nStarting GRPO training...")
    print(f"Model: {model_name}")
    print(f"Training steps: {num_train_steps}")
    print(f"Output directory: {output_dir}")

    try:
        optimized_program = compiler.compile(
            student=program,
            trainset=trainset,
            valset=valset,
        )

        # Save the optimized program locally
        model_save_path = (
            f"grpo_optimized_memory_qa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        optimized_program.save(model_save_path)
        print(f"\nTraining complete! Saved optimized program to {model_save_path}")

        # Log final metrics to Wandb
        wandb.log(
            {
                "training/completed": True,
                "training/final_model_path": model_save_path,
            }
        )

        # Upload model to Hugging Face if requested
        if push_to_hub:
            try:
                print(f"Uploading model to Hugging Face: {hf_repo_id}")
                upload_model_to_hf(
                    model_path=model_save_path,
                    repo_id=hf_repo_id,
                    model_name=model_name,
                    train_config=train_kwargs,
                    num_train_steps=num_train_steps,
                )
                wandb.log(
                    {"huggingface/uploaded": True, "huggingface/repo_id": hf_repo_id}
                )
                print(
                    f"✅ Model successfully uploaded to: https://huggingface.co/{hf_repo_id}"
                )
            except Exception as e:
                print(f"❌ Failed to upload to Hugging Face: {e}")
                wandb.log({"huggingface/uploaded": False, "huggingface/error": str(e)})

        wandb.finish()
        return optimized_program

    except Exception as e:
        print(f"\nError during training: {e}")
        print(
            "Make sure Arbor server is running with: python -m arbor.cli serve --arbor-config arbor.yaml"
        )
        wandb.log({"training/error": str(e)})
        wandb.finish()
        raise


def evaluate_grpo_model(
    model_path: str = "grpo_optimized_memory_qa.json",
    test_file: str = "data/locomo_test.json",
    num_test: int = 100,
):
    """
    Evaluate the GRPO-trained model on test set.

    Args:
        model_path: Path to saved GRPO model
        test_file: Path to test data
        num_test: Number of test examples
    """
    # Setup evaluation LM
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError("TOGETHER_API_KEY environment variable not set")

    eval_lm = dspy.LM(
        "together_ai/deepseek-ai/DeepSeek-R1-0528-tput",
        api_key=api_key,
        max_tokens=20_000,
    )
    dspy.configure(lm=eval_lm)

    print("Loading test data...")
    test_data = prepare_training_data(test_file, num_test)

    print("Loading GRPO-optimized model...")
    memory_system = MemorySystem(persist=False, resource_limit=20_000)
    optimized_program = OptimizedMemoryQA(memory_system)
    optimized_program.load(model_path)

    # Evaluate
    metric = LOCOMOMetric()
    correct = 0

    print("Evaluating on test set...")
    for i, example in enumerate(test_data):
        try:
            pred = optimized_program(
                conversation=example.conversation, question=example.question
            )
            score = metric(example, pred)
            if score > 0:
                correct += 1

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(test_data)} examples...")

        except Exception as e:
            print(f"Error on example {i}: {e}")

    accuracy = correct / len(test_data)
    print(f"\nTest accuracy: {accuracy:.2%} ({correct}/{len(test_data)})")

    return accuracy


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GRPO training for memory system")
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate"],
        default="train",
        help="Mode: train or evaluate",
    )
    parser.add_argument(
        "--num-train", type=int, default=600, help="Number of training examples"
    )
    parser.add_argument(
        "--num-val", type=int, default=100, help="Number of validation examples"
    )
    parser.add_argument(
        "--num-train-steps", type=int, default=500, help="Number of GRPO training steps"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-3B",
        help="Model to fine-tune (must be available locally)",
    )
    parser.add_argument(
        "--inference-gpus",
        type=str,
        default="0",
        help="GPU IDs for inference (comma-separated)",
    )
    parser.add_argument(
        "--training-gpus",
        type=str,
        default="1,2",
        help="GPU IDs for training (comma-separated)",
    )
    parser.add_argument(
        "--num-test",
        type=int,
        default=100,
        help="Number of test examples for evaluation",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="memory-system-grpo",
        help="Wandb project name",
    )
    parser.add_argument(
        "--hf-repo-id",
        type=str,
        default=None,
        help="Hugging Face repository ID (auto-generated if not provided)",
    )
    parser.add_argument(
        "--no-push-to-hub",
        action="store_true",
        help="Disable pushing model to Hugging Face Hub",
    )

    args = parser.parse_args()

    if args.mode == "train":
        # Create Arbor config with specified GPUs
        create_arbor_config(
            inference_gpus=args.inference_gpus, training_gpus=args.training_gpus
        )

        # Run GRPO training
        grpo_train_memory_system(
            num_train=args.num_train,
            num_val=args.num_val,
            num_train_steps=args.num_train_steps,
            model_name=args.model,
            wandb_project=args.wandb_project,
            hf_repo_id=args.hf_repo_id,
            push_to_hub=not args.no_push_to_hub,
        )
    else:
        # Evaluate GRPO model
        evaluate_grpo_model(num_test=args.num_test)
