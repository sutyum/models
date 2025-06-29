import dspy
import json
import os
from dspy.teleprompt import SIMBA
from memory_system import MemorySystem
from locomo.evaluate import LOCOMOMetric


class MemoryQASignature(dspy.Signature):
    conversation = dspy.InputField(
        desc="Full conversation history with multiple sessions"
    )
    question = dspy.InputField(desc="Question to answer based on the conversation")
    answer = dspy.OutputField(
        desc="Answer to the question based on conversation history"
    )


class OptimizedMemoryQA(dspy.Module):
    def __init__(self, memory_system: MemorySystem):
        super().__init__()
        self.memory = memory_system
        self.qa = dspy.ChainOfThought(MemoryQASignature)

    def forward(self, conversation: str, question: str) -> dspy.Prediction:
        self.memory.update(conversation)

        memory_context = self.memory.query(question)

        enriched_conversation = f"{conversation}\n\nMemory Context:\n{memory_context}"

        return self.qa(conversation=enriched_conversation, question=question)


def create_memory_system_qa(persist: bool = False) -> OptimizedMemoryQA:
    memory = MemorySystem(persist=persist, resource_limit=20000)
    return OptimizedMemoryQA(memory)


def load_split_data(file_path: str) -> list[dict]:
    """Load data from our split JSON files."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['examples']


def format_conversation(conversation_data: dict) -> str:
    """Format conversation data into readable text (same as LocomoDataset)."""
    conversation_text = ""
    
    session_keys = [k for k in conversation_data.keys() if k.startswith("session_") and not k.endswith("_date_time")]
    session_keys.sort(key=lambda x: int(x.split("_")[1]))
    
    for session_key in session_keys:
        session_num = session_key.split("_")[1]
        date_key = f"session_{session_num}_date_time"
        
        if session_key in conversation_data and conversation_data[session_key]:
            if date_key in conversation_data:
                conversation_text += f"\nDATE: {conversation_data[date_key]}\n"
            
            conversation_text += "CONVERSATION:\n"
            
            for dialog in conversation_data[session_key]:
                speaker = dialog["speaker"]
                text = dialog["text"]
                conversation_text += f'{speaker}: "{text}"\n'
                
                if "blip_caption" in dialog and dialog["blip_caption"]:
                    conversation_text += f"[shared image: {dialog['blip_caption']}]\n"
            
            conversation_text += "\n"
    
    return conversation_text.strip()


def prepare_training_data(file_path: str, num_examples: int = 100) -> list[dspy.Example]:
    """Load and prepare training data from split files."""
    examples_data = load_split_data(file_path)
    examples = []

    for idx, example_data in enumerate(examples_data[:num_examples]):
        # The conversation is nested under 'conversation' key
        conversation_text = format_conversation(example_data['conversation']['conversation'])
        
        qa_example = dspy.Example(
            conversation=conversation_text,
            question=example_data['question'],
            answer=str(example_data['answer']),
            category=example_data.get('category', 0)
        ).with_inputs("conversation", "question")

        examples.append(qa_example)

    return examples


def optimize_memory_system(
    train_file: str = "data/locomo_train.json",
    num_examples: int = 50,
    num_threads: int = 4,
    max_demos: int = 5,
):
    """
    Optimize the memory system using DSPy's SIMBA optimizer.
    
    This function:
    1. Loads training data from split LOCOMO files
    2. Initializes the memory system with QA capabilities
    3. Uses SIMBA to optimize prompts and few-shot examples (much faster than MIPROv2)
    4. Saves the optimized program to 'optimized_memory_qa.json'
    
    Args:
        train_file: Path to training data JSON file
        num_examples: Number of examples to use for optimization
        num_threads: Number of threads for parallel optimization
        max_demos: Maximum demonstrations per predictor
    """
    # Configure DSPy with language model
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError("TOGETHER_API_KEY environment variable not set")
    
    MODEL = "together_ai/deepseek-ai/DeepSeek-R1-0528-tput"
    lm = dspy.LM(MODEL, api_key=api_key, max_tokens=20_000)
    dspy.configure(lm=lm)
    
    print("Loading LOCOMO dataset...")
    train_data = prepare_training_data(train_file, num_examples)
    
    print(f"Prepared {len(train_data)} examples for optimization")

    print("Initializing memory system...")
    memory_qa = create_memory_system_qa(persist=True)

    metric = LOCOMOMetric()

    print("Starting optimization with SIMBA (much faster than MIPROv2)...")
    optimizer = SIMBA(
        metric=metric,
        max_demos=max_demos,
        num_threads=num_threads,
        max_steps=6,  # Reduced for faster optimization
        num_candidates=4,  # Reduced for faster optimization
        bsize=min(len(train_data), 8),  # Smaller batch size for speed
    )
    
    optimized_program = optimizer.compile(
        memory_qa,
        trainset=train_data,
    )

    optimized_program.save("optimized_memory_qa.json")
    print("Optimization complete! Saved to optimized_memory_qa.json")

    return optimized_program


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Optimize memory system for LOCOMO")
    parser.add_argument(
        "--num-examples", type=int, default=50, help="Number of examples for optimization"
    )
    parser.add_argument(
        "--num-threads", type=int, default=4, help="Number of threads for optimization"
    )
    parser.add_argument(
        "--max-demos", type=int, default=5,
        help="Maximum demonstrations per predictor"
    )

    args = parser.parse_args()

    optimize_memory_system(
        num_examples=args.num_examples,
        num_threads=args.num_threads,
        max_demos=args.max_demos
    )
