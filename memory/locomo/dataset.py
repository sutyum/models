"""
DSPy dataset wrapper for LOCOMO conversational QA benchmark.
"""
import json
import random
from typing import List, Dict, Tuple, Optional
import dspy
from pathlib import Path


class LocomoDataset:
    """Dataset class for LOCOMO conversational QA data."""
    
    def __init__(self, data_path: str):
        """
        Initialize LOCOMO dataset.
        
        Args:
            data_path: Path to the LOCOMO JSON data file
        """
        self.data_path = data_path
        self.raw_data = self._load_data()
        self.examples = self._create_examples()
    
    def _load_data(self) -> List[Dict]:
        """Load raw LOCOMO data from JSON file."""
        with open(self.data_path, 'r') as f:
            return json.load(f)
    
    def _create_examples(self) -> List[dspy.Example]:
        """Convert LOCOMO data to DSPy Examples."""
        examples = []
        
        for sample in self.raw_data:
            # Extract conversation context
            conversation_text = self._format_conversation(sample["conversation"])
            
            # Create examples for each QA pair in the sample
            for qa in sample["qa"]:
                # Ensure answer is a string
                answer = qa.get("answer", qa.get("adversarial_answer", ""))
                if answer is None:
                    answer = ""
                answer = str(answer)
                
                example = dspy.Example(
                    conversation=conversation_text,
                    question=qa["question"],
                    answer=answer,
                    category=qa["category"],
                    evidence=qa.get("evidence", []),
                    sample_id=sample["sample_id"],
                    qa_id=qa.get("id", "")
                ).with_inputs("conversation", "question")
                
                examples.append(example)
        
        return examples
    
    def _format_conversation(self, conversation_data: Dict) -> str:
        """Format conversation data into a readable text."""
        conversation_text = ""
        
        # Get all session keys and sort them
        session_keys = [k for k in conversation_data.keys() if k.startswith("session_") and not k.endswith("_date_time")]
        session_keys.sort(key=lambda x: int(x.split("_")[1]))
        
        for session_key in session_keys:
            session_num = session_key.split("_")[1]
            date_key = f"session_{session_num}_date_time"
            
            if session_key in conversation_data and conversation_data[session_key]:
                # Add date if available
                if date_key in conversation_data:
                    conversation_text += f"\nDATE: {conversation_data[date_key]}\n"
                
                conversation_text += "CONVERSATION:\n"
                
                for dialog in conversation_data[session_key]:
                    speaker = dialog["speaker"]
                    text = dialog["text"]
                    conversation_text += f'{speaker}: "{text}"\n'
                    
                    # Add image caption if available
                    if "blip_caption" in dialog and dialog["blip_caption"]:
                        conversation_text += f"[shared image: {dialog['blip_caption']}]\n"
                
                conversation_text += "\n"
        
        return conversation_text.strip()
    
    def get_examples(self, limit: Optional[int] = None) -> List[dspy.Example]:
        """Get DSPy examples with optional limit."""
        if limit is not None:
            return self.examples[:limit]
        return self.examples
    
    def split_data(self, train_ratio: float = 0.6, val_ratio: float = 0.2, test_ratio: float = 0.2, 
                   random_seed: int = 42) -> Tuple[List[dspy.Example], List[dspy.Example], List[dspy.Example]]:
        """
        Split data into train/validation/test sets.
        
        Args:
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set  
            test_ratio: Proportion for test set
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_set, val_set, test_set)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        random.seed(random_seed)
        examples = self.examples.copy()
        random.shuffle(examples)
        
        n_total = len(examples)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_set = examples[:n_train]
        val_set = examples[n_train:n_train + n_val]
        test_set = examples[n_train + n_val:]
        
        return train_set, val_set, test_set
    
    def get_category_split(self, categories: List[int], limit_per_category: Optional[int] = None) -> List[dspy.Example]:
        """
        Get examples filtered by specific categories.
        
        Args:
            categories: List of category numbers to include
            limit_per_category: Maximum examples per category
            
        Returns:
            Filtered list of examples
        """
        category_examples = {cat: [] for cat in categories}
        
        for example in self.examples:
            if example.category in categories:
                category_examples[example.category].append(example)
        
        result = []
        for cat in categories:
            cat_examples = category_examples[cat]
            if limit_per_category is not None:
                cat_examples = cat_examples[:limit_per_category]
            result.extend(cat_examples)
        
        return result
    
    def get_stats(self) -> Dict:
        """Get dataset statistics."""
        category_counts = {}
        for example in self.examples:
            cat = example.category
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        return {
            "total_examples": len(self.examples),
            "total_samples": len(self.raw_data),
            "category_distribution": category_counts,
            "avg_qa_per_sample": len(self.examples) / len(self.raw_data) if self.raw_data else 0
        }


def load_locomo_dataset(data_path: str = "./data/locomo10.json") -> List[dspy.Example]:
    """
    Convenience function to load LOCOMO dataset.
    
    Args:
        data_path: Path to LOCOMO data file
        
    Returns:
        List of dspy.Example instances
    """
    # Check if it's the new format (with 'version' key)
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and 'version' in data:
        # New format with version and examples
        examples = []
        for item in data['examples']:
            # Each example has direct question/answer fields
            if 'question' in item and 'answer' in item:
                # For memory QA, we need current_state and new_information
                # Since we don't have conversation context, use empty state
                example = dspy.Example(
                    current_state="No previous context available.",
                    new_information=f"Question: {item['question']}",
                    question=item['question'],
                    answer=item['answer'],
                    category=item.get('category', 1),
                    evidence=item.get('evidence', []),
                    id=item['id']
                ).with_inputs("current_state", "new_information")
                examples.append(example)
        return examples
    else:
        # Old format
        dataset = LocomoDataset(data_path)
        return dataset.get_examples()


if __name__ == "__main__":
    # Example usage
    dataset = load_locomo_dataset()
    
    print("Dataset Statistics:")
    stats = dataset.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\nFirst example:")
    first_example = dataset.get_examples(limit=1)[0]
    print(f"Question: {first_example.question}")
    print(f"Answer: {first_example.answer}")
    print(f"Category: {first_example.category}")
    print(f"Conversation (first 200 chars): {first_example.conversation[:200]}...")
    
    # Test data splitting
    train, val, test = dataset.split_data()
    print(f"\nData split: Train={len(train)}, Val={len(val)}, Test={len(test)}")