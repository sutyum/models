import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple


def split_locomo_data(
    input_file: str = "data/locomo10.json",
    output_dir: str = "data",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
    stratify_by_category: bool = True
) -> Tuple[str, str, str]:
    """
    Split LOCOMO dataset into train, validation, and test sets.
    
    Args:
        input_file: Path to the input JSON file
        output_dir: Directory to save the split files
        train_ratio: Ratio of data for training set
        val_ratio: Ratio of data for validation set
        test_ratio: Ratio of data for test set
        random_seed: Random seed for reproducibility
        stratify_by_category: Whether to stratify split by question category
    
    Returns:
        Tuple of (train_file, val_file, test_file) paths
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"
    
    random.seed(random_seed)
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    all_examples = []
    for conversation in data:
        for qa_pair in conversation['qa']:
            answer = qa_pair.get('answer') or qa_pair.get('adversarial_answer', '')
            example = {
                'id': f"{conversation.get('id', len(all_examples))}_{len(all_examples)}",
                'conversation': conversation,
                'question': qa_pair['question'],
                'answer': answer,
                'evidence': qa_pair.get('evidence', []),
                'category': qa_pair.get('category', 0)
            }
            all_examples.append(example)
    
    print(f"Total examples: {len(all_examples)}")
    
    if stratify_by_category:
        category_buckets = {}
        for example in all_examples:
            category = example.get('category', 0)
            if category not in category_buckets:
                category_buckets[category] = []
            category_buckets[category].append(example)
        
        print("\nExamples per category:")
        for cat, bucket in category_buckets.items():
            print(f"  Category {cat}: {len(bucket)} examples")
        
        train_examples = []
        val_examples = []
        test_examples = []
        
        for category, bucket in category_buckets.items():
            random.shuffle(bucket)
            
            n_total = len(bucket)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)
            
            train_examples.extend(bucket[:n_train])
            val_examples.extend(bucket[n_train:n_train + n_val])
            test_examples.extend(bucket[n_train + n_val:])
    else:
        random.shuffle(all_examples)
        
        n_total = len(all_examples)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_examples = all_examples[:n_train]
        val_examples = all_examples[n_train:n_train + n_val]
        test_examples = all_examples[n_train + n_val:]
    
    random.shuffle(train_examples)
    random.shuffle(val_examples)
    random.shuffle(test_examples)
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_examples)} ({len(train_examples)/len(all_examples):.1%})")
    print(f"  Val: {len(val_examples)} ({len(val_examples)/len(all_examples):.1%})")
    print(f"  Test: {len(test_examples)} ({len(test_examples)/len(all_examples):.1%})")
    
    train_file = output_path / "locomo_train.json"
    val_file = output_path / "locomo_val.json"
    test_file = output_path / "locomo_test.json"
    
    train_data = {
        "version": "1.0",
        "split": "train",
        "examples": train_examples
    }
    with open(train_file, 'w') as f:
        json.dump(train_data, f, indent=2)
    
    val_data = {
        "version": "1.0",
        "split": "validation",
        "examples": val_examples
    }
    with open(val_file, 'w') as f:
        json.dump(val_data, f, indent=2)
    
    test_data = {
        "version": "1.0",
        "split": "test",
        "examples": test_examples
    }
    with open(test_file, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"\nFiles saved:")
    print(f"  Train: {train_file}")
    print(f"  Val: {val_file}")
    print(f"  Test: {test_file}")
    
    return str(train_file), str(val_file), str(test_file)


def verify_split(train_file: str, val_file: str, test_file: str):
    """Verify that the data split was done correctly."""
    
    print("\nVerifying data split...")
    
    with open(train_file, 'r') as f:
        train_data = json.load(f)
    with open(val_file, 'r') as f:
        val_data = json.load(f)
    with open(test_file, 'r') as f:
        test_data = json.load(f)
    
    train_ids = {ex['id'] for ex in train_data['examples']}
    val_ids = {ex['id'] for ex in val_data['examples']}
    test_ids = {ex['id'] for ex in test_data['examples']}
    
    overlap_train_val = train_ids & val_ids
    overlap_train_test = train_ids & test_ids
    overlap_val_test = val_ids & test_ids
    
    print(f"  Train-Val overlap: {len(overlap_train_val)} examples")
    print(f"  Train-Test overlap: {len(overlap_train_test)} examples")
    print(f"  Val-Test overlap: {len(overlap_val_test)} examples")
    
    if overlap_train_val or overlap_train_test or overlap_val_test:
        print("  WARNING: Data leakage detected!")
    else:
        print("  ✓ No data leakage - splits are disjoint")
    
    all_categories = {}
    for split_name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        categories = {}
        for ex in data['examples']:
            cat = ex.get('category', 0)
            categories[cat] = categories.get(cat, 0) + 1
        all_categories[split_name] = categories
    
    print("\nCategory distribution:")
    for split_name, categories in all_categories.items():
        print(f"  {split_name}:")
        for cat, count in sorted(categories.items()):
            print(f"    Category {cat}: {count}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Split LOCOMO dataset into train/val/test")
    parser.add_argument("--input", type=str, default="data/locomo10.json",
                        help="Input JSON file")
    parser.add_argument("--output-dir", type=str, default="data",
                        help="Output directory for split files")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="Training set ratio")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Validation set ratio")
    parser.add_argument("--test-ratio", type=float, default=0.1,
                        help="Test set ratio")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--no-stratify", action="store_true",
                        help="Disable stratification by category")
    parser.add_argument("--verify", action="store_true",
                        help="Verify the split after creation")
    
    args = parser.parse_args()
    
    train_file, val_file, test_file = split_locomo_data(
        input_file=args.input,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
        stratify_by_category=not args.no_stratify
    )
    
    if args.verify:
        verify_split(train_file, val_file, test_file)