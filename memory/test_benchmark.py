#!/usr/bin/env python3
"""
Test benchmark to diagnose why we're scoring zero.
"""

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import dspy
from locomo.dataset import load_locomo_dataset
from simple_memory_system import create_simple_memory_system

# Set API key
os.environ["TOGETHER_API_KEY"] = "cc77ff73446653162baed2536ce1348abda599bd27012fd5b1d946a22a6e1c76"

# Configure DSPy
MODEL = "together_ai/deepseek-ai/DeepSeek-R1-0528-tput"
lm = dspy.LM(MODEL, api_key=os.environ["TOGETHER_API_KEY"], max_tokens=1000)
dspy.configure(lm=lm)

print("🔍 Testing Memory System Performance\n")

# Load dataset
print("📚 Loading dataset...")
dataset = load_locomo_dataset("./data/locomo10.json")
examples = dataset.get_examples(limit=5)
print(f"✓ Loaded {len(examples)} examples")

# Create memory system
print("\n🏗️ Creating simple memory system...")
memory_system = create_simple_memory_system()
print(f"✓ Memory system created")

# Build memories from first conversation
print("\n💾 Building memories...")
if examples:
    sample_id = examples[0].sample_id
    for sample in dataset.raw_data:
        if sample["sample_id"] == sample_id:
            memory_system.process_conversation(sample, sample_id)
            break

print(f"✓ Built {len(memory_system.memories)} memories")

# Show some memories
print("\n📝 Sample memories:")
for i, (mid, memory) in enumerate(list(memory_system.memories.items())[:3]):
    print(f"  {i+1}. {memory.content[:80]}...")

# Test questions
print("\n🧪 Testing questions:")
category_map = {1: "multi-hop", 2: "temporal", 3: "single-hop", 4: "unanswerable", 5: "ambiguous"}

correct = 0
total = 0

for i, example in enumerate(examples[:3]):
    print(f"\n--- Test {i+1} ---")
    print(f"Q: {example.question}")
    print(f"Expected: {example.answer}")
    
    # Get answer
    question_category = category_map.get(example.category, "single-hop")
    result = memory_system.answer_question(example.question, question_category)
    print(f"Predicted: {result['answer']}")
    print(f"Reasoning: {result['reasoning']}")
    
    # Simple evaluation
    if result['answer'].lower() != "information not found" and result['answer'].lower() != "unable to determine":
        # Check with LLM judge
        try:
            eval_result = memory_system.evaluate_with_llm_judge(
                example.question, 
                example.answer,
                result['answer']
            )
            is_correct = eval_result["is_correct"]
            print(f"LLM Judge: {'✅ CORRECT' if is_correct else '❌ WRONG'} - {eval_result['reasoning']}")
            
            if is_correct:
                correct += 1
            total += 1
        except Exception as e:
            print(f"Judge error: {e}")
            total += 1
    else:
        print("❌ No answer generated")
        total += 1

# Results
print(f"\n📊 Results: {correct}/{total} = {(correct/total*100 if total > 0 else 0):.1f}%")
print("\n✅ Test complete!")