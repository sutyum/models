#!/usr/bin/env python3
"""
Test script to demonstrate ReACT memory search without embeddings.
"""
import os
import sys
from pathlib import Path

# Add the locomo package to the path
sys.path.append(str(Path(__file__).parent))

import dspy
import json
from locomo.sota_memory_system import create_sota_memory_system

def test_react_memory_system():
    """Test the ReACT-based memory system."""
    print("🧪 Testing ReACT Memory System (No Embeddings)")
    print("=" * 50)
    
    # Setup DSPy
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        print("❌ TOGETHER_API_KEY not found!")
        print("   Using mock data for demonstration")
    else:
        MODEL = "together_ai/deepseek-ai/DeepSeek-R1-0528-tput"
        lm = dspy.LM(MODEL, api_key=api_key)
        dspy.configure(lm=lm)
        print(f"✅ Configured with {MODEL}")
    
    # Create memory system
    system = create_sota_memory_system()
    print("✅ Created ReACT memory system")
    
    # Create sample conversation data
    conversation_data = {
        "conversation": {
            "session_1": [
                {"speaker": "Alice", "text": "I love hiking in the mountains during summer"},
                {"speaker": "Bob", "text": "That sounds amazing! Which mountains do you prefer?"},
                {"speaker": "Alice", "text": "I usually go to the Rocky Mountains. The views are breathtaking!"},
                {"speaker": "Bob", "text": "I've heard the trails there are challenging but rewarding."},
            ],
            "session_1_date_time": "2023-05-01",
            "session_2": [
                {"speaker": "Alice", "text": "Last week I tried a new recipe for chocolate cake"},
                {"speaker": "Bob", "text": "How did it turn out? I love baking too!"},
                {"speaker": "Alice", "text": "It was delicious! I added extra cocoa for a richer flavor"},
                {"speaker": "Bob", "text": "You should share the recipe sometime!"},
            ],
            "session_2_date_time": "2023-05-15",
        }
    }
    
    # Process conversation
    print("\n📚 Processing conversation...")
    system.process_conversation(conversation_data, "test_conv_1")
    print(f"✅ Created {len(system.memories)} memories")
    
    # Show memory summary
    print("\n📋 Memory Summary:")
    print(system._generate_memory_summary())
    
    # Test questions with ReACT search
    test_questions = [
        ("What outdoor activities does Alice enjoy?", "single-hop"),
        ("When did Alice talk about baking and what did she make?", "multi-hop"),
        ("What are Alice's hobbies?", "open-domain"),
    ]
    
    print("\n🔍 Testing ReACT Memory Search:")
    for question, category in test_questions:
        print(f"\n{'=' * 50}")
        print(f"Question: {question}")
        print(f"Category: {category}")
        
        # Answer question
        result = system.answer_question(question, category)
        
        print(f"\n🤔 ReACT Search Process:")
        for step in result.get('search_history', []):
            print(f"\nIteration {step['iteration']}:")
            print(f"  Thought: {step['thought']}")
            print(f"  Action: {step['action']}")
        
        print(f"\n💡 Answer: {result['answer']}")
        print(f"🧠 Reasoning: {result['reasoning']}")
        print(f"📊 Confidence: {result['confidence']}")
        print(f"📝 Found {len(result['relevant_memories'])} relevant memories")
    
    print("\n✅ ReACT memory system test complete!")
    print("🎯 Key Features:")
    print("  - No embeddings or vector search required")
    print("  - Iterative reasoning through memories")
    print("  - Plain text search with semantic understanding")
    print("  - Transparent search process via ReACT")


if __name__ == "__main__":
    test_react_memory_system()