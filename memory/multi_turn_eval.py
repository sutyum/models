#!/usr/bin/env python3
"""
Multi-turn Memory Evaluation for LOCOMO
======================================
Proper evaluation that processes conversations turn-by-turn to build up memory state.
"""

import dspy
from typing import List, Dict, Any
import re
from locomo.dataset import load_locomo_dataset
from locomo.evaluate import LOCOMOMetric
import logging

# Suppress LiteLLM's chatty INFO logs
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

def parse_conversation_turns(conversation_text: str) -> List[str]:
    """Parse conversation text into individual turns."""
    turns = []
    
    # Split by DATE sections first
    date_sections = re.split(r'\n\nDATE:', conversation_text)
    
    for i, section in enumerate(date_sections):
        if i > 0:  # Add DATE back (except for first section)
            section = 'DATE:' + section
            
        # Extract individual speaker turns
        lines = section.split('\n')
        current_turn = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if it's a speaker line (Name: "text")
            if re.match(r'^[A-Za-z]+:\s*".*"$', line):
                if current_turn:
                    turns.append('\n'.join(current_turn))
                    current_turn = []
                current_turn.append(line)
            elif line.startswith('[shared image:'):
                # Add image captions to current turn
                if current_turn:
                    current_turn.append(line)
            elif line.startswith('DATE:') or line.startswith('CONVERSATION:'):
                if current_turn:
                    turns.append('\n'.join(current_turn))
                    current_turn = []
                current_turn.append(line)
            else:
                # Continuation of previous line
                if current_turn:
                    current_turn.append(line)
        
        if current_turn:
            turns.append('\n'.join(current_turn))
    
    return [turn for turn in turns if turn.strip()]

def evaluate_memory_system_multiturn(memory_system, examples: List[dspy.Example], 
                                   metric=None, max_examples: int = None) -> Dict[str, Any]:
    """
    Evaluate memory system with proper multi-turn conversation processing.
    
    Args:
        memory_system: Memory system to evaluate
        examples: List of LOCOMO examples
        metric: Evaluation metric (defaults to LOCOMOMetric)
        max_examples: Maximum number of examples to evaluate
        
    Returns:
        Evaluation results
    """
    if metric is None:
        metric = LOCOMOMetric()
    
    if max_examples:
        examples = examples[:max_examples]
    
    results = []
    total_score = 0.0
    
    print(f"Evaluating {len(examples)} examples with multi-turn memory...")
    
    for i, example in enumerate(examples):
        print(f"Example {i+1}/{len(examples)}: Processing conversation...")
        
        # Clear memory state for each new conversation
        if hasattr(memory_system, 'state'):
            memory_system.state = ""
        elif hasattr(memory_system, 'memories'):
            memory_system.memories = []
            
        # Parse conversation into turns
        turns = parse_conversation_turns(example.conversation)
        
        # Process conversation turn by turn
        for j, turn in enumerate(turns):
            if turn.strip():
                print(f"  Turn {j+1}: {turn[:100]}...")
                # Update memory with each turn
                if hasattr(memory_system, 'update'):
                    memory_system.update(turn)
                elif hasattr(memory_system, 'forward'):
                    # Use forward method for updating
                    memory_system.forward(text=turn, update_memory=True)
        
        # Ask the question after processing all turns
        print(f"  Question: {example.question}")
        
        # Get answer from memory system
        if hasattr(memory_system, 'query'):
            answer = memory_system.query(example.question)
            pred = dspy.Prediction(answer=answer)
        else:
            # Use forward method for querying
            pred = memory_system.forward(text=example.question, update_memory=False)
            if isinstance(pred, str):
                pred = dspy.Prediction(answer=pred)
        
        print(f"  Answer: {pred.answer}")
        print(f"  Expected: {example.answer}")
        
        # Evaluate the answer
        score = metric(example, pred)
        total_score += score
        
        results.append({
            'example_id': i,
            'question': example.question,
            'predicted_answer': pred.answer,
            'ground_truth': example.answer,
            'score': score,
            'category': example.category,
            'num_turns': len(turns)
        })
        
        print(f"  Score: {score:.1f}")
        print(f"  Running average: {total_score/(i+1):.1%}")
        print()
    
    overall_score = total_score / len(examples) if examples else 0.0
    
    return {
        'overall_score': overall_score,
        'total_examples': len(examples),
        'detailed_results': results
    }

if __name__ == "__main__":
    import os
    from graph_memory import MemorySystem
    
    # Configure LM
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: Set GEMINI_API_KEY environment variable")
        exit(1)
        
    MODEL = "gemini/gemini-2.5-flash"
    dspy.configure(lm=dspy.LM(MODEL, api_key=api_key))
    
    # Load test data
    examples = load_locomo_dataset("data/locomo_test.json")[:2]  # Test with 2 examples
    
    # Test GraphMemory with multi-turn evaluation
    print("Testing GraphMemory with multi-turn evaluation...")
    memory_system = MemorySystem()
    results = evaluate_memory_system_multiturn(memory_system, examples)
    
    print(f"\nFinal Results:")
    print(f"Overall Score: {results['overall_score']:.1%}")
    print(f"Total Examples: {results['total_examples']}")