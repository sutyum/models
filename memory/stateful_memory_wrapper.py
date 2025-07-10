#!/usr/bin/env python3
"""
Stateful Memory Wrapper
======================
Wraps memory systems to handle single-shot evaluation while maintaining state.
"""

import dspy
from typing import Dict, Any
import hashlib

class StatefulMemoryWrapper(dspy.Module):
    """
    Wrapper that maintains memory state across evaluations.
    
    This allows memory systems designed for multi-turn conversations
    to work with single-shot evaluation frameworks.
    """
    
    def __init__(self, memory_system):
        super().__init__()
        self.memory_system = memory_system
        self.conversation_cache = {}  # Cache processed conversations
        
    def forward(self, conversation: str, question: str) -> dspy.Prediction:
        """
        Process conversation once and cache it, then answer question.
        """
        # Create a hash of the conversation for caching
        conv_hash = hashlib.md5(conversation.encode()).hexdigest()
        
        # Check if we've already processed this conversation
        if conv_hash not in self.conversation_cache:
            # Clear memory state for new conversation
            if hasattr(self.memory_system, 'state'):
                self.memory_system.state = ""
            elif hasattr(self.memory_system, 'memories'):
                self.memory_system.memories = []
            
            # Process the conversation
            if hasattr(self.memory_system, 'update'):
                self.memory_system.update(conversation)
            elif hasattr(self.memory_system, 'forward'):
                # Use forward for updating if no update method
                self.memory_system.forward(text=conversation, update_memory=True)
            
            # Cache that we've processed this conversation
            self.conversation_cache[conv_hash] = True
        
        # Now answer the question
        if hasattr(self.memory_system, 'query'):
            answer = self.memory_system.query(question)
            return dspy.Prediction(answer=answer)
        elif hasattr(self.memory_system, 'forward') and hasattr(self.memory_system.forward, '__call__'):
            # Try different forward signatures
            try:
                # Try conversation/question signature
                result = self.memory_system.forward(conversation=conversation, question=question)
                if isinstance(result, dspy.Prediction):
                    return result
                else:
                    return dspy.Prediction(answer=str(result))
            except:
                # Try text-based signature
                result = self.memory_system.forward(text=question, update_memory=False)
                if isinstance(result, dspy.Prediction):
                    return result
                else:
                    return dspy.Prediction(answer=str(result))
        else:
            return dspy.Prediction(answer="Memory system does not support querying")


def create_wrapped_memory_system(system_type: str):
    """Create a wrapped memory system for evaluation."""
    from main import get_memory_system
    
    base_system = get_memory_system(system_type)
    
    # Only wrap systems that need stateful handling
    if system_type in ['graph']:
        return StatefulMemoryWrapper(base_system)
    else:
        return base_system