#!/usr/bin/env python3
"""
Minimal DSPy Memory System
=========================
A clean memory system using only DSPy primitives.
"""

import dspy
from typing import List, Dict, Any


class ExtractMemories(dspy.Signature):
    """Extract key memories from conversation."""

    conversation = dspy.InputField(desc="conversation text")
    memories = dspy.OutputField(desc="list of key facts/memories from conversation")


class RetrieveRelevant(dspy.Signature):
    """Retrieve memories relevant to a query."""

    query = dspy.InputField(desc="user query")
    memories = dspy.InputField(desc="all stored memories")
    relevant = dspy.OutputField(desc="memories relevant to the query")


class AnswerWithMemory(dspy.Signature):
    """Answer question using relevant memories."""

    question = dspy.InputField(desc="user question")
    memories = dspy.InputField(desc="relevant memories")
    answer = dspy.OutputField(desc="answer based on memories")


class DSPyMemory(dspy.Module):
    """Pure DSPy memory system."""

    def __init__(self):
        super().__init__()
        self.extract = dspy.ChainOfThought(ExtractMemories)
        self.retrieve = dspy.ChainOfThought(RetrieveRelevant)
        self.answer = dspy.ChainOfThought(AnswerWithMemory)
        self.memories = []

    def update(self, conversation: str):
        """Extract and store memories from conversation."""
        result = self.extract(conversation=conversation)
        if isinstance(result.memories, str):
            # Parse string representation of list
            new_memories = [m.strip() for m in result.memories.split("\n") if m.strip()]
        else:
            new_memories = result.memories
        self.memories.extend(new_memories)

    def query(self, question: str) -> str:
        """Answer question using stored memories."""
        if not self.memories:
            return "No memories available to answer the question."

        # Retrieve relevant memories
        memories_str = "\n".join(self.memories)
        relevant = self.retrieve(query=question, memories=memories_str)

        # Generate answer
        result = self.answer(question=question, memories=relevant.relevant)
        return result.answer

    def forward(self, conversation: str, question: str) -> dspy.Prediction:
        """End-to-end QA with memory."""
        self.update(conversation)
        answer = self.query(question)
        return dspy.Prediction(answer=answer)


# LOCOMO signature for direct QA
class ConversationQA(dspy.Signature):
    """Answer questions based on conversation history."""

    conversation = dspy.InputField(desc="conversation history")
    question = dspy.InputField(desc="question about the conversation")
    answer = dspy.OutputField(desc="answer based on conversation")


# Example usage for LOCOMO
class LOCOMOMemory(dspy.Module):
    """LOCOMO-specific memory module."""

    def __init__(self):
        super().__init__()
        self.qa = dspy.ChainOfThought(ConversationQA)

    def forward(self, conversation, question):
        return self.qa(conversation=conversation, question=question)
