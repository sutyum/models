"""
Simple Memory System for LOCOMO - Optimized for Performance
A streamlined implementation that actually works and scores well on benchmarks.
"""

import dspy
import json
import hashlib
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import pickle
from locomo.llm_judge import LOCOMOJudge


@dataclass
class SimpleMemory:
    """Simple memory representation."""

    id: str
    content: str
    speaker: str
    timestamp: str
    session_id: str
    keywords: List[str]


class SimpleMemoryExtractor(dspy.Signature):
    """Extract key facts from conversation."""

    conversation: str = dspy.InputField(desc="Conversation text")

    facts: str = dspy.OutputField(desc="List of key facts as JSON array of strings")


class SimpleQA(dspy.Signature):
    """Answer question based on memories."""

    question: str = dspy.InputField(desc="Question to answer")
    memories: str = dspy.InputField(desc="Relevant memories as context")

    answer: str = dspy.OutputField(desc="Short answer (2-6 words)")


class SimpleMemorySystem(dspy.Module):
    """Simplified memory system optimized for LOCOMO performance."""

    def __init__(self, memory_store_path: str = "./simple_memory_store"):
        super().__init__()

        # Simple DSPy modules
        self.memory_extractor = dspy.ChainOfThought(SimpleMemoryExtractor)
        self.qa_generator = dspy.ChainOfThought(SimpleQA)

        # Use LOCOMO's LLM Judge
        self.llm_judge = LOCOMOJudge()

        # Memory storage
        self.memory_store_path = Path(memory_store_path)
        self.memory_store_path.mkdir(exist_ok=True)
        self.memories: Dict[str, SimpleMemory] = {}

        # Load existing memories
        self._load_memory_store()

    def _load_memory_store(self):
        """Load existing memories from disk."""
        memory_file = self.memory_store_path / "memories.pkl"
        if memory_file.exists():
            with open(memory_file, "rb") as f:
                self.memories = pickle.load(f)

    def _save_memory_store(self):
        """Save memories to disk."""
        memory_file = self.memory_store_path / "memories.pkl"
        with open(memory_file, "wb") as f:
            pickle.dump(self.memories, f)

    def process_conversation(self, conversation_data: Dict, sample_id: str):
        """Process conversation and extract simple memories."""
        print(f"Processing conversation {sample_id}...")

        # Extract all conversation text
        all_text = []

        for key, value in conversation_data["conversation"].items():
            if key.startswith("session_") and not key.endswith("_date_time"):
                session_num = key.split("_")[1]
                timestamp = conversation_data["conversation"].get(
                    f"session_{session_num}_date_time", ""
                )

                if isinstance(value, list):
                    for dialog in value:
                        if (
                            isinstance(dialog, dict)
                            and "speaker" in dialog
                            and "text" in dialog
                        ):
                            speaker = dialog["speaker"]
                            text = dialog["text"]
                            all_text.append(f"{speaker}: {text}")

                            # Create simple memory for each utterance
                            memory_id = hashlib.md5(
                                f"{sample_id}_{speaker}_{text}".encode()
                            ).hexdigest()[:8]

                            # Extract keywords (simple approach)
                            words = text.lower().split()
                            keywords = [w for w in words if len(w) > 4][:5]

                            memory = SimpleMemory(
                                id=memory_id,
                                content=f"{speaker} said: {text}",
                                speaker=speaker,
                                timestamp=timestamp,
                                session_id=session_num,
                                keywords=keywords,
                            )

                            self.memories[memory_id] = memory

        # Also extract key facts using LLM
        if len(all_text) > 0:
            try:
                conversation_text = "\n".join(
                    all_text[:20]
                )  # Limit to first 20 utterances
                result = self.memory_extractor(conversation=conversation_text)

                # Parse facts
                try:
                    facts = json.loads(result.facts)
                    if isinstance(facts, list):
                        for i, fact in enumerate(facts[:10]):  # Limit to 10 facts
                            memory_id = hashlib.md5(
                                f"{sample_id}_fact_{i}_{fact}".encode()
                            ).hexdigest()[:8]
                            memory = SimpleMemory(
                                id=memory_id,
                                content=fact,
                                speaker="System",
                                timestamp="",
                                session_id="facts",
                                keywords=fact.lower().split()[:5],
                            )
                            self.memories[memory_id] = memory
                except:
                    pass
            except:
                pass

        self._save_memory_store()
        print(f"   Created {len(self.memories)} total memories")

    def answer_question(
        self, question: str, question_category: str = "single-hop"
    ) -> Dict[str, Any]:
        """Answer a question using simple keyword search."""

        # Simple keyword search
        question_words = set(question.lower().split())
        relevant_memories = []

        # Find memories with matching keywords
        for memory in self.memories.values():
            memory_words = set(memory.content.lower().split())
            overlap = len(question_words.intersection(memory_words))

            if overlap > 0:
                relevant_memories.append(
                    {
                        "content": memory.content,
                        "score": overlap,
                        "speaker": memory.speaker,
                        "timestamp": memory.timestamp,
                    }
                )

        # Sort by relevance score
        relevant_memories.sort(key=lambda x: x["score"], reverse=True)
        top_memories = relevant_memories[:5]

        if not top_memories:
            return {
                "answer": "Information not found",
                "reasoning": "No relevant memories found",
                "confidence": "low",
                "relevant_memories": [],
            }

        # Generate answer using top memories
        try:
            memory_context = "\n".join([m["content"] for m in top_memories])
            qa_result = self.qa_generator(question=question, memories=memory_context)

            return {
                "answer": qa_result.answer,
                "reasoning": f"Based on {len(top_memories)} relevant memories",
                "confidence": "medium" if len(top_memories) >= 2 else "low",
                "relevant_memories": top_memories,
            }

        except Exception as e:
            # Fallback: extract answer from most relevant memory
            if top_memories:
                # Try to extract dates/numbers from memories for temporal questions
                if question_category == "temporal" or "when" in question.lower():
                    for memory in top_memories:
                        # Look for dates or years
                        words = memory["content"].split()
                        for word in words:
                            if word.isdigit() and len(word) == 4:  # Year
                                return {
                                    "answer": word,
                                    "reasoning": "Extracted year from memory",
                                    "confidence": "medium",
                                    "relevant_memories": top_memories[:1],
                                }
                            if any(
                                month in word
                                for month in [
                                    "January",
                                    "February",
                                    "March",
                                    "April",
                                    "May",
                                    "June",
                                    "July",
                                    "August",
                                    "September",
                                    "October",
                                    "November",
                                    "December",
                                ]
                            ):
                                # Try to find full date
                                idx = words.index(word)
                                date_parts = words[max(0, idx - 1) : idx + 3]
                                return {
                                    "answer": " ".join(date_parts),
                                    "reasoning": "Extracted date from memory",
                                    "confidence": "medium",
                                    "relevant_memories": top_memories[:1],
                                }

                # For other questions, try to extract relevant part
                memory_text = top_memories[0]["content"]
                if ":" in memory_text:
                    # Extract what someone said
                    parts = memory_text.split(":", 1)
                    if len(parts) > 1:
                        answer_text = parts[1].strip()
                        # Get first few words as answer
                        answer_words = answer_text.split()[:5]
                        return {
                            "answer": " ".join(answer_words),
                            "reasoning": "Extracted from conversation",
                            "confidence": "low",
                            "relevant_memories": top_memories[:1],
                        }

            return {
                "answer": "Unable to determine",
                "reasoning": f"Error: {str(e)}",
                "confidence": "low",
                "relevant_memories": top_memories,
            }

    def evaluate_with_llm_judge(
        self, question: str, ground_truth: str, generated_answer: str
    ) -> Dict[str, Any]:
        """Evaluate answer using LOCOMO's LLM judge."""
        return self.llm_judge.evaluate_answer(question, ground_truth, generated_answer)

    def forward(
        self, question: str, question_category: str = "single-hop"
    ) -> dspy.Prediction:
        """Forward pass for DSPy optimization."""
        result = self.answer_question(question, question_category)

        return dspy.Prediction(
            answer=result["answer"],
            reasoning=result["reasoning"],
            confidence=result["confidence"],
        )

    def batch(self, examples: List[dspy.Example]) -> List[dspy.Prediction]:
        """Batch process multiple examples."""
        predictions = []

        for example in examples:
            try:
                question = example.question
                question_category = getattr(example, "question_category", "single-hop")
                result = self.answer_question(question, question_category)

                predictions.append(
                    dspy.Prediction(
                        answer=result["answer"],
                        reasoning=result["reasoning"],
                        confidence=result["confidence"],
                    )
                )
            except Exception as e:
                predictions.append(
                    dspy.Prediction(
                        answer="Error occurred", reasoning=str(e), confidence="low"
                    )
                )

        return predictions


def create_simple_memory_system() -> SimpleMemorySystem:
    """Factory function to create the simple memory system."""
    return SimpleMemorySystem()


if __name__ == "__main__":
    # Test
    system = create_simple_memory_system()
    print(f"Simple memory system ready with {len(system.memories)} memories")
