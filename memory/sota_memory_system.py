"""
SOTA Memory System for LOCOMO - DSPy Implementation
Inspired by Mem0 paper architecture with LLM-as-Judge evaluation.
"""

import dspy
import json
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import pickle
from pathlib import Path

import mlflow


@dataclass
class Memory:
    """Structured memory representation."""

    id: str
    content: str
    speaker: str
    timestamp: str
    session_id: str
    importance_score: float
    keywords: List[str]
    memory_type: str  # 'fact', 'preference', 'event', 'relationship'
    created_at: str
    updated_at: str


class MemoryExtractorSignature(dspy.Signature):
    """Extract salient memories from conversation exchanges."""

    conversation_summary: str = dspy.InputField(
        desc="High-level summary of the conversation context and themes"
    )
    recent_messages: str = dspy.InputField(
        desc="Recent message history for context (last 10 messages)"
    )
    current_exchange: str = dspy.InputField(
        desc="Current message pair (user-assistant exchange) to extract memories from"
    )
    speaker_context: str = dspy.InputField(
        desc="Information about the speaker for personalized memory extraction"
    )

    extracted_memories: str = dspy.OutputField(
        desc="JSON list of extracted memories with fields: content, speaker, importance_score (0-1), keywords, memory_type. Focus on facts, preferences, events, and relationships that could be referenced later."
    )


class MemoryUpdateSignature(dspy.Signature):
    """Determine update operations for extracted memories."""

    candidate_memory: str = dspy.InputField(desc="New memory candidate to be processed")
    similar_memories: str = dspy.InputField(
        desc="JSON list of existing similar memories from the memory store"
    )

    operation: str = dspy.OutputField(
        desc="Memory operation: ADD (new memory), UPDATE (enhance existing), DELETE (remove contradicted), NOOP (no action needed)"
    )
    reasoning: str = dspy.OutputField(
        desc="Brief explanation of why this operation was chosen"
    )
    updated_content: str = dspy.OutputField(
        desc="If UPDATE operation, provide the enhanced memory content. Otherwise, return the original content."
    )


class ReACTMemorySearchSignature(dspy.Signature):
    """
    ReACT-style iterative memory search without embeddings.
    Think step-by-step through memories to find relevant ones.
    """

    question: str = dspy.InputField(
        desc="Question that needs to be answered using conversation memories"
    )
    all_memories_summary: str = dspy.InputField(
        desc="Summary of all available memories organized by type and speaker"
    )
    question_category: str = dspy.InputField(
        desc="Type of question: single-hop, multi-hop, temporal, or open-domain"
    )
    search_iteration: int = dspy.InputField(
        desc="Current search iteration number (1-3)"
    )
    previous_thoughts: str = dspy.InputField(
        desc="Previous reasoning steps and findings from earlier iterations"
    )

    thought: str = dspy.OutputField(
        desc="Current reasoning about which memories might be relevant and why"
    )
    action: str = dspy.OutputField(
        desc="Specific search action: SEARCH_KEYWORD(keywords), SEARCH_SPEAKER(speaker), SEARCH_TIME(timeframe), SEARCH_TYPE(memory_type), or DONE"
    )
    search_criteria: str = dspy.OutputField(
        desc="JSON with search parameters based on the action (e.g., {'keywords': ['hiking', 'mountains']})"
    )
    found_memories: str = dspy.OutputField(
        desc="JSON list of memory IDs that match the current search criteria"
    )
    continue_search: str = dspy.OutputField(
        desc="YES if more search iterations needed, NO if sufficient memories found"
    )


class MemoryRankingSignature(dspy.Signature):
    """Rank and filter memories found through ReACT search."""

    question: str = dspy.InputField(desc="Original question to be answered")
    found_memories: str = dspy.InputField(
        desc="JSON list of all memories found through ReACT search iterations"
    )
    search_history: str = dspy.InputField(
        desc="History of search thoughts and actions taken"
    )
    question_category: str = dspy.InputField(
        desc="Type of question: single-hop, multi-hop, temporal, or open-domain"
    )

    reasoning: str = dspy.OutputField(
        desc="Explanation of how memories were ranked and why certain ones are most relevant"
    )
    relevant_memories: str = dspy.OutputField(
        desc="JSON list of top memories ranked by relevance to the question"
    )


class MemoryBasedQASignature(dspy.Signature):
    """Generate answers using retrieved memories with temporal and contextual awareness."""

    question: str = dspy.InputField(
        desc="Question to be answered based on conversation memories"
    )
    relevant_memories: str = dspy.InputField(
        desc="JSON list of relevant memories that should be used to answer the question"
    )
    question_category: str = dspy.InputField(
        desc="Question type: single-hop, multi-hop, temporal, or open-domain"
    )

    reasoning: str = dspy.OutputField(
        desc="Step-by-step reasoning using the memories, especially for temporal calculations and multi-hop connections"
    )
    answer: str = dspy.OutputField(
        desc="Concise, accurate answer (2-6 words) based solely on the provided memories. For temporal questions, provide specific dates/times."
    )
    confidence: str = dspy.OutputField(
        desc="Confidence level: high, medium, low based on memory clarity and completeness"
    )


class LLMJudgeSignature(dspy.Signature):
    """LLM-as-Judge evaluation following Mem0 methodology."""

    question: str = dspy.InputField(desc="Original question that was asked")
    ground_truth: str = dspy.InputField(desc="Gold standard answer for comparison")
    generated_answer: str = dspy.InputField(
        desc="Answer generated by the memory system"
    )
    question_category: str = dspy.InputField(
        desc="Category of question (1-5) for category-specific evaluation"
    )

    reasoning: str = dspy.OutputField(
        desc="One sentence explanation of why the answer is correct or incorrect"
    )
    judgment: str = dspy.OutputField(
        desc="CORRECT or WRONG - be generous if the generated answer touches on the same topic as ground truth"
    )


class SOTAMemorySystem(dspy.Module):
    """
    State-of-the-art memory system for LOCOMO benchmark.
    Implements Mem0-inspired architecture with DSPy optimization.
    """

    def __init__(self, memory_store_path: str = "./sota_memory_store"):
        super().__init__()

        # DSPy modules
        self.memory_extractor = dspy.ChainOfThought(MemoryExtractorSignature)
        self.memory_updater = dspy.ChainOfThought(MemoryUpdateSignature)
        self.memory_searcher = dspy.ChainOfThought(ReACTMemorySearchSignature)
        self.memory_ranker = dspy.ChainOfThought(MemoryRankingSignature)
        self.qa_generator = dspy.ChainOfThought(MemoryBasedQASignature)
        self.llm_judge = dspy.ChainOfThought(LLMJudgeSignature)

        # Memory storage
        self.memory_store_path = Path(memory_store_path)
        self.memory_store_path.mkdir(exist_ok=True)
        self.memories: Dict[str, Memory] = {}
        self.conversation_summaries: Dict[str, str] = {}

        # Load existing memories if available
        self._load_memory_store()

    def _load_memory_store(self):
        """Load existing memories from disk."""
        memory_file = self.memory_store_path / "memories.pkl"
        if memory_file.exists():
            with open(memory_file, "rb") as f:
                data = pickle.load(f)
                self.memories = data.get("memories", {})
                self.conversation_summaries = data.get("summaries", {})

    def _save_memory_store(self):
        """Save memories to disk."""
        memory_file = self.memory_store_path / "memories.pkl"
        data = {
            "memories": self.memories,
            "summaries": self.conversation_summaries,
        }
        with open(memory_file, "wb") as f:
            pickle.dump(data, f)

    def _generate_memory_summary(self) -> str:
        """Generate a summary of all memories for ReACT search."""
        summary_parts = []

        # Group memories by type
        by_type = {}
        for memory in self.memories.values():
            if memory.memory_type not in by_type:
                by_type[memory.memory_type] = []
            by_type[memory.memory_type].append(memory)

        # Group memories by speaker
        by_speaker = {}
        for memory in self.memories.values():
            if memory.speaker not in by_speaker:
                by_speaker[memory.speaker] = []
            by_speaker[memory.speaker].append(memory)

        summary_parts.append(f"Total memories: {len(self.memories)}")
        summary_parts.append(f"Memory types: {', '.join(by_type.keys())}")
        summary_parts.append(f"Speakers: {', '.join(by_speaker.keys())}")

        # Add brief overview of each type
        for mem_type, mems in by_type.items():
            summary_parts.append(f"\n{mem_type.capitalize()} memories ({len(mems)}):")
            # Show first few examples
            for mem in mems[:3]:
                summary_parts.append(f"  - {mem.speaker}: {mem.content[:60]}...")

        return "\n".join(summary_parts)

    def _search_memories_by_criteria(
        self, search_criteria: Dict[str, Any]
    ) -> List[Memory]:
        """Search memories based on criteria from ReACT action."""
        results = []

        for memory in self.memories.values():
            match = False

            # Keyword search
            if "keywords" in search_criteria:
                keywords = search_criteria["keywords"]
                content_lower = memory.content.lower()
                if any(kw.lower() in content_lower for kw in keywords):
                    match = True

            # Speaker search
            if "speaker" in search_criteria:
                if memory.speaker.lower() == search_criteria["speaker"].lower():
                    match = True

            # Time search
            if "timeframe" in search_criteria:
                if search_criteria["timeframe"] in memory.timestamp:
                    match = True

            # Memory type search
            if "memory_type" in search_criteria:
                if memory.memory_type == search_criteria["memory_type"]:
                    match = True

            if match:
                results.append(memory)

        return results

    def process_conversation(self, conversation_data: Dict, sample_id: str):
        """Process a complete conversation and extract memories."""
        print(f"Processing conversation {sample_id}...")

        initial_memory_count = len(self.memories)

        # Extract conversation text and build session structure
        sessions = self._extract_sessions(conversation_data["conversation"])

        # Process each session incrementally
        session_count = 0
        message_pairs_processed = 0

        for session_id, messages in sessions.items():
            session_count += 1
            session_summary = self._generate_session_summary(messages, session_id)
            self.conversation_summaries[f"{sample_id}_{session_id}"] = session_summary

            # Process message pairs within session
            for i in range(0, len(messages) - 1, 2):
                if i + 1 < len(messages):
                    self._process_message_pair(
                        messages[i],
                        messages[i + 1],
                        session_id,
                        sample_id,
                        session_summary,
                    )
                    message_pairs_processed += 1

        # Log memory building metrics to MLflow if available
        if MLFLOW_AVAILABLE:
            try:
                memories_created = len(self.memories) - initial_memory_count
                mlflow.log_metric(f"memories_created_{sample_id}", memories_created)
                mlflow.log_metric(f"sessions_processed_{sample_id}", session_count)
                mlflow.log_metric(
                    f"message_pairs_processed_{sample_id}", message_pairs_processed
                )
            except Exception:
                pass  # Fail silently if MLflow not properly configured

        self._save_memory_store()

    def _extract_sessions(self, conversation_data: Dict) -> Dict[str, List[Dict]]:
        """Extract and organize conversation sessions."""
        sessions = {}

        # Get session keys and sort them
        session_keys = [
            k
            for k in conversation_data.keys()
            if k.startswith("session_") and not k.endswith("_date_time")
        ]
        session_keys.sort(key=lambda x: int(x.split("_")[1]))

        for session_key in session_keys:
            session_num = session_key.split("_")[1]
            date_key = f"session_{session_num}_date_time"

            if session_key in conversation_data and conversation_data[session_key]:
                session_messages = []
                for dialog in conversation_data[session_key]:
                    message = {
                        "speaker": dialog["speaker"],
                        "text": dialog["text"],
                        "timestamp": conversation_data.get(date_key, ""),
                        "session_id": session_num,
                    }
                    session_messages.append(message)

                sessions[session_num] = session_messages

        return sessions

    def _generate_session_summary(self, messages: List[Dict], session_id: str) -> str:
        """Generate a summary for a conversation session."""
        session_text = " ".join(
            [f"{msg['speaker']}: {msg['text']}" for msg in messages]
        )

        # Use LLM to generate summary (simplified)
        summary_prompt = f"Summarize this conversation session in 2-3 sentences: {session_text[:1000]}"

        # Mock summary generation - in practice use DSPy module
        return f"Session {session_id} summary: Discussion between speakers covering various topics."

    def _process_message_pair(
        self,
        msg1: Dict,
        msg2: Dict,
        session_id: str,
        sample_id: str,
        session_summary: str,
    ):
        """Process a pair of messages and extract memories."""

        # Prepare context for memory extraction
        current_exchange = (
            f"{msg1['speaker']}: {msg1['text']}\n{msg2['speaker']}: {msg2['text']}"
        )
        recent_messages = self._get_recent_context(sample_id, session_id)

        # Extract memories using DSPy module
        try:
            extraction_result = self.memory_extractor(
                conversation_summary=session_summary,
                recent_messages=recent_messages,
                current_exchange=current_exchange,
                speaker_context=f"Speakers: {msg1['speaker']}, {msg2['speaker']}",
            )

            # Parse extracted memories
            try:
                extracted_memories = json.loads(extraction_result.extracted_memories)
                if not isinstance(extracted_memories, list):
                    extracted_memories = [extracted_memories]
            except json.JSONDecodeError:
                print(
                    f"Failed to parse extracted memories: {extraction_result.extracted_memories}"
                )
                return

            # Process each extracted memory
            for mem_data in extracted_memories:
                if isinstance(mem_data, dict) and "content" in mem_data:
                    self._update_memory_store(mem_data, msg1["timestamp"], session_id)

        except Exception as e:
            print(f"Error in memory extraction: {e}")

    def _get_recent_context(
        self, sample_id: str, session_id: str, window_size: int = 10
    ) -> str:
        """Get recent message context for memory extraction."""
        # Simplified - return basic context
        return f"Recent context for {sample_id} session {session_id}"

    def _update_memory_store(self, memory_data: Dict, timestamp: str, session_id: str):
        """Update memory store with new memory using DSPy module."""

        # Create memory object
        memory_id = hashlib.md5(
            f"{memory_data['content']}_{timestamp}".encode()
        ).hexdigest()

        # Find similar existing memories using plain text search
        similar_memories = self._find_similar_memories_plain(memory_data["content"])

        try:
            # Use DSPy module to determine update operation
            update_result = self.memory_updater(
                candidate_memory=json.dumps(memory_data),
                similar_memories=json.dumps(similar_memories),
            )

            operation = update_result.operation.upper()

            if operation == "ADD":
                # Add new memory
                new_memory = Memory(
                    id=memory_id,
                    content=memory_data["content"],
                    speaker=memory_data.get("speaker", ""),
                    timestamp=timestamp,
                    session_id=session_id,
                    importance_score=memory_data.get("importance_score", 0.5),
                    keywords=memory_data.get("keywords", []),
                    memory_type=memory_data.get("memory_type", "fact"),
                    created_at=datetime.now().isoformat(),
                    updated_at=datetime.now().isoformat(),
                )
                self.memories[memory_id] = new_memory

            elif operation == "UPDATE":
                # Update existing memory
                if similar_memories:
                    existing_id = similar_memories[0]["id"]
                    if existing_id in self.memories:
                        self.memories[
                            existing_id
                        ].content = update_result.updated_content
                        self.memories[
                            existing_id
                        ].updated_at = datetime.now().isoformat()

            elif operation == "DELETE":
                # Delete contradicted memory
                if similar_memories:
                    existing_id = similar_memories[0]["id"]
                    if existing_id in self.memories:
                        del self.memories[existing_id]

            # NOOP - no action needed

        except Exception as e:
            print(f"Error in memory update: {e}")

    def _find_similar_memories_plain(self, content: str, top_k: int = 5) -> List[Dict]:
        """Find similar memories using plain text matching."""
        similar_memories = []
        content_lower = content.lower()

        # Extract key terms from content
        key_terms = set(content_lower.split())

        for memory_id, memory in self.memories.items():
            memory_terms = set(memory.content.lower().split())

            # Calculate simple overlap score
            overlap = len(key_terms.intersection(memory_terms))
            if overlap > 3:  # Threshold for similarity
                similar_memories.append(
                    {
                        "id": memory_id,
                        "content": memory.content,
                        "overlap_score": overlap,
                        "speaker": memory.speaker,
                        "timestamp": memory.timestamp,
                    }
                )

        # Sort by overlap score
        similar_memories.sort(key=lambda x: x["overlap_score"], reverse=True)
        return similar_memories[:top_k]

    def _react_memory_search(
        self, question: str, question_category: str
    ) -> tuple[List[Dict], List[Dict]]:
        """Use ReACT to iteratively search through memories."""
        all_memories_summary = self._generate_memory_summary()
        found_memories = []
        search_history = []
        previous_thoughts = ""

        # Maximum 3 iterations of ReACT search
        for iteration in range(1, 4):
            try:
                search_result = self.memory_searcher(
                    question=question,
                    all_memories_summary=all_memories_summary,
                    question_category=question_category,
                    search_iteration=iteration,
                    previous_thoughts=previous_thoughts,
                )

                # Record the thought and action
                search_history.append(
                    {
                        "iteration": iteration,
                        "thought": search_result.thought,
                        "action": search_result.action,
                    }
                )

                # Execute the search action
                if search_result.action != "DONE":
                    try:
                        criteria = json.loads(search_result.search_criteria)
                        matching_memories = self._search_memories_by_criteria(criteria)

                        # Add found memories to results
                        for mem in matching_memories:
                            memory_dict = {
                                "id": mem.id,
                                "content": mem.content,
                                "speaker": mem.speaker,
                                "timestamp": mem.timestamp,
                                "memory_type": mem.memory_type,
                                "importance_score": mem.importance_score,
                                "keywords": mem.keywords,
                            }
                            if memory_dict not in found_memories:
                                found_memories.append(memory_dict)
                    except:
                        pass

                # Update previous thoughts
                previous_thoughts += f"\nIteration {iteration}: {search_result.thought}\nAction: {search_result.action}"

                # Check if we should continue
                if (
                    search_result.continue_search == "NO"
                    or search_result.action == "DONE"
                ):
                    break

            except Exception as e:
                print(f"Error in ReACT search iteration {iteration}: {e}")
                break

        return found_memories, search_history

    def answer_question(
        self, question: str, question_category: str = "single-hop"
    ) -> Dict[str, Any]:
        """Answer a question using ReACT memory search."""

        # Use ReACT to find relevant memories
        found_memories, search_history = self._react_memory_search(
            question, question_category
        )

        # Log question answering metrics to MLflow if available
        if MLFLOW_AVAILABLE:
            try:
                mlflow.log_metric("qa_candidate_memories_count", len(found_memories))
                mlflow.log_param("qa_question_category", question_category)
                mlflow.log_param("qa_search_iterations", len(search_history))
            except Exception:
                pass  # Fail silently if MLflow not properly configured

        try:
            # Use ranking module to select most relevant memories
            ranking_result = self.memory_ranker(
                question=question,
                found_memories=json.dumps(found_memories),
                search_history=json.dumps(search_history),
                question_category=question_category,
            )

            # Parse relevant memories
            try:
                relevant_memories = json.loads(ranking_result.relevant_memories)
            except json.JSONDecodeError:
                relevant_memories = found_memories[:3]  # Fallback

            # Generate answer using DSPy module
            qa_result = self.qa_generator(
                question=question,
                relevant_memories=json.dumps(relevant_memories),
                question_category=question_category,
            )

            # Log answer generation metrics to MLflow if available
            if MLFLOW_AVAILABLE:
                try:
                    mlflow.log_metric(
                        "qa_relevant_memories_count", len(relevant_memories)
                    )
                    mlflow.log_param("qa_confidence", qa_result.confidence)
                except Exception:
                    pass  # Fail silently if MLflow not properly configured

            return {
                "answer": qa_result.answer,
                "reasoning": qa_result.reasoning,
                "confidence": qa_result.confidence,
                "relevant_memories": relevant_memories,
                "retrieval_reasoning": ranking_result.reasoning,
                "search_history": search_history,
            }

        except Exception as e:
            print(f"Error in question answering: {e}")
            return {
                "answer": "Unable to determine from available information",
                "reasoning": f"Error occurred: {e}",
                "confidence": "low",
                "relevant_memories": [],
                "retrieval_reasoning": "Error in retrieval",
                "search_history": search_history,
            }

    def evaluate_with_llm_judge(
        self, question: str, ground_truth: str, generated_answer: str
    ) -> Dict[str, Any]:
        """Evaluate answer using LLM-as-Judge methodology."""

        try:
            judge_result = self.llm_judge(
                question=question,
                ground_truth=ground_truth,
                generated_answer=generated_answer,
            )

            return {
                "judgment": judge_result.judgment,
                "reasoning": judge_result.reasoning,
                "is_correct": judge_result.judgment.upper() == "CORRECT",
            }

        except Exception as e:
            print(f"Error in LLM judge evaluation: {e}")
            return {
                "judgment": "ERROR",
                "reasoning": f"Evaluation error: {e}",
                "is_correct": False,
            }

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


def create_sota_memory_system() -> SOTAMemorySystem:
    """Factory function to create the SOTA memory system."""
    return SOTAMemorySystem()


if __name__ == "__main__":
    # Example usage and testing
    system = create_sota_memory_system()

    # Example conversation data (mock)
    conversation_data = {
        "conversation": {
            "session_1": [
                {"speaker": "Alice", "text": "I love hiking in the mountains"},
                {
                    "speaker": "Bob",
                    "text": "That sounds amazing! When did you start hiking?",
                },
            ],
            "session_1_date_time": "2023-05-01",
        }
    }

    # Process conversation
    system.process_conversation(conversation_data, "test_conv_1")

    # Answer question
    result = system.answer_question("What does Alice enjoy doing?", "single-hop")
    print(f"Answer: {result['answer']}")
    print(f"Reasoning: {result['reasoning']}")
    print(f"Search History: {json.dumps(result['search_history'], indent=2)}")

    # Evaluate with LLM judge
    evaluation = system.evaluate_with_llm_judge(
        "What does Alice enjoy doing?", "hiking", result["answer"]
    )
    print(f"LLM Judge: {evaluation['judgment']} - {evaluation['reasoning']}")
