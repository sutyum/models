"""
Memory System for LOCOMO - Clean DSPy Implementation
Implements a memory-based architecture with LLM-as-Judge evaluation.
"""

import dspy
import json
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import pickle
from pathlib import Path
from locomo.llm_judge import LOCOMOJudge


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
    conversation_summary: str = dspy.InputField(desc="High-level summary of the conversation context")
    recent_messages: str = dspy.InputField(desc="Recent message history for context")
    current_exchange: str = dspy.InputField(desc="Current message pair to extract memories from")
    speaker_context: str = dspy.InputField(desc="Information about the speakers")
    
    extracted_memories: str = dspy.OutputField(
        desc="JSON list of extracted memories with fields: content, speaker, importance_score (0-1), keywords, memory_type"
    )


class MemoryUpdateSignature(dspy.Signature):
    """Determine update operations for extracted memories."""
    candidate_memory: str = dspy.InputField(desc="New memory candidate to be processed")
    similar_memories: str = dspy.InputField(desc="JSON list of existing similar memories")
    
    operation: str = dspy.OutputField(desc="Memory operation: ADD, UPDATE, DELETE, or NOOP")
    reasoning: str = dspy.OutputField(desc="Brief explanation of why this operation was chosen")
    updated_content: str = dspy.OutputField(desc="If UPDATE, provide enhanced content; else original")


class ReACTMemorySearchSignature(dspy.Signature):
    """ReACT-style iterative memory search."""
    question: str = dspy.InputField(desc="Question that needs to be answered")
    all_memories_summary: str = dspy.InputField(desc="Summary of available memories")
    question_category: str = dspy.InputField(desc="Type of question")
    search_iteration: int = dspy.InputField(desc="Current search iteration (1-3)")
    previous_thoughts: str = dspy.InputField(desc="Previous reasoning steps")
    
    thought: str = dspy.OutputField(desc="Current reasoning about relevant memories")
    action: str = dspy.OutputField(desc="Search action: SEARCH_KEYWORD, SEARCH_SPEAKER, SEARCH_TIME, SEARCH_TYPE, or DONE")
    search_criteria: str = dspy.OutputField(desc="JSON with search parameters")
    found_memories: str = dspy.OutputField(desc="JSON list of memory IDs that match")
    continue_search: str = dspy.OutputField(desc="YES if more iterations needed, NO otherwise")


class MemoryRankingSignature(dspy.Signature):
    """Rank and filter memories found through search."""
    question: str = dspy.InputField(desc="Original question to be answered")
    found_memories: str = dspy.InputField(desc="JSON list of memories found through search")
    search_history: str = dspy.InputField(desc="History of search thoughts and actions")
    question_category: str = dspy.InputField(desc="Type of question")
    
    reasoning: str = dspy.OutputField(desc="Explanation of memory ranking")
    relevant_memories: str = dspy.OutputField(desc="JSON list of top memories ranked by relevance")


class MemoryBasedQASignature(dspy.Signature):
    """Generate answers using retrieved memories."""
    question: str = dspy.InputField(desc="Question to be answered")
    relevant_memories: str = dspy.InputField(desc="JSON list of relevant memories")
    question_category: str = dspy.InputField(desc="Question type")
    
    reasoning: str = dspy.OutputField(desc="Step-by-step reasoning using the memories")
    answer: str = dspy.OutputField(desc="Concise answer (2-6 words) based on memories")
    confidence: str = dspy.OutputField(desc="Confidence level: high, medium, or low")


class MemorySystem(dspy.Module):
    """Memory system for LOCOMO benchmark."""
    
    def __init__(self, memory_store_path: str = "./memory_store"):
        super().__init__()
        
        # DSPy modules
        self.memory_extractor = dspy.ChainOfThought(MemoryExtractorSignature)
        self.memory_updater = dspy.ChainOfThought(MemoryUpdateSignature)
        self.memory_searcher = dspy.ChainOfThought(ReACTMemorySearchSignature)
        self.memory_ranker = dspy.ChainOfThought(MemoryRankingSignature)
        self.qa_generator = dspy.ChainOfThought(MemoryBasedQASignature)
        
        # Use LOCOMO's LLM Judge
        self.llm_judge = LOCOMOJudge()
        
        # Memory storage
        self.memory_store_path = Path(memory_store_path)
        self.memory_store_path.mkdir(exist_ok=True)
        self.memories: Dict[str, Memory] = {}
        self.conversation_summaries: Dict[str, str] = {}
        
        # Load existing memories
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
        if not self.memories:
            return "No memories available yet."
        
        summary_parts = []
        
        # Group by type and speaker
        by_type = {}
        by_speaker = {}
        
        for memory in self.memories.values():
            if memory.memory_type not in by_type:
                by_type[memory.memory_type] = []
            by_type[memory.memory_type].append(memory)
            
            if memory.speaker not in by_speaker:
                by_speaker[memory.speaker] = []
            by_speaker[memory.speaker].append(memory)
        
        summary_parts.append(f"Total memories: {len(self.memories)}")
        summary_parts.append(f"Memory types: {', '.join(by_type.keys())}")
        summary_parts.append(f"Speakers: {', '.join(by_speaker.keys())}")
        
        # Add samples from each type
        for mem_type, mems in by_type.items():
            summary_parts.append(f"\n{mem_type.capitalize()} memories ({len(mems)}):")
            for mem in mems[:3]:
                summary_parts.append(f"  - {mem.speaker}: {mem.content[:60]}...")
        
        return "\n".join(summary_parts)
    
    def _search_memories_by_criteria(self, search_criteria: Dict[str, Any]) -> List[Memory]:
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
        
        # Extract conversation sessions
        sessions = self._extract_sessions(conversation_data["conversation"])
        
        # Process each session
        for session_id, messages in sessions.items():
            session_summary = self._generate_session_summary(messages, session_id)
            self.conversation_summaries[f"{sample_id}_{session_id}"] = session_summary
            
            # Process message pairs
            for i in range(0, len(messages) - 1, 2):
                if i + 1 < len(messages):
                    self._process_message_pair(
                        messages[i], 
                        messages[i + 1], 
                        session_id, 
                        sample_id, 
                        session_summary
                    )
        
        self._save_memory_store()
    
    def _extract_sessions(self, conversation_data: Dict) -> Dict[str, List[Dict]]:
        """Extract and organize conversation sessions."""
        sessions = {}
        
        # Get session keys
        session_keys = [k for k in conversation_data.keys() 
                       if k.startswith("session_") and not k.endswith("_date_time")]
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
        """Generate a summary for a conversation session using LLM."""
        # For demo/development, use simple fallback to avoid getting stuck
        speakers = set(msg['speaker'] for msg in messages)
        topics = []
        
        # Extract simple topics from first few messages
        for msg in messages[:3]:
            words = msg['text'].lower().split()
            # Look for topic-indicating words
            for word in words:
                if len(word) > 5 and word not in ['conversation', 'discussion', 'talking']:
                    topics.append(word)
                    if len(topics) >= 3:
                        break
        
        topic_str = f" about {', '.join(topics[:3])}" if topics else ""
        return f"Session {session_id}: Conversation between {', '.join(speakers)}{topic_str}."
        
        # Original LLM-based approach (commented out to avoid getting stuck in demo)
        # try:
        #     conversation_text = []
        #     for msg in messages[:5]:  # Limit to first 5 messages
        #         conversation_text.append(f"{msg['speaker']}: {msg['text'][:100]}")  # Truncate long messages
        #     
        #     class SummarizeSignature(dspy.Signature):
        #         conversation: str = dspy.InputField(desc="Conversation to summarize")
        #         summary: str = dspy.OutputField(desc="Brief 1-sentence summary")
        #     
        #     summarizer = dspy.ChainOfThought(SummarizeSignature)
        #     result = summarizer(conversation="\n".join(conversation_text))
        #     return result.summary
        # except Exception as e:
        #     print(f"   Warning: Summary generation failed: {e}")
        #     return f"Session {session_id}: Conversation between {', '.join(speakers)}."
    
    def _process_message_pair(self, msg1: Dict, msg2: Dict, session_id: str, 
                             sample_id: str, session_summary: str):
        """Process a pair of messages and extract memories."""
        # Prepare context
        current_exchange = f"{msg1['speaker']}: {msg1['text']}\n{msg2['speaker']}: {msg2['text']}"
        recent_messages = f"Session {session_id} in conversation {sample_id}"
        
        # Extract memories
        try:
            extraction_result = self.memory_extractor(
                conversation_summary=session_summary,
                recent_messages=recent_messages,
                current_exchange=current_exchange,
                speaker_context=f"Speakers: {msg1['speaker']}, {msg2['speaker']}"
            )
            
            # Parse extracted memories
            try:
                extracted_memories = json.loads(extraction_result.extracted_memories)
                if not isinstance(extracted_memories, list):
                    extracted_memories = [extracted_memories]
            except:
                return
            
            # Process each memory
            for mem_data in extracted_memories:
                if isinstance(mem_data, dict) and "content" in mem_data:
                    self._update_memory_store(mem_data, msg1["timestamp"], session_id)
        
        except Exception as e:
            print(f"Error in memory extraction: {e}")
    
    def _update_memory_store(self, memory_data: Dict, timestamp: str, session_id: str):
        """Update memory store with new memory."""
        # Create memory ID
        memory_id = hashlib.md5(
            f"{memory_data['content']}_{timestamp}".encode()
        ).hexdigest()
        
        # Find similar memories
        similar_memories = self._find_similar_memories(memory_data["content"])
        
        try:
            # Determine update operation
            update_result = self.memory_updater(
                candidate_memory=json.dumps(memory_data),
                similar_memories=json.dumps(similar_memories)
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
                    updated_at=datetime.now().isoformat()
                )
                self.memories[memory_id] = new_memory
            
            elif operation == "UPDATE" and similar_memories:
                # Update existing memory
                existing_id = similar_memories[0]["id"]
                if existing_id in self.memories:
                    self.memories[existing_id].content = update_result.updated_content
                    self.memories[existing_id].updated_at = datetime.now().isoformat()
            
            elif operation == "DELETE" and similar_memories:
                # Delete contradicted memory
                existing_id = similar_memories[0]["id"]
                if existing_id in self.memories:
                    del self.memories[existing_id]
        
        except Exception as e:
            print(f"Error in memory update: {e}")
    
    def _find_similar_memories(self, content: str, top_k: int = 5) -> List[Dict]:
        """Find similar memories using text overlap."""
        similar_memories = []
        content_lower = content.lower()
        key_terms = set(content_lower.split())
        
        for memory_id, memory in self.memories.items():
            memory_terms = set(memory.content.lower().split())
            overlap = len(key_terms.intersection(memory_terms))
            
            if overlap > 3:  # Threshold for similarity
                similar_memories.append({
                    "id": memory_id,
                    "content": memory.content,
                    "overlap_score": overlap,
                    "speaker": memory.speaker,
                    "timestamp": memory.timestamp
                })
        
        # Sort by overlap score
        similar_memories.sort(key=lambda x: x["overlap_score"], reverse=True)
        return similar_memories[:top_k]
    
    def _react_memory_search(self, question: str, question_category: str) -> tuple[List[Dict], List[Dict]]:
        """Use ReACT to iteratively search through memories."""
        all_memories_summary = self._generate_memory_summary()
        found_memories = []
        search_history = []
        previous_thoughts = ""
        
        # Maximum 3 iterations
        for iteration in range(1, 4):
            try:
                search_result = self.memory_searcher(
                    question=question,
                    all_memories_summary=all_memories_summary,
                    question_category=question_category,
                    search_iteration=iteration,
                    previous_thoughts=previous_thoughts
                )
                
                # Record search step
                search_history.append({
                    "iteration": iteration,
                    "thought": search_result.thought,
                    "action": search_result.action
                })
                
                # Execute search
                if search_result.action != "DONE":
                    try:
                        criteria = json.loads(search_result.search_criteria)
                        matching_memories = self._search_memories_by_criteria(criteria)
                        
                        # Add to results
                        for mem in matching_memories:
                            memory_dict = {
                                "id": mem.id,
                                "content": mem.content,
                                "speaker": mem.speaker,
                                "timestamp": mem.timestamp,
                                "memory_type": mem.memory_type,
                                "importance_score": mem.importance_score,
                                "keywords": mem.keywords
                            }
                            if memory_dict not in found_memories:
                                found_memories.append(memory_dict)
                    except:
                        pass
                
                # Update thoughts
                previous_thoughts += f"\nIteration {iteration}: {search_result.thought}"
                
                # Check if done
                if search_result.continue_search == "NO" or search_result.action == "DONE":
                    break
            
            except Exception as e:
                print(f"Error in ReACT search iteration {iteration}: {e}")
                break
        
        return found_memories, search_history
    
    def answer_question(self, question: str, question_category: str = "single-hop") -> Dict[str, Any]:
        """Answer a question using ReACT memory search."""
        # For demo mode with empty memory store, provide a fallback answer
        if not self.memories:
            return {
                "answer": "Information not available in memory",
                "reasoning": "No memories stored yet - this is a demo with empty memory store",
                "confidence": "low",
                "relevant_memories": [],
                "retrieval_reasoning": "No memories to search",
                "search_history": []
            }
        
        # Search for relevant memories
        found_memories, search_history = self._react_memory_search(question, question_category)
        
        try:
            # Rank memories
            ranking_result = self.memory_ranker(
                question=question,
                found_memories=json.dumps(found_memories),
                search_history=json.dumps(search_history),
                question_category=question_category
            )
            
            # Parse relevant memories
            try:
                relevant_memories = json.loads(ranking_result.relevant_memories)
            except:
                relevant_memories = found_memories[:3]
            
            # Generate answer
            qa_result = self.qa_generator(
                question=question,
                relevant_memories=json.dumps(relevant_memories),
                question_category=question_category
            )
            
            return {
                "answer": qa_result.answer,
                "reasoning": qa_result.reasoning,
                "confidence": qa_result.confidence,
                "relevant_memories": relevant_memories,
                "retrieval_reasoning": ranking_result.reasoning,
                "search_history": search_history
            }
        
        except Exception as e:
            print(f"Error in question answering: {e}")
            return {
                "answer": "Unable to determine from available information",
                "reasoning": f"Error occurred: {e}",
                "confidence": "low",
                "relevant_memories": [],
                "retrieval_reasoning": "Error in retrieval",
                "search_history": search_history
            }
    
    def evaluate_with_llm_judge(self, question: str, ground_truth: str, 
                               generated_answer: str) -> Dict[str, Any]:
        """Evaluate answer using LOCOMO's LLM judge."""
        return self.llm_judge.evaluate_answer(question, ground_truth, generated_answer)
    
    def forward(self, question: str, question_category: str = "single-hop") -> dspy.Prediction:
        """Forward pass for DSPy optimization."""
        result = self.answer_question(question, question_category)
        
        return dspy.Prediction(
            answer=result["answer"],
            reasoning=result["reasoning"],
            confidence=result["confidence"]
        )
    
    def batch(self, examples: List[dspy.Example]) -> List[dspy.Prediction]:
        """Batch process multiple examples for parallel inference."""
        predictions = []
        
        for example in examples:
            try:
                # Extract inputs
                question = example.question
                question_category = getattr(example, 'question_category', 'single-hop')
                
                # Get prediction
                result = self.answer_question(question, question_category)
                
                predictions.append(dspy.Prediction(
                    answer=result["answer"],
                    reasoning=result["reasoning"],
                    confidence=result["confidence"]
                ))
            except Exception as e:
                # Handle errors gracefully
                predictions.append(dspy.Prediction(
                    answer="Unable to determine",
                    reasoning=f"Error: {str(e)}",
                    confidence="low"
                ))
        
        return predictions


def create_memory_system() -> MemorySystem:
    """Factory function to create the memory system."""
    return MemorySystem()


if __name__ == "__main__":
    # Example usage
    system = create_memory_system()
    
    # Example conversation
    conversation_data = {
        "conversation": {
            "session_1": [
                {"speaker": "Alice", "text": "I love hiking in the mountains"},
                {"speaker": "Bob", "text": "That sounds amazing! When did you start?"}
            ],
            "session_1_date_time": "2023-05-01"
        }
    }
    
    # Process conversation
    system.process_conversation(conversation_data, "test_conv_1")
    
    # Answer question
    result = system.answer_question("What does Alice enjoy doing?", "single-hop")
    print(f"Answer: {result['answer']}")
    print(f"Reasoning: {result['reasoning']}")