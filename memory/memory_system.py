"""
Memory System with Minimal Structure
================================================================
A lightweight, plain‑context memory system that lives entirely inside an LLM prompt.

Core principle: Let the LLM discover optimal strategies through scaling rather than
encoding human assumptions about memory organization, retrieval, and reasoning.
"""

import dspy
from datetime import datetime
import pickle

DEFAULT_MAX_TOKENS = 20_000


class MemoryEvolution(dspy.Signature):
    """Evolve the memory system based on new information.

    Given current memory state and new information, determine how to update the
    memory system. You may add, modify, connect, or reorganize memories in any
    way that improves future utility.
    """

    current_state: str = dspy.InputField(
        description="Current memory system state in any textual representation."
    )
    new_information: str = dspy.InputField(
        description="New information to potentially incorporate."
    )
    updated_state: str = dspy.OutputField(
        description="Updated memory system state optimized for future queries."
    )


class MemoryReasoning(dspy.Signature):
    """Reason over memory system to answer queries and generate insights.

    Use available tools to explore the memory system and derive relevant information.
    Generate new insights when supported by existing evidence.
    """

    query: str = dspy.InputField(description="Information need to satisfy.")
    response: str = dspy.OutputField(
        description="Comprehensive response including relevant memories and derived insights."
    )


class MemoryOptimization(dspy.Signature):
    """Optimize memory system for efficiency within resource constraints.

    Reorganize, compress, or restructure the memory system to fit within resource
    limits while maximizing utility for future queries.
    """

    memory_state: str = dspy.InputField(description="Current memory system state.")
    resource_limit: int = dspy.InputField(
        description="Maximum resource budget (e.g., token count)."
    )
    optimized_state: str = dspy.OutputField(
        description="Optimized memory system state within resource constraints."
    )


class MemorySystem(dspy.Module):
    """Adaptive memory system with minimal architectural constraints."""

    def __init__(
        self,
        persist: bool = False,
        filename: str = "memory.pkl",
        resource_limit: int = DEFAULT_MAX_TOKENS,
    ):
        super().__init__()
        self.persist = persist
        self.filename = filename
        self.resource_limit = resource_limit

        self.state: str = ""

        self.evolver = dspy.ChainOfThought(MemoryEvolution)
        self.reasoner = dspy.ReAct(
            MemoryReasoning,
            tools=[self._search, self._add_insight, self._get_state],
            max_iters=15,
        )
        self.optimizer = dspy.ChainOfThought(MemoryOptimization)

    def _search(self, query: str) -> str:
        """Search current memory state for relevant information."""
        if not self.state:
            return "Memory system is empty."
        return f"Current memory state:\n{self.state}\n\nQuery: {query}"

    def _add_insight(self, insight: str, evidence: str, certainty: float) -> str:
        """Add new insight to memory system."""
        timestamp = datetime.now().isoformat()

        new_info = f"Insight: {insight}\nEvidence: {evidence}\nCertainty: {certainty}\nTimestamp: {timestamp}"

        self.state = self.evolver(
            current_state=self.state, new_information=new_info
        ).updated_state

        if self.persist:
            self._save_state()

        return f"Added insight: {insight}"

    def _get_state(self) -> str:
        """Get current memory system state."""
        return self.state if self.state else "Memory system is empty."

    def update(self, information: str | list[str]) -> None:
        """Update memory system with new information."""
        if isinstance(information, list):
            information = "\n".join(information)

        timestamp = datetime.now().isoformat()
        new_info = f"Information: {information}\nTimestamp: {timestamp}"

        self.state = self.evolver(
            current_state=self.state, new_information=new_info
        ).updated_state

        if self.persist:
            self._save_state()

    def query(self, question: str) -> str:
        """Query the memory system using multi-hop reasoning."""
        result = self.reasoner(query=question)
        return result.response

    def optimize(self) -> None:
        """Optimize memory system for resource efficiency."""
        if not self.state:
            return

        self.state = self.optimizer(
            memory_state=self.state, resource_limit=self.resource_limit
        ).optimized_state

        if self.persist:
            self._save_state()

    def _save_state(self) -> None:
        """Persist memory state."""
        with open(self.filename, "wb") as f:
            pickle.dump(self.state, f)

    def load_state(self) -> None:
        """Load persisted memory state."""
        try:
            with open(self.filename, "rb") as f:
                self.state = pickle.load(f)
        except FileNotFoundError:
            self.state = ""


if __name__ == "__main__":
    import os

    api_key = os.environ.get("TOGETHER_API_KEY")

    MODEL = "together_ai/Qwen/Qwen3-235B-A22B-fp8-tput"
    # MODEL = "together_ai/google/gemma-3n-E4B-it"

    lm = dspy.LM(MODEL, api_key=api_key)
    dspy.configure(lm=lm)

    # Example usage
    memory = MemorySystem()

    # Add information
    memory.update(
        [
            "My name is Satyam",
            "My favourite color is blue",
            "I love programming in Python",
        ]
    )

    print("Memory state after updates:")
    print(memory.state)
    print("\n" + "=" * 50 + "\n")

    # Query with reasoning
    response = memory.query("What can you infer about my preferences and background?")
    print("Query response:")
    print(response)
