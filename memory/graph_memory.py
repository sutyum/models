"""
Memory System with Minimal Structure
================================================================
A lightweight, plain‑context memory system that lives entirely inside an LLM prompt.

Core principle: Let the LLM discover optimal strategies through scaling rather than
encoding human assumptions about memory organization, retrieval, and reasoning.
"""

import logging
from datetime import datetime
import pickle
import dspy

# Configure logging
LOG_LEVEL = logging.INFO
_fmt = "%(asctime)s | %(levelname)8s | %(name)s | %(message)s"
logging.basicConfig(level=LOG_LEVEL, format=_fmt)
log = logging.getLogger("MemorySystem")

# Suppress LiteLLM's chatty INFO logs
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

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
        self.optimizer = dspy.ChainOfThought(MemoryOptimization)
        self.reasoner = dspy.ReAct(
            MemoryReasoning,
            tools=[self.search_tool, self.add_insight_tool, self.get_state_tool],
            max_iters=12,
        )
        log.info(
            "MemorySystem initialised (persist=%s, limit=%s tokens)",
            persist,
            resource_limit,
        )

    # ReAct tools (single-arg callables)
    def search_tool(self, query: str) -> str:
        snippet = self._search(query)
        log.debug("search_tool(%s) → %d chars", query, len(snippet))
        return snippet

    def add_insight_tool(self, payload: str) -> str:
        try:
            insight, evidence, cert = payload.split("|", 2)
            cert = float(cert)
        except ValueError:
            log.warning("Bad payload to add_insight_tool: %s", payload)
            return "FORMAT ERROR – use 'insight|evidence|certainty'"
        msg = self._add_insight(insight, evidence, cert)
        log.debug("add_insight_tool added: %s", insight)
        return msg

    def get_state_tool(self, _: str = "") -> str:
        log.debug("get_state_tool called")
        return self.state or "Memory empty."

    # Internal helpers with logging
    def _search(self, query: str) -> str:
        if not self.state:
            return "Memory empty."
        hits = [ln for ln in self.state.splitlines() if query.lower() in ln.lower()]
        result = "\n".join(hits)[:1_500] or "No match."
        return result

    def _add_insight(self, insight: str, evidence: str, certainty: float) -> str:
        timestamp = datetime.now().isoformat()
        new_info = (
            f"Insight: {insight}\nEvidence: {evidence}\nCertainty: "
            f"{certainty}\nTimestamp: {timestamp}"
        )
        self.state = self.evolver(
            current_state=self.state, new_information=new_info
        ).updated_state
        log.info("Insight added (certainty=%.2f): %s", certainty, insight)
        if self.persist:
            self._save_state()
        return f"Added insight: {insight}"

    def update(self, information: str | list[str]) -> None:
        """Update memory system with new information."""
        if isinstance(information, list):
            information = "\n".join(information)
        timestamp = datetime.now().isoformat()
        new_info = f"Information: {information}\nTimestamp: {timestamp}"
        self.state = self.evolver(
            current_state=self.state, new_information=new_info
        ).updated_state
        log.info("Memory updated (%d chars)", len(information))
        self.optimize()  # keep buffer in check
        if self.persist:
            self._save_state()

    def query(self, question: str) -> str:
        """Query the memory system using multi-hop reasoning."""
        log.info("Query: %s", question)
        result = self.reasoner(query=question)
        log.info(
            "Query finished after %d tokens used",
            getattr(result, "cost", {}).get("tokens", 0),
        )
        return result.response

    def optimize(self) -> None:
        """Optimize memory system for resource efficiency."""
        if not self.state:
            return
        before = len(self.state)
        self.state = self.optimizer(
            memory_state=self.state, resource_limit=self.resource_limit
        ).optimized_state
        log.debug("Optimiser shrunk %d → %d chars", before, len(self.state))

    def forward(self, conversation: str = None, question: str = None, text: str = None, update_memory: bool = True) -> str:
        """
        Single-entry API that either stores text (if update_memory=True)
        or treats text as a question and returns an answer.
        
        Also supports conversation/question interface for compatibility.
        """
        # Handle conversation/question interface  
        if conversation is not None and question is not None:
            # For multi-turn evaluation, process conversation incrementally
            self.update(conversation)
            answer = self.query(question)
            return dspy.Prediction(answer=answer)
        
        # Handle text interface
        if text is None:
            text = conversation or question
            
        if update_memory:
            self.update(text)
            log.debug(
                "forward() stored: %s", text[:50] + "..." if len(text) > 50 else text
            )
            return "✅ memory updated"

        answer = self.query(text)
        log.debug(
            "forward() answered: %s", text[:50] + "..." if len(text) > 50 else text
        )
        return answer

    def _save_state(self) -> None:
        """Persist memory state."""
        with open(self.filename, "wb") as f:
            pickle.dump(self.state, f)
        log.debug("State persisted to %s", self.filename)

    def load_state(self) -> None:
        """Load persisted memory state."""
        try:
            with open(self.filename, "rb") as f:
                self.state = pickle.load(f)
            log.info("State loaded from %s (%d chars)", self.filename, len(self.state))
        except FileNotFoundError:
            self.state = ""


if __name__ == "__main__":
    import os, sys

    # Configure LM
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        sys.exit("export GEMINI_API_KEY first")

    MODEL = "gemini/gemini-2.5-flash"
    dspy.configure(lm=dspy.LM(MODEL, api_key=api_key))

    # Test new single-entry API
    mem = MemorySystem()

    # Store facts using forward API
    print(mem("My name is Satyam", update_memory=True))
    print(mem("Favourite colour: blue", update_memory=True))
    print(mem("I love programming in Python", update_memory=True))

    # Add insight via tool interface
    mem.add_insight_tool("Satyam_is_coder|He codes in Python|0.95")

    # Ask question using forward API
    answer = mem("What do we know about Satyam's hobbies?", update_memory=False)
    print("\n=== RESPONSE ===\n", answer)
    print("\n=== MEMORY ===\n", mem.state[:800], "...\n")

    # Demo: simplified usage (default is update_memory=True)
    print("\n=== Simplified Usage ===")
    mem2 = MemorySystem()
    mem2("Caroline received support growing up and wants to be a counselor")
    mem2("She helps others because she was helped")
    response = mem2(
        "Would Caroline pursue counseling without early support?", update_memory=False
    )
    print(f"Q: Would Caroline pursue counseling without early support?")
    print(f"A: {response}")
