"""
LOCOMO-paper-accurate DSPy modules for conversational QA.
Based on the official LOCOMO paper methodology.
"""

import dspy
from typing import List, Dict, Optional


class LocomoConversationalQASignature(dspy.Signature):
    """
    Paper-accurate DSPy signature for LOCOMO conversational QA.
    Based on the paper's task definition.
    """

    conversation: str = dspy.InputField(
        desc="Multi-session conversation history between participants with dates and contexts. "
        "Look for information across all sessions to answer the question."
    )
    question: str = dspy.InputField(
        desc="Question about information from the conversation that may require reasoning across multiple turns"
    )
    answer: str = dspy.OutputField(
        desc="Answer based on the conversation. For unanswerable questions, respond with 'I don't know' or similar. "
        "For temporal questions, include specific dates/times. Use exact words from conversation when possible."
    )


class LocomoMultiHopSignature(dspy.Signature):
    """Signature specifically for multi-hop reasoning questions (Category 1)."""

    conversation: str = dspy.InputField(
        desc="Multi-session conversation history. You need to connect information from multiple turns."
    )
    question: str = dspy.InputField(
        desc="Question requiring multi-hop reasoning across different parts of the conversation"
    )
    reasoning: str = dspy.OutputField(
        desc="Step-by-step reasoning connecting multiple pieces of information from different conversation turns"
    )
    answer: str = dspy.OutputField(
        desc="Answer that synthesizes information from multiple conversation turns. May contain multiple parts separated by commas."
    )


class LocomoTemporalSignature(dspy.Signature):
    """Signature for temporal reasoning questions (Category 2)."""

    conversation: str = dspy.InputField(
        desc="Conversation history with dates and timestamps. Pay attention to temporal references."
    )
    question: str = dspy.InputField(
        desc="Question about when something happened or temporal relationships"
    )
    temporal_reasoning: str = dspy.OutputField(
        desc="Reasoning about temporal information and date calculations"
    )
    answer: str = dspy.OutputField(
        desc="Temporal answer with specific dates, times, or time periods. Use format from conversation."
    )


class LocomoUnanswerableSignature(dspy.Signature):
    """Signature for unanswerable questions (Category 4)."""

    conversation: str = dspy.InputField(
        desc="Conversation history that may not contain the information needed to answer the question"
    )
    question: str = dspy.InputField(
        desc="Question that may not be answerable from the given conversation"
    )
    analysis: str = dspy.OutputField(
        desc="Analysis of whether the conversation contains enough information to answer the question"
    )
    answer: str = dspy.OutputField(
        desc="Answer if information is available, or 'I don't know' / 'Cannot be determined' if not answerable"
    )


class LocomoAmbiguousSignature(dspy.Signature):
    """Signature for ambiguous questions (Category 5)."""

    conversation: str = dspy.InputField(
        desc="Conversation history that may support multiple valid interpretations"
    )
    question: str = dspy.InputField(
        desc="Question that could have multiple valid answers or interpretations"
    )
    interpretation: str = dspy.OutputField(
        desc="Analysis of possible interpretations and which one(s) are supported by the conversation"
    )
    answer: str = dspy.OutputField(
        desc="Answer based on the most supported interpretation, or multiple answers if appropriate"
    )


class PaperAccurateLocomoQA(dspy.Module):
    """
    Paper-accurate LOCOMO QA module that handles all question categories
    according to the methodology described in the paper.
    """

    def __init__(self):
        super().__init__()
        self.general_qa = dspy.ChainOfThought(LocomoConversationalQASignature)
        self.multi_hop_qa = dspy.ChainOfThought(LocomoMultiHopSignature)
        self.temporal_qa = dspy.ChainOfThought(LocomoTemporalSignature)
        self.unanswerable_qa = dspy.ChainOfThought(LocomoUnanswerableSignature)
        self.ambiguous_qa = dspy.ChainOfThought(LocomoAmbiguousSignature)

    def forward(
        self, conversation: str, question: str, category: Optional[int] = None
    ) -> dspy.Prediction:
        """
        Forward pass that routes to appropriate handler based on question category.
        """
        if category == 1:
            # Multi-hop reasoning
            response = self.multi_hop_qa(conversation=conversation, question=question)
            return dspy.Prediction(
                answer=response.answer,
                reasoning=response.reasoning,
                category=1,
                question_type="multi_hop",
            )

        elif category == 2:
            # Temporal reasoning
            response = self.temporal_qa(conversation=conversation, question=question)
            return dspy.Prediction(
                answer=response.answer,
                reasoning=response.temporal_reasoning,
                category=2,
                question_type="temporal",
            )

        elif category == 3:
            # Single-hop factual
            response = self.general_qa(conversation=conversation, question=question)
            return dspy.Prediction(
                answer=response.answer, category=3, question_type="single_hop"
            )

        elif category == 4:
            # Unanswerable
            response = self.unanswerable_qa(
                conversation=conversation, question=question
            )
            return dspy.Prediction(
                answer=response.answer,
                reasoning=response.analysis,
                category=4,
                question_type="unanswerable",
            )

        elif category == 5:
            # Ambiguous
            response = self.ambiguous_qa(conversation=conversation, question=question)
            return dspy.Prediction(
                answer=response.answer,
                reasoning=response.interpretation,
                category=5,
                question_type="ambiguous",
            )

        else:
            # Default to general QA
            response = self.general_qa(conversation=conversation, question=question)
            return dspy.Prediction(
                answer=response.answer, category=category or 3, question_type="general"
            )


class MemoryAwareLocomoQA(dspy.Module):
    """
    Memory-aware LOCOMO QA that considers conversation distance
    as emphasized in the paper.
    """

    def __init__(self):
        super().__init__()
        self.context_analyzer = dspy.ChainOfThought(
            "conversation, question -> relevant_sessions: list[str], memory_distance: str, context_summary: str",
            instructions="Identify which conversation sessions contain relevant information and how far back they are",
        )
        self.answer_generator = dspy.ChainOfThought(
            "context_summary, question, memory_distance -> answer: str, confidence: str",
            instructions="Answer based on the relevant context, considering how distant the information is in the conversation history",
        )

    def forward(self, conversation: str, question: str) -> dspy.Prediction:
        # Analyze conversation for relevant context and memory distance
        context_analysis = self.context_analyzer(
            conversation=conversation, question=question
        )

        # Generate answer considering memory distance
        answer_result = self.answer_generator(
            context_summary=context_analysis.context_summary,
            question=question,
            memory_distance=context_analysis.memory_distance,
        )

        return dspy.Prediction(
            answer=answer_result.answer,
            relevant_sessions=context_analysis.relevant_sessions,
            memory_distance=context_analysis.memory_distance,
            confidence=answer_result.confidence,
            context_summary=context_analysis.context_summary,
        )


class SimpleLocomoQA(dspy.Module):
    """Simplified version for quick testing."""

    def __init__(self):
        super().__init__()
        self.qa = dspy.ChainOfThought(LocomoConversationalQASignature)

    def forward(self, conversation: str, question: str) -> dspy.Prediction:
        return self.qa(conversation=conversation, question=question)


def create_module(module_type: str = "paper_accurate") -> dspy.Module:
    """
    Factory function to create paper-accurate LOCOMO QA modules.

    Args:
        module_type: Type of module to create
            - "paper_accurate": Full category-aware implementation
            - "memory_aware": Memory distance-aware implementation
            - "simple": Simplified for quick testing

    Returns:
        Initialized DSPy module
    """
    if module_type == "paper_accurate":
        return PaperAccurateLocomoQA()
    elif module_type == "memory_aware":
        return MemoryAwareLocomoQA()
    elif module_type == "simple":
        return SimpleLocomoQA()
    else:
        raise ValueError(f"Unknown module type: {module_type}")


# if __name__ == "__main__":
#     # Example usage
#     import os
#
#     # Configure DSPy with a language model
#     lm = dspy.LM("openai/gpt-4o-mini", api_key=os.environ.get("OPENAI_API_KEY", ""))
#     dspy.configure(lm=lm)
#
#     # Create paper-accurate module
#     qa_module = create_module("paper_accurate")
#
#     # Example conversation
#     conversation = """
#     DATE: 2023-05-01
#     CONVERSATION:
#     Alice: "I'm planning to visit the new art museum next week."
#     Bob: "Which one? The modern art museum or the classical one?"
#     Alice: "The modern art museum. They have a new exhibition on contemporary sculpture."
#
#     DATE: 2023-05-08
#     CONVERSATION:
#     Alice: "I went to the museum yesterday. The sculpture exhibition was amazing!"
#     Bob: "Did you see the piece by the famous artist we discussed?"
#     Alice: "Yes! The metal sculpture was incredible. Very thought-provoking."
#     """
#
#     # Test different question categories
#
#     # Category 1: Multi-hop
#     question1 = "What type of art did Alice see at the museum she planned to visit?"
#     result1 = qa_module(conversation=conversation, question=question1, category=1)
#     print(f"Multi-hop Q: {question1}")
#     print(f"Answer: {result1.answer}")
#     print(f"Reasoning: {result1.reasoning}\n")
#
#     # Category 2: Temporal
#     question2 = "When did Alice actually visit the museum?"
#     result2 = qa_module(conversation=conversation, question=question2, category=2)
#     print(f"Temporal Q: {question2}")
#     print(f"Answer: {result2.answer}")
#     print(f"Reasoning: {result2.reasoning}\n")
#
#     # Category 4: Unanswerable
#     question4 = "How much did Alice pay for the museum ticket?"
#     result4 = qa_module(conversation=conversation, question=question4, category=4)
#     print(f"Unanswerable Q: {question4}")
#     print(f"Answer: {result4.answer}")
#     print(f"Analysis: {result4.reasoning}")
