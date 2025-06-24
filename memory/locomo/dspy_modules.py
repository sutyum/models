"""
DSPy modules and signatures for LOCOMO conversational QA.
"""
import dspy
from typing import List, Dict, Optional
from locomo.evaluation import f1_score, exact_match_score, normalize_answer


class ConversationalQASignature(dspy.Signature):
    """
    DSPy signature for conversational question answering based on conversation history.
    """
    conversation: str = dspy.InputField(
        desc="Multi-turn conversation history between participants, including dates and context"
    )
    question: str = dspy.InputField(
        desc="Question about information discussed in the conversation"
    )
    answer: str = dspy.OutputField(
        desc="Short, precise answer based on the conversation. Use exact words from conversation when possible."
    )


class ConversationalQAWithReasoningSignature(dspy.Signature):
    """
    DSPy signature for conversational QA with explicit reasoning.
    """
    conversation: str = dspy.InputField(
        desc="Multi-turn conversation history between participants, including dates and context"
    )
    question: str = dspy.InputField(
        desc="Question about information discussed in the conversation"
    )
    reasoning: str = dspy.OutputField(
        desc="Step-by-step reasoning about how to find the answer in the conversation"
    )
    answer: str = dspy.OutputField(
        desc="Short, precise answer based on the conversation. Use exact words from conversation when possible."
    )


class TemporalQASignature(dspy.Signature):
    """
    Specialized signature for temporal questions (category 2).
    """
    conversation: str = dspy.InputField(
        desc="Multi-turn conversation history with dates and timestamps"
    )
    question: str = dspy.InputField(
        desc="Question asking about when something happened or temporal relationships"
    )
    answer: str = dspy.OutputField(
        desc="Date or time-based answer. Use the conversation dates to provide approximate timing."
    )


class AdversarialQASignature(dspy.Signature):
    """
    Specialized signature for adversarial questions (category 5).
    """
    conversation: str = dspy.InputField(
        desc="Multi-turn conversation history between participants"
    )
    question: str = dspy.InputField(
        desc="Question that may or may not be answerable from the conversation"
    )
    answer: str = dspy.OutputField(
        desc="Answer if information is available, or 'Not mentioned in the conversation' if not answerable"
    )


class BasicConversationalQA(dspy.Module):
    """Basic conversational QA module using simple prediction."""
    
    def __init__(self):
        super().__init__()
        self.qa = dspy.Predict(ConversationalQASignature)
    
    def forward(self, conversation: str, question: str) -> dspy.Prediction:
        return self.qa(conversation=conversation, question=question)


class ChainOfThoughtConversationalQA(dspy.Module):
    """Conversational QA module with chain of thought reasoning."""
    
    def __init__(self):
        super().__init__()
        self.qa = dspy.ChainOfThought(ConversationalQAWithReasoningSignature)
    
    def forward(self, conversation: str, question: str) -> dspy.Prediction:
        return self.qa(conversation=conversation, question=question)


class CategoryAwareConversationalQA(dspy.Module):
    """
    Category-aware conversational QA that uses different strategies 
    based on question type.
    """
    
    def __init__(self):
        super().__init__()
        self.general_qa = dspy.ChainOfThought(ConversationalQAWithReasoningSignature)
        self.temporal_qa = dspy.ChainOfThought(TemporalQASignature) 
        self.adversarial_qa = dspy.ChainOfThought(AdversarialQASignature)
        self.category_classifier = dspy.Predict(
            "conversation, question -> category: int",
            instructions="Classify the question type: 1=multi-hop, 2=temporal, 3=single-hop, 4=open-domain, 5=adversarial"
        )
    
    def forward(self, conversation: str, question: str, category: Optional[int] = None) -> dspy.Prediction:
        # If category not provided, predict it
        if category is None:
            pred_category = self.category_classifier(conversation=conversation, question=question)
            category = pred_category.category
        
        # Route to appropriate module based on category
        if category == 2:
            # Temporal questions
            response = self.temporal_qa(conversation=conversation, question=question)
        elif category == 5:
            # Adversarial questions
            response = self.adversarial_qa(conversation=conversation, question=question)
        else:
            # General questions (categories 1, 3, 4)
            response = self.general_qa(conversation=conversation, question=question)
        
        # Add predicted category to response
        response.predicted_category = category
        return response


class MultiStepConversationalQA(dspy.Module):
    """
    Multi-step conversational QA that first extracts relevant context,
    then answers the question.
    """
    
    def __init__(self):
        super().__init__()
        self.context_extractor = dspy.ChainOfThought(
            "conversation, question -> relevant_context: str, reasoning: str",
            instructions="Extract the most relevant parts of the conversation for answering the question"
        )
        self.answer_generator = dspy.ChainOfThought(
            "relevant_context, question -> answer: str",
            instructions="Answer the question based on the relevant context. Be precise and use exact words when possible."
        )
    
    def forward(self, conversation: str, question: str) -> dspy.Prediction:
        # Step 1: Extract relevant context
        context_result = self.context_extractor(conversation=conversation, question=question)
        
        # Step 2: Generate answer from relevant context
        answer_result = self.answer_generator(
            relevant_context=context_result.relevant_context,
            question=question
        )
        
        # Combine results
        return dspy.Prediction(
            answer=answer_result.answer,
            relevant_context=context_result.relevant_context,
            context_reasoning=context_result.reasoning
        )


def create_locomo_qa_module(module_type: str = "chain_of_thought") -> dspy.Module:
    """
    Factory function to create different types of conversational QA modules.
    
    Args:
        module_type: Type of module to create
            - "basic": Basic prediction
            - "chain_of_thought": Chain of thought reasoning
            - "category_aware": Category-specific handling
            - "multi_step": Multi-step context extraction and answering
    
    Returns:
        Initialized DSPy module
    """
    if module_type == "basic":
        return BasicConversationalQA()
    elif module_type == "chain_of_thought":
        return ChainOfThoughtConversationalQA()
    elif module_type == "category_aware":
        return CategoryAwareConversationalQA()
    elif module_type == "multi_step":
        return MultiStepConversationalQA()
    else:
        raise ValueError(f"Unknown module type: {module_type}")


if __name__ == "__main__":
    # Example usage
    import os
    
    # Configure DSPy with a language model
    lm = dspy.LM('openai/gpt-4o-mini', api_key=os.environ.get('OPENAI_API_KEY', ''))
    dspy.configure(lm=lm)
    
    # Create a module
    qa_module = create_locomo_qa_module("chain_of_thought")
    
    # Example conversation and question
    conversation = """
    DATE: 2023-01-15
    CONVERSATION:
    Alice: "I went to the new Italian restaurant yesterday."
    Bob: "How was the food?"
    Alice: "The pasta was amazing, especially the carbonara."
    Bob: "I should try it sometime."
    """
    
    question = "What did Alice think of the carbonara?"
    
    # Get answer
    result = qa_module(conversation=conversation, question=question)
    print(f"Question: {question}")
    print(f"Answer: {result.answer}")
    if hasattr(result, 'reasoning'):
        print(f"Reasoning: {result.reasoning}")