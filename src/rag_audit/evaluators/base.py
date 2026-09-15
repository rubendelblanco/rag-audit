from abc import ABC, abstractmethod

from rag_audit.models import EvaluationMetrics, RAGItem

SYSTEM_PROMPT = """You are a strict evaluator of Retrieval-Augmented Generation (RAG) systems.
Given a question, the retrieved contexts, and the generated answer, score:

- faithfulness (0.0-1.0): how well the answer is supported by the contexts, \
without adding facts that are not present in them.
- answer_relevance (0.0-1.0): how directly the answer addresses the question asked.

Respond only with JSON matching the required schema. Keep "reasoning" brief \
(one or two sentences) and point out any unsupported claim or off-topic content."""


def build_user_prompt(item: RAGItem) -> str:
    contexts = "\n---\n".join(item.contexts)
    parts = [
        f"Question:\n{item.question}",
        f"Contexts:\n{contexts}",
        f"Answer:\n{item.answer}",
    ]
    if item.ground_truth:
        parts.append(f"Ground truth (reference answer):\n{item.ground_truth}")
    return "\n\n".join(parts)


class Evaluator(ABC):
    """Common interface for anything that can score a RAGItem.

    Adding a new provider (OpenAI, Anthropic, etc.) means writing one class
    that implements `evaluate` and registering it in `evaluators/__init__.py`
    — the CLI and server never need to change.
    """

    @abstractmethod
    def evaluate(self, item: RAGItem) -> EvaluationMetrics:
        """Score a single RAGItem, returning its EvaluationMetrics."""
