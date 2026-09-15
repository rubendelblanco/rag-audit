import ollama

from rag_audit.evaluators.base import SYSTEM_PROMPT, Evaluator, build_user_prompt
from rag_audit.models import EvaluationMetrics, RAGItem


class OllamaEvaluator(Evaluator):
    """Evaluator backed by a local Ollama model."""

    def __init__(self, model: str):
        self.model = model

    def evaluate(self, item: RAGItem) -> EvaluationMetrics:
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(item)},
            ],
            format=EvaluationMetrics.model_json_schema(),
            options={"temperature": 0},
        )
        return EvaluationMetrics.model_validate_json(response["message"]["content"])
