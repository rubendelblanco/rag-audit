from typing import Type, TypeVar

import ollama
from pydantic import BaseModel

from rag_audit.evaluators.base import Evaluator, run_evaluation
from rag_audit.models import EvaluationMetrics, RAGItem

M = TypeVar("M", bound=BaseModel)


class OllamaEvaluator(Evaluator):
    """Evaluator backed by a local Ollama model."""

    def __init__(self, model: str):
        self.model = model

    def _chat_json(self, system: str, user: str, schema: Type[M]) -> M:
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            format=schema.model_json_schema(),
            options={"temperature": 0},
        )
        return schema.model_validate_json(response["message"]["content"])

    def evaluate(self, item: RAGItem) -> EvaluationMetrics:
        return run_evaluation(self._chat_json, item)
