from rag_audit.evaluators.base import Evaluator
from rag_audit.evaluators.ollama_evaluator import OllamaEvaluator

# provider prefix (the part before "/" in --model) -> Evaluator class.
# To add a new provider, write its Evaluator in this package and register it here.
_PROVIDERS: dict[str, type[Evaluator]] = {
    "ollama": OllamaEvaluator,
}


def get_evaluator(model_spec: str) -> Evaluator:
    """Build the right Evaluator from a "provider/model" spec, e.g. "ollama/llama3.2"."""
    provider, sep, model_name = model_spec.partition("/")
    if not sep:
        supported = ", ".join(_PROVIDERS)
        raise ValueError(
            f"Model spec must be 'provider/model' (e.g. 'ollama/llama3.2'). "
            f"Supported providers: {supported}"
        )

    evaluator_cls = _PROVIDERS.get(provider)
    if evaluator_cls is None:
        supported = ", ".join(_PROVIDERS)
        raise ValueError(
            f"Unknown evaluator provider '{provider}'. Supported providers: {supported}"
        )

    return evaluator_cls(model_name)


__all__ = ["Evaluator", "get_evaluator"]
