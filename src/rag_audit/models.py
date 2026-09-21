from typing import List, Optional
from pydantic import BaseModel, Field


class AtomicClaim(BaseModel):
    """A single, indivisible factual claim extracted from an answer."""

    statement: str
    supported: bool
    reason: str = ""


class EvaluationMetrics(BaseModel):
    faithfulness: float = Field(..., ge=0.0, le=1.0)
    answer_relevance: float = Field(..., ge=0.0, le=1.0)
    reasoning: Optional[str] = None
    is_abstention: bool = False
    claims: List[AtomicClaim] = Field(default_factory=list)


class RAGItem(BaseModel):
    id: str
    question: str
    contexts: List[str]
    answer: str
    ground_truth: Optional[str] = None
    metrics: Optional[EvaluationMetrics] = None


class AuditDataset(BaseModel):
    total_queries: int
    avg_faithfulness: float
    avg_relevance: float
    items: List[RAGItem]