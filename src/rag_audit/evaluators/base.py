from abc import ABC, abstractmethod
from typing import Callable, List, Type

from pydantic import BaseModel, Field

from rag_audit.models import AtomicClaim, EvaluationMetrics, RAGItem

# A provider only needs to implement this: send a system+user prompt, get
# back a validated instance of the given schema. Everything else (which
# prompts to send, how to combine the results) is shared in run_evaluation
# below, so adding a new provider means writing one small function like this.
JsonChatFn = Callable[[str, str, Type[BaseModel]], BaseModel]


# --- Step 1: extract claims from the answer, and only the answer ---------
#
# The extractor never sees the question, contexts, or ground_truth. Feeding
# it anything besides the answer risks it "extracting" claims from the
# wrong source (we hit this: passing ground_truth here made the judge pull
# claims out of the reference answer instead of the one being evaluated).

EXTRACTION_SYSTEM_PROMPT = """You extract factual claims from a single piece of text.

Given only the Response below, first write one sentence of reasoning about \
whether it explicitly declines to answer or says it lacks enough information \
(e.g. "the documentation does not specify X", "I don't have enough context \
to determine Y"). Then set is_abstention based on that reasoning.

If it is not an abstention, extract the atomic (minimal, indivisible) \
factual claims the Response makes. Do not include any fact that comes from \
anywhere else — only what this Response actually says. When is_abstention is \
true, leave statements empty.

A claim that links an entity to a category, membership, location, or \
ownership (e.g. "X belongs to Y", "X is part of Y", "X is located in Y") is \
ONE claim about that relationship — never split the entity from what is \
being asserted about it. For example, given the sentence "Mount Kilimanjaro \
is one of the main peaks of the Himalayan range", extract the single claim \
"Mount Kilimanjaro is part of the Himalayan range" — do NOT split it into \
"Mount Kilimanjaro is a main peak" and "the Himalayas are a mountain range", \
since that loses the actual relationship being asserted and each fragment \
alone can look true even when the real claim is false.

Respond only with JSON matching the required schema."""


class ExtractionResult(BaseModel):
    # `reasoning` is declared before `is_abstention` on purpose: this is
    # generated token-by-token, so asking for the justification first lets
    # the model's own reasoning inform the boolean that follows. Asking for
    # the boolean first (as we originally did) meant the model committed to
    # an answer before "thinking", and would sometimes back-fill a
    # nonsensical justification for whatever it had already decided.
    reasoning: str = Field(
        default="",
        description="One sentence: does the Response decline to answer or say it lacks enough information?",
    )
    is_abstention: bool = Field(
        default=False,
        description="Based on the reasoning above: true if the Response explicitly says it lacks enough information to answer.",
    )
    statements: List[str] = Field(
        default_factory=list,
        description="Atomic factual claims found in the Response text, and nothing else. Leave empty when is_abstention is true.",
    )


def _extraction_prompt(item: RAGItem) -> str:
    return f'Response:\n"""{item.answer}"""'


# --- Step 2: verify one claim at a time against the contexts -------------
#
# One claim per call keeps each prompt small and focused, which is far more
# reliable for small/local models than asking for a whole verified list at
# once — and it's what lets us get a real per-claim "reason" back instead
# of an empty string.

VERIFICATION_SYSTEM_PROMPT = """You are a strict fact-checker. Given a Context and a single Claim, decide \
whether the Context logically supports the Claim.

First, explicitly check these cases:

1. Membership, not exclusivity: if the Claim says something IS one of several \
things listed in the Context (e.g. the Claim is "X is made of A" and the \
Context lists "A, B and C"), that is supported — the Context does not need to \
say A is the ONLY one. Only treat it as unsupported on these grounds if the \
Claim itself asserts exclusivity ("only", "exclusively", "solely", etc.) and \
the Context contradicts that.

2. Unit/format conversions always count as supported, including ones that \
need addition or a formula, not just multiplication or relabeling (e.g. \
"5000 ms" = "5 s", "1000 MB" = "1 GB", and just as much "100 °C" = "373.15 K" \
or any other exact conversion between measurement scales) — never reject a \
conversion just because the Context does not use the same unit name.

3. Direct synonyms (e.g. "rollback" = "revert the change") of something \
stated in the Context also count as supported.

Then set supported based on that check.

Respond only with JSON matching the required schema."""


class ClaimVerdict(BaseModel):
    # See ExtractionResult for why `reason` is declared before `supported`:
    # writing the justification first — including the explicit unit/synonym
    # check — makes the model actually reason through equivalences instead
    # of committing to a verdict and rationalizing it afterwards.
    reason: str = Field(
        ...,
        description=(
            "First check whether the claim is a word-for-word match, a member of a "
            "list without needing to be the only one, a unit/format conversion "
            "(including ones needing addition or a formula), or a direct synonym "
            "of something in the context. Then give a one-sentence justification "
            "for the verdict below."
        ),
    )
    supported: bool = Field(
        ...,
        description="Based on the reasoning above: true if the context logically supports the claim.",
    )


def _verification_prompt(contexts: str, statement: str) -> str:
    return f'Context:\n"""{contexts}"""\n\nClaim:\n"""{statement}"""'


# --- Step 3: score answer relevance, independent of faithfulness ---------

RELEVANCE_SYSTEM_PROMPT = """Score how directly the Answer addresses what the Question actually asked, \
on a 0.0-1.0 scale, independent of whether the answer is factually correct. \
An honest "I don't know" style answer should usually score low here, since \
it does not resolve the user's need.

First write a one-sentence reasoning, then set answer_relevance based on it.

Respond only with JSON matching the required schema."""


class RelevanceScore(BaseModel):
    reasoning: str = Field(default="", description="One sentence on how directly the Answer addresses the Question.")
    answer_relevance: float = Field(..., ge=0.0, le=1.0, description="Based on the reasoning above.")


def _relevance_prompt(item: RAGItem) -> str:
    return f"Question:\n{item.question}\n\nAnswer:\n{item.answer}"


def compute_faithfulness(is_abstention: bool, claims: List[AtomicClaim]) -> float:
    """Deterministically derive faithfulness from the judge's claim-level verdicts.

    An honest abstention ("the context doesn't say") is never a hallucination,
    so it always scores 1.0. Otherwise faithfulness is the fraction of atomic
    claims the contexts actually support — we never ask the LLM for this
    number directly, since LLMs are unreliable at producing calibrated
    floats but are good at classifying one claim at a time.
    """
    if is_abstention:
        return 1.0
    if not claims:
        return 0.0
    supported = sum(1 for claim in claims if claim.supported)
    return round(supported / len(claims), 2)


def run_evaluation(chat_json: JsonChatFn, item: RAGItem) -> EvaluationMetrics:
    """Shared three-step evaluation pipeline.

    Any provider just needs to supply `chat_json`; this drives the same
    extract -> verify-each-claim -> score-relevance flow so every provider
    behaves consistently and gets the same faithfulness math.
    """
    extraction = chat_json(EXTRACTION_SYSTEM_PROMPT, _extraction_prompt(item), ExtractionResult)

    claims: List[AtomicClaim] = []
    if not extraction.is_abstention:
        contexts = "\n---\n".join(item.contexts)
        for statement in extraction.statements:
            verdict = chat_json(
                VERIFICATION_SYSTEM_PROMPT, _verification_prompt(contexts, statement), ClaimVerdict
            )
            claims.append(
                AtomicClaim(statement=statement, supported=verdict.supported, reason=verdict.reason)
            )

    relevance = chat_json(RELEVANCE_SYSTEM_PROMPT, _relevance_prompt(item), RelevanceScore)

    return EvaluationMetrics(
        faithfulness=compute_faithfulness(extraction.is_abstention, claims),
        answer_relevance=relevance.answer_relevance,
        reasoning=relevance.reasoning or None,
        is_abstention=extraction.is_abstention,
        claims=claims,
    )


class Evaluator(ABC):
    """Common interface for anything that can score a RAGItem.

    Adding a new provider (OpenAI, Anthropic, etc.) means writing one class
    that implements a JsonChatFn for that provider's SDK and calls
    run_evaluation with it — the CLI and server never need to change.
    """

    @abstractmethod
    def evaluate(self, item: RAGItem) -> EvaluationMetrics:
        """Score a single RAGItem, returning its EvaluationMetrics."""
