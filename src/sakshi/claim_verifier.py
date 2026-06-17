"""
claim_verifier.py — Sakshi-Protocol V3
NLI verification of claims against retrieved context.

Phase 2 of the V3 Omega redesign.

This is the Anumana instrument — inference from evidence.
Claims extracted by claim_extractor.py are verified against
retrieved context passages using Natural Language Inference.

The verifier determines whether retrieved context:
    ENTAILS      the claim (supports it)
    CONTRADICTS  the claim (conflicts with it)
    NEUTRAL      (neither supports nor contradicts)

Design principle: no language model in the verification loop.
The NLI model (cross-encoder/nli-deberta-v3-small) is a discriminative
classifier — it does not generate text, it classifies relationships.
It brings no parametric knowledge of its own — it evaluates only the
logical relationship between the claim and the context passage.

This is epistemically independent from the generator in a way that
self-verification (using the same LLM to evaluate its own outputs)
is not.

Model: cross-encoder/nli-deberta-v3-small (HuggingFace)
    - ~180MB one-time download
    - Runs locally, zero API cost after download
    - 3-class NLI: entailment, contradiction, neutral
    - Fast inference: ~50ms per claim on CPU

Fallback: when the model is not installed, a keyword-based heuristic
verifier is used. This is less accurate but preserves the pipeline
structure and produces conservative (neutral-biased) verdicts.

Paper correspondence: Anumana — inference from evidence.
The verifier draws conclusions from retrieved text rather than
asserting correctness from parametric knowledge.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from sakshi.claim_extractor import Claim, ExtractionResult, CLASS_A, CLASS_B, CLASS_C

# =============================================================================
# NLI MODEL SETUP
# =============================================================================

_NLI_MODEL   = None
_NLI_AVAILABLE = False
_NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

def _load_nli_model():
    """
    Load the NLI model. Called once on first use.
    Returns True if model loaded successfully, False otherwise.
    """
    global _NLI_MODEL, _NLI_AVAILABLE

    if _NLI_AVAILABLE:
        return True

    try:
        from sentence_transformers import CrossEncoder
        _NLI_MODEL = CrossEncoder(
            _NLI_MODEL_NAME,
            max_length=512,
        )
        _NLI_AVAILABLE = True
        return True
    except ImportError:
        return False
    except Exception:
        return False


# =============================================================================
# VERDICT
# =============================================================================

ENTAILMENT    = "entailment"
CONTRADICTION = "contradiction"
NEUTRAL       = "neutral"

@dataclass
class ClaimVerdict:
    """
    NLI verdict for a single claim against a context passage.

    Attributes:
        claim_text      : the claim being verified
        context_used    : the context passage used for verification
        verdict         : entailment | contradiction | neutral
        confidence      : confidence in the verdict [0, 1]
        entailment_score: raw entailment probability
        contradiction_score: raw contradiction probability
        neutral_score   : raw neutral probability
        method          : "nli_model" | "keyword_heuristic"
        claim_class     : CLASS_A | CLASS_B | CLASS_C
    """
    claim_text:          str
    context_used:        str   = ""
    verdict:             str   = NEUTRAL
    confidence:          float = 0.0
    entailment_score:    float = 0.0
    contradiction_score: float = 0.0
    neutral_score:       float = 1.0
    method:              str   = "keyword_heuristic"
    claim_class:         str   = CLASS_C

    def to_dict(self) -> dict:
        return {
            "claim_text":          self.claim_text[:200],
            "verdict":             self.verdict,
            "confidence":          round(self.confidence, 4),
            "entailment_score":    round(self.entailment_score, 4),
            "contradiction_score": round(self.contradiction_score, 4),
            "neutral_score":       round(self.neutral_score, 4),
            "method":              self.method,
            "claim_class":         self.claim_class,
        }


# =============================================================================
# VERIFICATION RESULT
# =============================================================================

@dataclass
class VerificationResult:
    """
    Aggregate verification result for all claims in an output.

    Attributes:
        verdicts            : list of ClaimVerdict per verified claim
        consistency_score   : float [0, 1] — aggregate grounding score
                              1.0 = all claims entailed by context
                              0.0 = all claims contradicted by context
        entailment_count    : number of claims entailed
        contradiction_count : number of claims contradicted
        neutral_count       : number of claims neutral
        has_contradiction   : True if any claim is directly contradicted
        verified_claims     : claims that passed verification (entailed)
        contradicted_claims : claims that failed verification (contradicted)
        method              : "nli_model" | "keyword_heuristic" | "skipped"
    """
    verdicts:             list  = field(default_factory=list)
    consistency_score:    float = 0.5
    entailment_count:     int   = 0
    contradiction_count:  int   = 0
    neutral_count:        int   = 0
    has_contradiction:    bool  = False
    verified_claims:      list  = field(default_factory=list)
    contradicted_claims:  list  = field(default_factory=list)
    method:               str   = "skipped"

    def to_dict(self) -> dict:
        return {
            "verdicts":             [v.to_dict() for v in self.verdicts],
            "consistency_score":    round(self.consistency_score, 4),
            "entailment_count":     self.entailment_count,
            "contradiction_count":  self.contradiction_count,
            "neutral_count":        self.neutral_count,
            "has_contradiction":    self.has_contradiction,
            "verified_claims":      self.verified_claims[:5],
            "contradicted_claims":  self.contradicted_claims[:5],
            "method":               self.method,
        }


# =============================================================================
# NLI VERIFICATION
# =============================================================================

def _verify_with_nli(claim_text: str, context: str) -> tuple[str, float, float, float, float]:
    """
    Verify a claim against context using the NLI model.

    Returns (verdict, confidence, entailment_score, contradiction_score, neutral_score).

    The model scores three hypotheses:
        premise = context passage
        hypothesis = claim text

    Output: probabilities for [contradiction, entailment, neutral]
    (DeBERTa NLI label order)
    """
    if not _NLI_AVAILABLE or _NLI_MODEL is None:
        return NEUTRAL, 0.0, 0.0, 0.0, 1.0

    try:
        import numpy as np

        # Truncate inputs to avoid exceeding max_length
        context_truncated = context[:800]
        claim_truncated   = claim_text[:200]

        scores = _NLI_MODEL.predict(
            [(context_truncated, claim_truncated)],
            apply_softmax=True,
        )[0]

        # DeBERTa NLI output order: [contradiction, entailment, neutral]
        contradiction_score = float(scores[0])
        entailment_score    = float(scores[1])
        neutral_score       = float(scores[2])

        # Determine verdict by highest score
        max_score = max(contradiction_score, entailment_score, neutral_score)

        if entailment_score == max_score:
            verdict    = ENTAILMENT
            confidence = entailment_score
        elif contradiction_score == max_score:
            verdict    = CONTRADICTION
            confidence = contradiction_score
        else:
            verdict    = NEUTRAL
            confidence = neutral_score

        return verdict, confidence, entailment_score, contradiction_score, neutral_score

    except Exception:
        return NEUTRAL, 0.0, 0.0, 0.0, 1.0


# =============================================================================
# KEYWORD HEURISTIC FALLBACK
# Used when NLI model is not installed.
# Conservative — biased toward NEUTRAL to avoid false contradictions.
# =============================================================================

def _verify_with_heuristic(claim_text: str, context: str) -> tuple[str, float, float, float, float]:
    """
    Keyword-based heuristic verification fallback.

    Conservative — biased toward NEUTRAL.
    Only flags CONTRADICTION when explicit negation of claim content
    is present in context. Only flags ENTAILMENT when claim content
    is directly present in context.
    """
    if not context or not claim_text:
        return NEUTRAL, 0.5, 0.0, 0.0, 1.0

    claim_lower   = claim_text.lower()
    context_lower = context.lower()

    # Extract key terms from claim (words > 4 chars, not stopwords)
    _STOPWORDS = {"this", "that", "with", "from", "have", "been", "were",
                  "they", "their", "there", "these", "those", "which", "would",
                  "could", "should", "about", "after", "before", "while"}
    claim_words = [
        w for w in re.findall(r'\b[a-z]{5,}\b', claim_lower)
        if w not in _STOPWORDS
    ]

    if not claim_words:
        return NEUTRAL, 0.5, 0.0, 0.0, 1.0

    # Count claim words appearing in context
    overlap = sum(1 for w in claim_words if w in context_lower)
    overlap_ratio = overlap / len(claim_words)

    # Check for explicit negation of key terms
    negation_patterns = [
        r"not\s+(?:found|approved|confirmed|established|known)",
        r"no\s+(?:evidence|approval|confirmation|record)",
        r"cannot\s+(?:confirm|verify|establish)",
        r"does\s+not\s+exist",
        r"never\s+(?:approved|established|confirmed)",
    ]
    has_negation = any(
        re.search(p, context_lower) for p in negation_patterns
    )

    if overlap_ratio > 0.5 and not has_negation:
        return ENTAILMENT, overlap_ratio, overlap_ratio, 0.0, 1 - overlap_ratio
    elif has_negation and overlap_ratio > 0.2:
        return CONTRADICTION, 0.6, 0.0, 0.6, 0.4
    else:
        return NEUTRAL, 0.5, max(0, overlap_ratio - 0.1), 0.0, 1.0


# =============================================================================
# CONSISTENCY SCORE COMPUTATION
# =============================================================================

def _compute_consistency_score(verdicts: list[ClaimVerdict]) -> float:
    """
    Compute aggregate consistency score from individual verdicts.

    Weighting:
        Entailment   → positive contribution
        Contradiction → strong negative contribution
        Neutral       → slight negative (neither supports nor contradicts)

    Score in [0, 1]:
        1.0 = all verifiable claims entailed by context
        0.0 = all verifiable claims contradicted by context
        ~0.5 = all neutral (no clear signal)

    Claims with higher verification_priority (lower number) weighted more.
    """
    if not verdicts:
        return 0.5

    total_weight = 0.0
    score_sum    = 0.0

    for v in verdicts:
        # Weight by confidence in verdict
        weight = max(0.1, v.confidence)

        if v.verdict == ENTAILMENT:
            score_sum    += weight * 1.0
        elif v.verdict == CONTRADICTION:
            score_sum    += weight * 0.0   # contradiction = 0 contribution
        else:
            score_sum    += weight * 0.4   # neutral = partial contribution

        total_weight += weight

    if total_weight == 0:
        return 0.5

    return round(score_sum / total_weight, 4)


# =============================================================================
# MAIN VERIFICATION FUNCTION
# =============================================================================

def verify_claims(
    extraction: ExtractionResult,
    context: str,
    max_claims: int = 5,
) -> VerificationResult:
    """
    Verify extracted claims against retrieved context.

    Args:
        extraction  : ExtractionResult from claim_extractor.py
        context     : retrieved context string from Omega retrieval
        max_claims  : maximum claims to verify (rate limit / latency control)

    Returns:
        VerificationResult with per-claim verdicts and aggregate score.

    Only verifiable claims (CLASS_A and CLASS_B) are sent to the NLI model.
    CLASS_C claims are recorded as neutral without verification.

    Verification order follows claim priority — highest priority claims
    (specific entity claims with high model confidence) verified first.
    """
    result = VerificationResult()

    if not context or not context.strip():
        result.method = "skipped"
        result.consistency_score = 0.4  # No context = conservative score
        return result

    # Load NLI model if not yet loaded
    nli_loaded = _load_nli_model()
    verify_fn  = _verify_with_nli if nli_loaded else _verify_with_heuristic
    result.method = "nli_model" if nli_loaded else "keyword_heuristic"

    # Verify only verifiable claims, capped at max_claims
    claims_to_verify = extraction.verifiable_claims[:max_claims]

    verdicts = []
    verified_claims     = []
    contradicted_claims = []

    for claim in claims_to_verify:
        verdict_str, confidence, ent, contra, neutral = verify_fn(
            claim.text, context
        )

        cv = ClaimVerdict(
            claim_text          = claim.text,
            context_used        = context[:200],
            verdict             = verdict_str,
            confidence          = confidence,
            entailment_score    = ent,
            contradiction_score = contra,
            neutral_score       = neutral,
            method              = result.method,
            claim_class         = claim.claim_class,
        )
        verdicts.append(cv)

        if verdict_str == ENTAILMENT:
            result.entailment_count += 1
            verified_claims.append(claim.text[:100])
        elif verdict_str == CONTRADICTION:
            result.contradiction_count += 1
            contradicted_claims.append(claim.text[:100])
        else:
            result.neutral_count += 1

    # Add CLASS_C claims as neutral without verification
    for claim in extraction.class_c_claims:
        cv = ClaimVerdict(
            claim_text          = claim.text,
            context_used        = "",
            verdict             = NEUTRAL,
            confidence          = 0.5,
            entailment_score    = 0.0,
            contradiction_score = 0.0,
            neutral_score       = 1.0,
            method              = "not_verifiable",
            claim_class         = CLASS_C,
        )
        verdicts.append(cv)
        result.neutral_count += 1

    result.verdicts             = verdicts
    result.has_contradiction    = result.contradiction_count > 0
    result.verified_claims      = verified_claims
    result.contradicted_claims  = contradicted_claims
    result.consistency_score    = _compute_consistency_score(
        [v for v in verdicts if v.method != "not_verifiable"]
    )

    return result


def run_claim_verifier(
    extraction: ExtractionResult,
    context: str,
    max_claims: int = 5,
) -> VerificationResult:
    """Public interface. Never raises."""
    try:
        return verify_claims(extraction, context, max_claims=max_claims)
    except Exception:
        r = VerificationResult()
        r.method = "error"
        r.consistency_score = 0.4
        return r
