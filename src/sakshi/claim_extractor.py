"""
claim_extractor.py — Sakshi-Protocol V3
Extract and classify verifiable claims from model output.

Phase 2 of the V3 Omega redesign.

V2 Omega sent the raw output to the generator model for self-verification.
This is the self-verification problem — the model that generated the claim
is the same model evaluating it.

Phase 2 replaces self-verification with a three-stage independent
verification pipeline:
    1. claim_extractor.py   — extract and classify claims (this file)
    2. claim_verifier.py    — NLI verification against retrieved context
    3. completeness_scorer.py — Adhyasa detection on non-existent entities

This file handles Step 1: claim extraction and classification.

Claim taxonomy (maps to three-class hallucination taxonomy):
    CLASS_A — entity claims: about named entities that may or may not exist
              Addressable: entity existence checking + NLI verification
              Example: "Neurodex is FDA-approved for Parkinson's disease"

    CLASS_B — factual claims: about real entities, specific facts may be wrong
              Partially addressable: NLI contradiction detection
              Example: "The Inca empire was founded in 1438"

    CLASS_C — abstract claims: no external referent for verification
              Not externally addressable: internal signals only
              Example: "Education is the foundation of democracy"

Design principle: no language model in the extraction loop.
spaCy is used for linguistic analysis only — it does not generate
or evaluate content. This preserves the epistemic independence of
the verification layer from the generator.

Paper correspondence:
    Claim extraction as prerequisite for Anumana (NLI inference)
    and Adhyasa detection (structural completeness scoring).
"""

import re
from dataclasses import dataclass, field
from typing import Optional

try:
    import spacy
    _NLP = spacy.load("en_core_web_sm")
    _SPACY_AVAILABLE = True
except Exception:
    _NLP = None
    _SPACY_AVAILABLE = False


# =============================================================================
# CLAIM CLASSES
# =============================================================================

CLASS_A = "entity_claim"      # Named entity — existence verifiable
CLASS_B = "factual_claim"     # Real entity, specific fact — partially verifiable
CLASS_C = "abstract_claim"    # No external referent — internal signals only


# =============================================================================
# CLAIM
# =============================================================================

@dataclass
class Claim:
    """
    A single verifiable claim extracted from the output.

    Attributes:
        text            : the claim text as extracted
        claim_class     : CLASS_A | CLASS_B | CLASS_C
        entities        : named entities present in the claim
        is_specific     : True if claim contains specific figures, dates, names
        is_temporal     : True if claim references a current state
        confidence      : model's apparent confidence in this claim (0-1)
                          derived from linguistic markers, not model internals
        verifiable      : True if external verification is possible
        verification_priority: 0 (highest) to 3 (lowest) — determines
                          which claims are verified first when rate-limited
    """
    text:                  str
    claim_class:           str   = CLASS_C
    entities:              list  = field(default_factory=list)
    is_specific:           bool  = False
    is_temporal:           bool  = False
    confidence:            float = 0.5
    verifiable:            bool  = False
    verification_priority: int   = 3

    def to_dict(self) -> dict:
        return {
            "text":                  self.text,
            "claim_class":           self.claim_class,
            "entities":              self.entities,
            "is_specific":           self.is_specific,
            "is_temporal":           self.is_temporal,
            "confidence":            round(self.confidence, 4),
            "verifiable":            self.verifiable,
            "verification_priority": self.verification_priority,
        }


# =============================================================================
# EXTRACTION RESULT
# =============================================================================

@dataclass
class ExtractionResult:
    """
    Claims extracted from a single output.

    Attributes:
        claims          : all extracted claims
        class_a_claims  : entity claims (highest priority for verification)
        class_b_claims  : factual claims (medium priority)
        class_c_claims  : abstract claims (not externally verifiable)
        verifiable_claims: claims routed to verification pipeline
        entity_names    : unique entity names across all claims
        has_high_specificity: True if any claim contains exact figures/dates
        extraction_method: "spacy" | "regex" | "sentence"
    """
    claims:               list = field(default_factory=list)
    class_a_claims:       list = field(default_factory=list)
    class_b_claims:       list = field(default_factory=list)
    class_c_claims:       list = field(default_factory=list)
    verifiable_claims:    list = field(default_factory=list)
    entity_names:         list = field(default_factory=list)
    has_high_specificity: bool = False
    extraction_method:    str  = "sentence"

    def to_dict(self) -> dict:
        return {
            "claims":               [c.to_dict() for c in self.claims],
            "class_a_count":        len(self.class_a_claims),
            "class_b_count":        len(self.class_b_claims),
            "class_c_count":        len(self.class_c_claims),
            "verifiable_count":     len(self.verifiable_claims),
            "entity_names":         self.entity_names,
            "has_high_specificity": self.has_high_specificity,
            "extraction_method":    self.extraction_method,
        }


# =============================================================================
# SIGNAL PATTERNS
# Used to classify claims without a language model
# =============================================================================

# Patterns indicating high specificity — exact figures, dates, measurements
_SPECIFICITY_PATTERNS = [
    r"\b\d+\.?\d*\s*(?:mg|ml|kg|%|percent|years?|months?|days?)\b",
    r"\b(?:in\s+)?\d{4}\b",                          # years
    r"\$\d+(?:\.\d+)?(?:\s*(?:million|billion))?\b",  # dollar amounts
    r"\b\d+(?:\.\d+)?\s*(?:times|fold|x)\b",
    r"\bp\s*[<>=]\s*0\.\d+\b",                        # p-values
    r"\barticle\s+\d+|section\s+\d+\b",               # legal references
    r"\b(?:chapter|verse|shloka)\s+\d+\b",            # text references
]
_SPECIFICITY_RE = [re.compile(p, re.IGNORECASE) for p in _SPECIFICITY_PATTERNS]

# Patterns indicating temporal sensitivity
_TEMPORAL_PATTERNS = [
    r"\b(current(ly)?|now|today|at\s+present|presently)\b",
    r"\b(latest|most\s+recent|as\s+of\s+\d{4})\b",
    r"\b(recently|just|newly)\s+(approved|published|announced|released)\b",
]
_TEMPORAL_RE = [re.compile(p, re.IGNORECASE) for p in _TEMPORAL_PATTERNS]

# Confidence markers — assertive phrasing in individual claims
_HIGH_CONFIDENCE_MARKERS = [
    r"\b(is|are|was|were)\s+(approved|established|confirmed|proven)\b",
    r"\b(clearly|definitely|certainly|undoubtedly)\b",
    r"\b(has\s+been|have\s+been)\s+(shown|demonstrated|confirmed)\b",
    r"\bthe\s+(study|research|evidence)\s+(shows|demonstrates|confirms)\b",
]
_HIGH_CONF_RE = [re.compile(p, re.IGNORECASE) for p in _HIGH_CONFIDENCE_MARKERS]

# Hedging markers — uncertain phrasing
_HEDGING_MARKERS = [
    r"\b(may|might|could|possibly|perhaps|probably)\b",
    r"\b(i\s+(?:think|believe|suspect)|it\s+(?:seems|appears))\b",
    r"\b(uncertain|unclear|unknown|unconfirmed|unverified)\b",
    r"\b(i\s+(?:cannot|can't|am\s+not\s+sure|don't\s+know))\b",
]
_HEDGING_RE = [re.compile(p, re.IGNORECASE) for p in _HEDGING_MARKERS]

# Abstract claim indicators — no external referent
_ABSTRACT_INDICATORS = [
    r"\b(should|ought\s+to|must|need\s+to)\b",
    r"\b(important|essential|crucial|vital|necessary)\b",
    r"\b(purpose|meaning|value|significance|role)\s+of\b",
    r"\b(in\s+general|broadly|typically|often|usually)\b",
    r"\b(philosophy|ethics|morality|justice|freedom|democracy)\b",
]
_ABSTRACT_RE = [re.compile(p, re.IGNORECASE) for p in _ABSTRACT_INDICATORS]


# =============================================================================
# SENTENCE SPLITTING
# =============================================================================

def _split_sentences(text: str) -> list[str]:
    """
    Split output into sentences for claim extraction.
    Uses spaCy sentence boundary detection when available,
    falls back to regex splitting.
    """
    if not text or not text.strip():
        return []

    # Clean the text
    text = text.strip()

    if _SPACY_AVAILABLE and _NLP is not None:
        try:
            doc = _NLP(text[:5000])  # Cap at 5000 chars for performance
            sentences = [sent.text.strip() for sent in doc.sents
                        if sent.text.strip() and len(sent.text.split()) > 3]
            return sentences
        except Exception:
            pass

    # Regex fallback
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip() and len(s.split()) > 3]


# =============================================================================
# ENTITY EXTRACTION FROM CLAIM
# =============================================================================

def _extract_claim_entities(text: str) -> list[str]:
    """Extract named entities from a single claim sentence."""
    entities = []

    if _SPACY_AVAILABLE and _NLP is not None:
        try:
            doc = _NLP(text)
            for ent in doc.ents:
                if ent.label_ not in ("DATE", "TIME", "PERCENT", "MONEY",
                                       "QUANTITY", "ORDINAL", "CARDINAL"):
                    if len(ent.text.strip()) > 2:
                        entities.append(ent.text.strip())
            return entities
        except Exception:
            pass

    # Regex fallback — capitalised multi-word sequences
    for match in re.finditer(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", text):
        entities.append(match.group())

    return list(set(entities))


# =============================================================================
# CLAIM CLASSIFICATION
# =============================================================================

def _classify_claim(sentence: str, entities: list) -> tuple[str, float, bool, bool]:
    """
    Classify a sentence as CLASS_A, CLASS_B, or CLASS_C.

    Returns (claim_class, confidence, is_specific, is_temporal).

    Classification logic:
        CLASS_A: has named entities AND makes specific claims about them
        CLASS_B: has named entities OR specific facts, but about clearly real entities
        CLASS_C: no named entities, abstract, or heavily hedged
    """
    text_lower = sentence.lower()

    # Specificity check
    is_specific = any(p.search(sentence) for p in _SPECIFICITY_RE)

    # Temporal check
    is_temporal = any(p.search(sentence) for p in _TEMPORAL_RE)

    # Confidence estimation from linguistic markers
    high_conf_hits = sum(1 for p in _HIGH_CONF_RE if p.search(sentence))
    hedging_hits   = sum(1 for p in _HEDGING_RE   if p.search(sentence))
    abstract_hits  = sum(1 for p in _ABSTRACT_RE  if p.search(sentence))

    # Base confidence
    if hedging_hits > 0:
        confidence = max(0.2, 0.5 - hedging_hits * 0.15)
    elif high_conf_hits > 0:
        confidence = min(0.9, 0.6 + high_conf_hits * 0.1)
    else:
        confidence = 0.5

    # Classification
    if len(entities) > 0 and (is_specific or high_conf_hits > 0):
        # Has entities + specificity or assertive confidence = Class A
        return CLASS_A, confidence, is_specific, is_temporal

    elif len(entities) > 0 and abstract_hits == 0:
        # Has entities but less specific = Class B
        return CLASS_B, confidence, is_specific, is_temporal

    elif abstract_hits > 0 or (hedging_hits > 0 and len(entities) == 0):
        # Abstract or heavily hedged = Class C
        return CLASS_C, max(0.2, confidence - 0.2), is_specific, is_temporal

    else:
        # Default to Class B for sentences with some content
        return CLASS_B, confidence, is_specific, is_temporal


# =============================================================================
# PRIORITY ASSIGNMENT
# =============================================================================

def _assign_priority(claim: Claim) -> int:
    """
    Assign verification priority (0=highest, 3=lowest).

    Priority 0: Class A claims with high confidence and specificity
    Priority 1: Class A claims, Class B with specificity
    Priority 2: Class B claims
    Priority 3: Class C claims (not externally verifiable)
    """
    if claim.claim_class == CLASS_C:
        return 3
    if claim.claim_class == CLASS_A and claim.confidence > 0.7 and claim.is_specific:
        return 0
    if claim.claim_class == CLASS_A:
        return 1
    if claim.claim_class == CLASS_B and claim.is_specific:
        return 1
    return 2


# =============================================================================
# MAIN EXTRACTION FUNCTION
# =============================================================================

def extract_claims(
    output: str,
    prompt: str = "",
    prompt_type: str = "",
    max_claims: int = 10,
) -> ExtractionResult:
    """
    Extract and classify verifiable claims from model output.

    Args:
        output      : the model output to extract claims from
        prompt      : the original prompt (used for context)
        prompt_type : pipeline taxonomy type
        max_claims  : maximum claims to extract and classify

    Returns:
        ExtractionResult with classified claims and routing information.

    No language model is used in this function. All classification
    is performed through linguistic pattern matching and spaCy NER.
    This preserves the epistemic independence of the verification layer.
    """
    result = ExtractionResult()

    if not output or not output.strip():
        return result

    # Split into sentences
    sentences = _split_sentences(output)
    if not sentences:
        return result

    result.extraction_method = "spacy" if _SPACY_AVAILABLE else "regex"

    # Cap at max_claims
    sentences = sentences[:max_claims]

    all_entities = set()
    claims       = []

    for sentence in sentences:
        # Extract entities from this sentence
        entities = _extract_claim_entities(sentence)
        all_entities.update(entities)

        # Classify the claim
        claim_class, confidence, is_specific, is_temporal = _classify_claim(
            sentence, entities
        )

        claim = Claim(
            text        = sentence,
            claim_class = claim_class,
            entities    = entities,
            is_specific = is_specific,
            is_temporal = is_temporal,
            confidence  = confidence,
            verifiable  = claim_class in (CLASS_A, CLASS_B),
        )
        claim.verification_priority = _assign_priority(claim)
        claims.append(claim)

    # Sort by verification priority
    claims.sort(key=lambda c: c.verification_priority)

    result.claims            = claims
    result.class_a_claims    = [c for c in claims if c.claim_class == CLASS_A]
    result.class_b_claims    = [c for c in claims if c.claim_class == CLASS_B]
    result.class_c_claims    = [c for c in claims if c.claim_class == CLASS_C]
    result.verifiable_claims = [c for c in claims if c.verifiable]
    result.entity_names      = sorted(all_entities)
    result.has_high_specificity = any(c.is_specific for c in claims)

    return result


def run_claim_extractor(
    output: str,
    prompt: str = "",
    prompt_type: str = "",
) -> ExtractionResult:
    """Public interface. Never raises."""
    try:
        return extract_claims(output, prompt=prompt, prompt_type=prompt_type)
    except Exception:
        return ExtractionResult()
