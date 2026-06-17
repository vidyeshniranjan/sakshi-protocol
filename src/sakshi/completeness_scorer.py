"""
completeness_scorer.py — Sakshi-Protocol V3
Structural completeness scoring for Adhyasa detection.

Phase 2 of the V3 Omega redesign.

Adhyasa — superimposition — is the mechanism by which the model generates
structured, plausible, template-complete fabrications about entities that
do not exist. The model superimposes the template of a real entity onto
a fabricated name, filling in expected information categories from
parametric knowledge of similar real entities.

The key insight: a fabricated entity produces a suspiciously complete
response precisely because the model has learned what a complete response
about a real entity of that type looks like. High structural completeness
about a confirmed non-existent entity is the Adhyasa signal.

This inverts the standard distortion logic:
    Standard distortion: fluency + specificity → lower distortion → accept
    Adhyasa detection: fluency + completeness + non-existent entity → REJECT

When does this fire:
    - Entity was confirmed non-existent by entity_screener.py
    - Output is structurally complete (covers expected information categories)
    - Output is fluent and confident (low surface distortion)

Expected information categories per entity type:
    drug:
        mechanism_of_action, approved_indications, dosage, side_effects,
        contraindications, drug_class, manufacturer
    paper/study:
        authors, institution, year, journal, methodology, findings, conclusions
    person_academic:
        affiliation, research_area, publications, credentials
    legal_case:
        court, year, parties, holding, reasoning, precedent_modified
    company:
        industry, founded, headquarters, products, revenue, leadership
    general:
        definition, characteristics, examples, context

Paper correspondence:
    Adhyasa — superimposition. The model projects the structural template
    of known entities onto fabricated names. Structural completeness on
    a confirmed non-existent entity is evidence of superimposition.
"""

import re
from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# INFORMATION CATEGORY PATTERNS
# Per entity type — what categories of information does a complete
# response about a real entity of this type typically contain?
# =============================================================================

_CATEGORY_PATTERNS = {
    "drug": {
        "mechanism_of_action": [
            r"\b(mechanism|works?\s+by|acts?\s+on|inhibit|agonist|antagonist|receptor)\b",
            r"\b(blocks?|binds?\s+to|targets?|pathway|neurotransmitter)\b",
        ],
        "approved_indications": [
            r"\b(approved|indicated|used\s+(?:for|to\s+treat)|treatment\s+of)\b",
            r"\b(FDA[\-\s]approved|licensed|authorised)\b",
        ],
        "dosage": [
            r"\b(dose|dosage|mg|milligram|daily|twice|regimen|administration)\b",
            r"\b(\d+\s*mg|\d+\s*ml|oral|intravenous|subcutaneous)\b",
        ],
        "side_effects": [
            r"\b(side\s+effects?|adverse|contraindicated|warning|risk)\b",
            r"\b(nausea|headache|dizziness|fatigue|common\s+reactions?)\b",
        ],
        "drug_class": [
            r"\b(class|category|type\s+of\s+drug|belongs\s+to|family\s+of)\b",
            r"\b(SSRI|SNRI|beta[\-\s]blocker|ACE\s+inhibitor|statin|antibiotic)\b",
        ],
    },
    "paper_academic": {
        "authors": [
            r"\b(author|researcher|conducted\s+by|written\s+by|team)\b",
            r"\b(et\s+al\.|colleagues|university|institute)\b",
        ],
        "methodology": [
            r"\b(study|method|design|sample|participants|subjects|experiment)\b",
            r"\b(randomised|controlled|cohort|longitudinal|cross[\-\s]sectional)\b",
        ],
        "findings": [
            r"\b(found|showed|demonstrated|revealed|concluded|results?)\b",
            r"\b(significant|p[\s<>=]+0\.\d+|effect\s+size|correlation)\b",
        ],
        "publication": [
            r"\b(published|journal|proceedings|conference|peer[\-\s]reviewed)\b",
            r"\b(Nature|Science|Lancet|NEJM|JAMA|Cell|doi)\b",
        ],
    },
    "legal_case": {
        "parties": [
            r"\b(plaintiff|defendant|appellant|respondent|petitioner|v\.)\b",
        ],
        "holding": [
            r"\b(held|ruled|decided|found|concluded|judgment|majority)\b",
            r"\b(affirmed|reversed|remanded|dismissed|upheld)\b",
        ],
        "court": [
            r"\b(Supreme\s+Court|Circuit|District|Court\s+of\s+Appeals|High\s+Court)\b",
        ],
        "reasoning": [
            r"\b(reasoning|rationale|opinion|basis|grounds|because|therefore)\b",
        ],
        "precedent": [
            r"\b(precedent|overruled|distinguished|modified|established)\b",
        ],
    },
    "person_academic": {
        "affiliation": [
            r"\b(professor|university|institute|department|faculty|chair)\b",
        ],
        "research_area": [
            r"\b(research|specialises?|focuses?\s+on|field|area|work)\b",
        ],
        "publications": [
            r"\b(published|paper|book|journal|authored|co[\-\s]authored)\b",
        ],
    },
    "general": {
        "definition": [
            r"\b(is\s+(?:a|an|the)|refers?\s+to|defined\s+as|means?)\b",
        ],
        "characteristics": [
            r"\b(characterised|features?|properties|attributes|aspects?)\b",
        ],
        "context": [
            r"\b(used\s+in|found\s+in|common\s+in|associated\s+with|related\s+to)\b",
        ],
    },
}

# Default to general if entity type not in map
_CATEGORY_PATTERNS["organisation"]   = _CATEGORY_PATTERNS["general"]
_CATEGORY_PATTERNS["location"]       = _CATEGORY_PATTERNS["general"]
_CATEGORY_PATTERNS["person"]         = _CATEGORY_PATTERNS["person_academic"]
_CATEGORY_PATTERNS["paper_medical"]  = _CATEGORY_PATTERNS["paper_academic"]
_CATEGORY_PATTERNS["company"]        = {
    "industry": [r"\b(industry|sector|field|business|company|firm)\b"],
    "products":  [r"\b(product|service|offer|provide|develop|manufacture)\b"],
    "leadership":[r"\b(CEO|founder|president|chairman|executive|management)\b"],
}


# =============================================================================
# COMPLETENESS RESULT
# =============================================================================

@dataclass
class CompletenessResult:
    """
    Structural completeness score for a single output.

    Attributes:
        completeness_score  : float [0, 1]
                              0.0 = no expected categories present
                              1.0 = all expected categories present
        categories_present  : list of category names found in output
        categories_expected : list of all expected categories for entity type
        adhyasa_signal      : bool — True when completeness is high AND
                              entity was confirmed non-existent
        adhyasa_score       : float [0, 1] — strength of Adhyasa signal
                              0.0 = no superimposition evidence
                              1.0 = strong superimposition evidence
        entity_type         : entity type used for category selection
        entity_confirmed_absent: bool — from entity screener
    """
    completeness_score:      float = 0.0
    categories_present:      list  = field(default_factory=list)
    categories_expected:     list  = field(default_factory=list)
    adhyasa_signal:          bool  = False
    adhyasa_score:           float = 0.0
    entity_type:             str   = "general"
    entity_confirmed_absent: bool  = False

    def to_dict(self) -> dict:
        return {
            "completeness_score":      round(self.completeness_score, 4),
            "categories_present":      self.categories_present,
            "categories_expected":     self.categories_expected,
            "adhyasa_signal":          self.adhyasa_signal,
            "adhyasa_score":           round(self.adhyasa_score, 4),
            "entity_type":             self.entity_type,
            "entity_confirmed_absent": self.entity_confirmed_absent,
        }


# =============================================================================
# SCORING
# =============================================================================

def _score_completeness(output: str, entity_type: str) -> tuple[float, list, list]:
    """
    Score structural completeness of output for the given entity type.

    Returns (completeness_score, categories_present, categories_expected).
    """
    patterns = _CATEGORY_PATTERNS.get(entity_type, _CATEGORY_PATTERNS["general"])
    output_lower = output.lower()

    categories_expected = list(patterns.keys())
    categories_present  = []

    for category, category_patterns in patterns.items():
        # Category is "present" if any pattern matches
        for pattern in category_patterns:
            if re.search(pattern, output_lower, re.IGNORECASE):
                categories_present.append(category)
                break

    if not categories_expected:
        return 0.0, [], []

    completeness = len(categories_present) / len(categories_expected)
    return round(completeness, 4), categories_present, categories_expected


def _compute_adhyasa_score(
    completeness_score: float,
    entity_confirmed_absent: bool,
    output_length: int,
    has_confidence_markers: bool,
) -> float:
    """
    Compute the Adhyasa score.

    Adhyasa is strong when:
        - Entity confirmed absent (existence checking failed)
        - Completeness is high (many expected categories present)
        - Output is substantial (not a brief hedge)
        - Output contains confident assertive phrasing

    Score in [0, 1] where:
        0.0 = no Adhyasa evidence
        1.0 = strong superimposition evidence
    """
    if not entity_confirmed_absent:
        return 0.0

    # Base score from completeness — high completeness on absent entity
    # is the core Adhyasa signal
    base = completeness_score

    # Amplify if output is substantial (not a brief hedge)
    # A brief "I don't know" has low completeness and short length
    # Adhyasa produces long, structured, complete-looking responses
    length_factor = min(output_length / 200, 1.0)

    # Amplify if confident phrasing present
    # Adhyasa produces confident-sounding fabrications
    confidence_factor = 1.2 if has_confidence_markers else 1.0

    score = min(base * length_factor * confidence_factor, 1.0)
    return round(score, 4)


# =============================================================================
# CONFIDENCE MARKER DETECTION
# =============================================================================

_CONFIDENCE_PATTERNS = [
    r"\b(is|are|was|were)\s+(approved|established|indicated|used)\b",
    r"\b(primarily|typically|commonly|generally)\s+used\b",
    r"\b(the\s+recommended|standard\s+treatment|first[\-\s]line)\b",
    r"\b(has\s+been|have\s+been)\s+(shown|demonstrated|approved)\b",
]
_CONFIDENCE_RE = [re.compile(p, re.IGNORECASE) for p in _CONFIDENCE_PATTERNS]


def _has_confidence_markers(output: str) -> bool:
    return any(p.search(output) for p in _CONFIDENCE_RE)


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def score_completeness(
    output: str,
    entity_type: str = "general",
    entity_confirmed_absent: bool = False,
) -> CompletenessResult:
    """
    Score structural completeness of output for Adhyasa detection.

    Args:
        output                  : the model output to score
        entity_type             : type of entity the output is about
                                  (from entity_screener classification)
        entity_confirmed_absent : True if entity_screener confirmed
                                  the entity does not exist in databases

    Returns:
        CompletenessResult with completeness score and Adhyasa signal.

    This function fires after entity existence checking has confirmed
    a named entity does not exist. It then measures how structurally
    complete the output about that non-existent entity is.

    High completeness + confirmed absence = Adhyasa signal.
    The model has superimposed a real entity template onto a fabricated name.
    """
    result = CompletenessResult()
    result.entity_type             = entity_type
    result.entity_confirmed_absent = entity_confirmed_absent

    if not output or not output.strip():
        return result

    # Score structural completeness
    completeness, present, expected = _score_completeness(output, entity_type)
    result.completeness_score  = completeness
    result.categories_present  = present
    result.categories_expected = expected

    # Compute Adhyasa score
    output_length        = len(output.split())
    has_confidence       = _has_confidence_markers(output)
    adhyasa_score        = _compute_adhyasa_score(
        completeness, entity_confirmed_absent, output_length, has_confidence
    )

    result.adhyasa_score  = adhyasa_score
    result.adhyasa_signal = adhyasa_score > 0.3

    return result


def run_completeness_scorer(
    output: str,
    entity_type: str = "general",
    entity_confirmed_absent: bool = False,
) -> CompletenessResult:
    """Public interface. Never raises."""
    try:
        return score_completeness(output, entity_type, entity_confirmed_absent)
    except Exception:
        return CompletenessResult()
