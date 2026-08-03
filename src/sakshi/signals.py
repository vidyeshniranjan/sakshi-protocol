import re
import numpy as np
from openai import OpenAI

# Lazily-initialised OpenAI client.
# Created on first use rather than at import time, so that importing this
# module (e.g. for computational-only signals) does not require an OpenAI
# key to be present. The key is read from the environment when first needed.
_client = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI()
    return _client

# =============================================================================
# signals.py — Sakshi-Protocol V3 signal extraction
#
# Architecture: hybrid computational + API
#
# Computational signals (all prompt types, no extra API calls):
#   - semantic_similarity   : embedding cosine similarity, prompt vs output
#   - uncertainty           : linguistically broad uncertainty detection
#   - confidence            : assertive phrasing detection
#   - coherence             : lexical type-token ratio
#   - specificity           : length-normalised output density
#
# API signals (reasoning prompt type only):
#   - step_consistency      : are reasoning steps internally consistent?
#   - contradiction_score   : does the output contradict itself?
#
# Design principle: the observer (Sakshi) is structurally separate from the
# generator. Computational signals preserve this separation entirely.
# API signals are used only where computation genuinely cannot do the job —
# detecting logical inconsistency in multi-step reasoning outputs.
# =============================================================================


# -----------------------------------------------------------------------------
# EMBEDDINGS
# -----------------------------------------------------------------------------

def get_embedding(text: str) -> np.ndarray:
    response = _get_client().embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# -----------------------------------------------------------------------------
# UNCERTAINTY DETECTION — computational
#
# V2 used 8 exact keyword phrases, missing paraphrases and negated certainty.
# V3 uses:
#   - a broader phrase list covering more hedging patterns
#   - regex patterns for negated certainty ("not certain", "can't be sure")
#   - normalised by output length to avoid penalising longer outputs unfairly
# -----------------------------------------------------------------------------

# Direct hedging phrases
_UNCERTAINTY_PHRASES = [
    "i don't know", "i do not know",
    "i'm not sure", "i am not sure",
    "i'm uncertain", "i am uncertain",
    "i cannot confirm", "i can't confirm",
    "i cannot verify", "i can't verify",
    "i'm unable to verify", "i am unable to verify",
    "i do not have reliable", "i don't have reliable",
    "not well documented", "poorly documented",
    "no evidence", "lack of evidence",
    "no reliable evidence", "limited evidence",
    "unclear", "it is unclear", "remains unclear",
    "it is uncertain", "this is uncertain",
    "as of my last knowledge", "as of my knowledge cutoff",
    "it is possible", "it may be", "it might be",
    "may not be", "might not be",
    "i cannot say for certain", "i can't say for certain",
    "i'm not confident", "i am not confident",
    "i would need to verify", "would need verification",
    "unverified", "unconfirmed",
    "to my knowledge", "as far as i know",
    "i believe but am not certain",
]

# Regex patterns for negated certainty constructions
_UNCERTAINTY_PATTERNS = [
    r"not\s+(entirely\s+)?certain",
    r"not\s+(completely\s+)?sure",
    r"can'?t\s+be\s+sure",
    r"hard\s+to\s+(say|know|confirm|verify)",
    r"difficult\s+to\s+(confirm|verify|determine)",
    r"no\s+(clear|definitive|reliable)\s+(answer|evidence|information|data)",
    r"(little|limited|scarce)\s+(evidence|data|information)",
    r"(may|might|could)\s+(not\s+)?(be\s+)?(accurate|correct|reliable|true)",
]

_UNCERTAINTY_RE = [re.compile(p, re.IGNORECASE) for p in _UNCERTAINTY_PATTERNS]


def uncertainty_score(output: str) -> float:
    """
    Returns a score in [0, 1] reflecting the degree of epistemic hedging
    in the output. Higher = more uncertain.

    Combines phrase matching and regex pattern matching.
    Normalised by output length to avoid length bias.
    """
    if not output or not output.strip():
        return 0.0

    output_lower = output.lower()
    words        = output.split()
    n_words      = max(len(words), 1)

    # Phrase hits
    phrase_hits = sum(1 for p in _UNCERTAINTY_PHRASES if p in output_lower)

    # Regex hits
    pattern_hits = sum(1 for r in _UNCERTAINTY_RE if r.search(output))

    total_hits = phrase_hits + pattern_hits

    # Normalise: 1 hit per 50 words = score of ~0.5; saturates at 3+ hits
    # Short outputs are not penalised for having fewer hits
    length_factor = min(n_words / 50, 1.0)
    raw = total_hits / 3.0
    score = min(raw * (0.5 + 0.5 * length_factor), 1.0)

    return round(score, 6)


# -----------------------------------------------------------------------------
# CONFIDENCE DETECTION — computational
#
# V2 used 10 exact phrases, saturating to 1.0 too easily.
# V3 uses a broader phrase list and regex for assertive constructions,
# with the same length-normalised scoring to avoid saturation.
# -----------------------------------------------------------------------------

_CONFIDENCE_PHRASES = [
    "clearly", "definitely", "certainly", "absolutely", "undoubtedly",
    "without doubt", "without question", "unquestionably",
    "indeed", "in fact", "as a matter of fact",
    "it is proven", "it has been proven", "proven",
    "it is established", "it has been established", "well established",
    "the study shows", "the study found", "the study demonstrates",
    "results indicate", "results show", "results demonstrate",
    "research confirms", "research shows", "research demonstrates",
    "evidence shows", "evidence indicates", "evidence confirms",
    "data shows", "data indicates", "data confirms",
    "experts agree", "scientists agree", "consensus shows",
    "it is known", "it is well known", "widely known",
    "conclusively", "conclusive evidence",
    "has been shown", "has been demonstrated", "has been confirmed",
    "according to a study", "a study published", "studies have shown",
    "a report found", "reports confirm",
    "statistics show", "statistics indicate",
    "the evidence clearly", "the evidence strongly",
]

_CONFIDENCE_PATTERNS = [
    r"(is|are|was|were)\s+(clearly|definitively|certainly|proven\s+to\s+be)",
    r"there\s+is\s+no\s+doubt",
    r"it\s+is\s+(a\s+fact|factual|confirmed|established)",
    r"(strong|compelling|overwhelming)\s+evidence",
    r"(unanimously|universally)\s+(agreed|accepted|confirmed)",
]

_CONFIDENCE_RE = [re.compile(p, re.IGNORECASE) for p in _CONFIDENCE_PATTERNS]


def confidence_score(output: str) -> float:
    """
    Returns a score in [0, 1] reflecting the degree of assertive, confident
    phrasing in the output. Higher = more confident/assertive.

    Used as a component of hallucination_risk in state.py:
    confident + specific + unhedged = elevated fabrication risk.
    """
    if not output or not output.strip():
        return 0.0

    output_lower = output.lower()
    words        = output.split()
    n_words      = max(len(words), 1)

    phrase_hits  = sum(1 for p in _CONFIDENCE_PHRASES if p in output_lower)
    pattern_hits = sum(1 for r in _CONFIDENCE_RE if r.search(output))

    total_hits = phrase_hits + pattern_hits

    length_factor = min(n_words / 50, 1.0)
    raw   = total_hits / 4.0
    score = min(raw * (0.5 + 0.5 * length_factor), 1.0)

    return round(score, 6)


# -----------------------------------------------------------------------------
# SPECIFICITY — computational
# Length-normalised output density. Unchanged from V2 — this signal is honest.
# -----------------------------------------------------------------------------

def specificity_score(output: str) -> float:
    words = output.split() if output else []
    return round(min(len(words) / 150, 1.0), 6)


# -----------------------------------------------------------------------------
# REASONING SIGNALS — API only, reasoning prompt type
#
# These fire only when prompt_type == "reasoning". They use a single
# structured API call per output to extract two scores:
#
#   step_consistency  : are the reasoning steps logically consistent?
#                       1.0 = fully consistent, 0.0 = contradictory
#
#   contradiction_score : does the output contradict itself at any point?
#                         0.0 = no contradiction, 1.0 = direct contradiction
#
# The model is prompted to return only a JSON object with two float fields.
# Parsing is defensive — defaults to neutral values on any failure.
# -----------------------------------------------------------------------------

_REASONING_EVAL_PROMPT = """You are an evaluator assessing the internal logical quality of a reasoning output.

Given the following question and response, evaluate two properties:

1. step_consistency (float 0.0 to 1.0):
   How consistent are the reasoning steps with each other?
   1.0 = all steps follow logically, no gaps or jumps
   0.5 = some steps are unclear or weakly connected
   0.0 = steps are contradictory or logically broken

2. contradiction_score (float 0.0 to 1.0):
   Does the output contradict itself at any point?
   0.0 = no contradiction
   0.5 = partial or implicit contradiction
   1.0 = direct explicit contradiction

Return ONLY a JSON object with exactly these two keys and float values.
Do not include any explanation, preamble, or markdown.

Example output:
{{"step_consistency": 0.85, "contradiction_score": 0.1}}

Question: {prompt}

Response: {output}"""


def reasoning_signals(prompt: str, output: str) -> dict:
    """
    Calls the OpenAI API to evaluate reasoning quality.
    Returns step_consistency and contradiction_score as floats in [0, 1].
    Defaults to neutral values on any failure.
    """
    neutral = {"step_consistency": 0.7, "contradiction_score": 0.1}

    try:
        response = _get_client().chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": _REASONING_EVAL_PROMPT.format(
                    prompt=prompt,
                    output=output
                )
            }],
            temperature=0.0,
            max_tokens=60
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if present
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()

        import json
        parsed = json.loads(raw)

        step_consistency    = float(parsed.get("step_consistency", 0.7))
        contradiction_score = float(parsed.get("contradiction_score", 0.1))

        # Clamp to [0, 1]
        step_consistency    = max(0.0, min(step_consistency, 1.0))
        contradiction_score = max(0.0, min(contradiction_score, 1.0))

        return {
            "step_consistency":    round(step_consistency, 6),
            "contradiction_score": round(contradiction_score, 6)
        }

    except Exception:
        # On any failure: return neutral defaults.
        # Neutral values do not artificially inflate or deflate distortion.
        return neutral


# -----------------------------------------------------------------------------
# MAIN EXTRACTION FUNCTION
# -----------------------------------------------------------------------------

def extract_signals(prompt: str, output: str, prompt_type: str = "") -> dict:
    """
    Extract all signals from a prompt/output pair.

    Computational signals are always extracted.
    Reasoning signals (step_consistency, contradiction_score) are extracted
    only when prompt_type == "reasoning".

    Returns a dict of named signal values, all floats in [0, 1].
    """

    # --- Embeddings ---
    emb_prompt = get_embedding(prompt)
    emb_output = get_embedding(output)
    similarity = cosine_similarity(emb_prompt, emb_output)

    # --- Lexical ---
    words       = output.split() if output else []
    n_words     = max(len(words), 1)
    coherence   = len(set(words)) / n_words          # type-token ratio
    length_score = min(n_words / 100, 1.0)

    # --- Uncertainty + confidence ---
    uncertainty = uncertainty_score(output)
    confidence  = confidence_score(output)
    specificity = specificity_score(output)

    signals = {
        "similarity":    round(float(similarity), 6),
        "length_score":  round(float(length_score), 6),
        "coherence":     round(float(coherence), 6),
        "uncertainty":   uncertainty,
        "specificity":   specificity,
        "confidence":    confidence,
        # Reasoning signals default to neutral for non-reasoning prompts
        "step_consistency":    0.7,
        "contradiction_score": 0.1,
    }

    # --- Reasoning signals (API, only for reasoning prompts) ---
    if prompt_type == "reasoning":
        r_signals = reasoning_signals(prompt, output)
        signals["step_consistency"]    = r_signals["step_consistency"]
        signals["contradiction_score"] = r_signals["contradiction_score"]

    return signals
