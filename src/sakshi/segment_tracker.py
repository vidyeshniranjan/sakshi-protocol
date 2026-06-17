"""
segment_tracker.py — Sakshi-Protocol V3
Dynamic state tracking across output segments.

Phase 1b of the V3 pre-generation layer extension.

V2 and early V3 computed a single Ct = [S, R, T, V, I] over the full output.
This single-pass measurement misses intra-output instability — a model that
starts grounded and fabricates mid-response produces the same endpoint
distortion as one that is uniformly uncertain throughout.

Phase 1b extends state computation to operate per paragraph segment,
producing a distortion trajectory rather than a single point.

New metrics derived from the trajectory:
    mean_distortion     — average D_combined across segments
    peak_distortion     — maximum D_combined across segments
    peak_segment        — index of the segment with highest distortion
    drift_score         — direction and magnitude of distortion change
                          positive = distortion increasing (drifting toward risk)
                          negative = distortion decreasing (converging toward stability)
    inflection_count    — number of direction changes in trajectory
    trajectory_shape    — characterisation: stable | drifting | converging | volatile

Paper correspondence: Vivartavada — the transformation path matters,
not just the endpoint. The shape of the trajectory encodes information
about how the generation unfolded.

Design principles:
    - Segmentation is paragraph-level (split on double newline)
    - Minimum segment length: 15 words (shorter segments merged with adjacent)
    - Each segment gets an independent Ct computation
    - Embedding calls: one per segment (not shared with full-output computation)
    - Reasoning signals: not re-evaluated per segment (too expensive)
      Segment-level step_consistency and contradiction use neutral defaults
    - The full-output Ct from pipeline.py remains the primary controller input
      Trajectory metrics are stored as additional diagnostic signals
"""

import re
import numpy as np
from typing import Optional

from sakshi.signals import (
    get_embedding,
    cosine_similarity,
    uncertainty_score,
    confidence_score,
    specificity_score,
)
from sakshi.state import compute_state
from sakshi.distortion import compute_distortion


# =============================================================================
# CONFIGURATION
# =============================================================================

# Minimum word count for a segment to be scored independently.
# Segments shorter than this are merged with the adjacent segment.
MIN_SEGMENT_WORDS = 15

# Maximum number of segments to score.
# Caps embedding cost for very long outputs.
# Outputs with more segments have the excess merged into the last scored segment.
MAX_SEGMENTS = 8


# =============================================================================
# TRAJECTORY RESULT
# =============================================================================

class TrajectoryResult:
    """
    Result of segment-level distortion tracking.

    Attributes:
        segments            : list of per-segment distortion dicts
                              each contains D_factual, D_reasoning, D_combined,
                              segment_index, word_count
        mean_distortion     : mean D_combined across segments
        peak_distortion     : maximum D_combined across segments
        peak_segment        : index of segment with highest distortion (0-based)
        drift_score         : float in [-1, 1]
                              positive = distortion increasing across output
                              negative = distortion decreasing across output
                              ~0 = stable or oscillating
        inflection_count    : number of direction changes in trajectory
        trajectory_shape    : "stable" | "drifting" | "converging" | "volatile"
        segment_count       : number of segments scored
        skipped             : bool — True if output was too short to segment
    """

    def __init__(self):
        self.segments         = []
        self.mean_distortion  = 0.0
        self.peak_distortion  = 0.0
        self.peak_segment     = 0
        self.drift_score      = 0.0
        self.inflection_count = 0
        self.trajectory_shape = "stable"
        self.segment_count    = 0
        self.skipped          = False

    def to_dict(self) -> dict:
        return {
            "segments":         self.segments,
            "mean_distortion":  round(self.mean_distortion, 4),
            "peak_distortion":  round(self.peak_distortion, 4),
            "peak_segment":     self.peak_segment,
            "drift_score":      round(self.drift_score, 4),
            "inflection_count": self.inflection_count,
            "trajectory_shape": self.trajectory_shape,
            "segment_count":    self.segment_count,
            "skipped":          self.skipped,
        }


# =============================================================================
# SEGMENTATION
# =============================================================================

def _split_segments(output: str) -> list[str]:
    """
    Split output into paragraph segments.

    Strategy:
        1. Split on double newline (paragraph boundary)
        2. Merge segments shorter than MIN_SEGMENT_WORDS with adjacent
        3. Cap at MAX_SEGMENTS (merge excess into last segment)
        4. Return list of cleaned segment strings

    Returns empty list if output is too short to segment meaningfully.
    """
    if not output or not output.strip():
        return []

    # Split on double newline — primary paragraph boundary
    raw_segments = re.split(r'\n\s*\n', output.strip())

    # Also split on numbered list markers as secondary boundary
    # (outputs like "1. ... 2. ... 3. ..." should be segmented)
    expanded = []
    for seg in raw_segments:
        # Split on numbered list items at line start
        sub = re.split(r'\n(?=\s*\d+[\.\)]\s)', seg)
        expanded.extend(sub)

    # Clean each segment
    cleaned = [s.strip() for s in expanded if s.strip()]

    if not cleaned:
        return []

    # Merge short segments with the next segment
    merged = []
    buffer = ""
    for seg in cleaned:
        if buffer:
            combined = buffer + " " + seg
            if len(combined.split()) >= MIN_SEGMENT_WORDS:
                merged.append(combined.strip())
                buffer = ""
            else:
                buffer = combined
        else:
            if len(seg.split()) < MIN_SEGMENT_WORDS:
                buffer = seg
            else:
                merged.append(seg)

    # Flush remaining buffer into last segment or as its own
    if buffer:
        if merged:
            merged[-1] = merged[-1] + " " + buffer
        else:
            merged.append(buffer)

    # Cap at MAX_SEGMENTS — merge excess into last segment
    if len(merged) > MAX_SEGMENTS:
        last = " ".join(merged[MAX_SEGMENTS - 1:])
        merged = merged[:MAX_SEGMENTS - 1] + [last]

    return merged


# =============================================================================
# SEGMENT SCORING
# =============================================================================

def _score_segment(
    prompt: str,
    segment: str,
    segment_index: int,
    prompt_type: str = "",
) -> dict:
    """
    Score a single output segment independently.

    Computes full Ct and distortion for the segment.
    Reasoning signals use neutral defaults — re-evaluating per segment
    would multiply API calls without adding meaningful signal since
    reasoning consistency is a property of the full output not a paragraph.

    Returns a dict with distortion values and segment metadata.
    """
    try:
        # Embeddings for this segment
        emb_prompt  = get_embedding(prompt)
        emb_segment = get_embedding(segment)
        similarity  = cosine_similarity(emb_prompt, emb_segment)

        # Lexical signals for this segment
        words        = segment.split() if segment else []
        n_words      = max(len(words), 1)
        coherence    = len(set(words)) / n_words
        length_score = min(n_words / 100, 1.0)

        # Uncertainty and confidence for this segment
        uncertainty = uncertainty_score(segment)
        confidence  = confidence_score(segment)
        specificity = specificity_score(segment)

        # Signals dict for this segment
        # Reasoning signals use neutral defaults — not re-evaluated per segment
        seg_signals = {
            "similarity":          round(float(similarity), 6),
            "length_score":        round(float(length_score), 6),
            "coherence":           round(float(coherence), 6),
            "uncertainty":         uncertainty,
            "specificity":         specificity,
            "confidence":          confidence,
            "step_consistency":    0.7,   # neutral — not re-evaluated per segment
            "contradiction_score": 0.1,   # neutral
        }

        # State and distortion for this segment
        seg_state      = compute_state(seg_signals)
        seg_distortion = compute_distortion(seg_state, prompt_type=prompt_type)

        return {
            "segment_index": segment_index,
            "word_count":    n_words,
            "D_factual":     seg_distortion["D_factual"],
            "D_reasoning":   seg_distortion["D_reasoning"],
            "D_combined":    seg_distortion["D_combined"],
            "S":             seg_state["S"],
            "R":             seg_state["R"],
            "T":             seg_state["T"],
            "V":             seg_state["V"],
            "I":             seg_state["I"],
            "uncertainty":   uncertainty,
            "confidence":    confidence,
        }

    except Exception as e:
        # On any failure, return a neutral segment score
        return {
            "segment_index": segment_index,
            "word_count":    len(segment.split()) if segment else 0,
            "D_factual":     0.25,
            "D_reasoning":   0.25,
            "D_combined":    0.25,
            "S":             0.8,
            "R":             0.3,
            "T":             0.1,
            "V":             0.05,
            "I":             0.6,
            "uncertainty":   0.2,
            "confidence":    0.2,
            "error":         str(e)[:80],
        }


# =============================================================================
# TRAJECTORY ANALYSIS
# =============================================================================

def _analyse_trajectory(d_values: list[float]) -> tuple[float, int, str]:
    """
    Analyse the distortion trajectory across segments.

    Args:
        d_values: list of D_combined values per segment, in order

    Returns:
        (drift_score, inflection_count, trajectory_shape)

    Drift score:
        Computed as the slope of the linear fit to D_combined values.
        Normalised to [-1, 1].
        Positive = distortion increasing (risk accumulating).
        Negative = distortion decreasing (stabilising).

    Inflection count:
        Number of times the direction of change reverses.
        0 = monotonic (either drifting or converging)
        1-2 = moderate oscillation
        3+ = volatile

    Trajectory shape:
        "stable"     — low variance, no strong drift
        "drifting"   — consistent increase in distortion
        "converging" — consistent decrease in distortion
        "volatile"   — high inflection count, irregular
    """
    n = len(d_values)

    if n < 2:
        return 0.0, 0, "stable"

    # Drift score — linear regression slope
    x = list(range(n))
    x_mean = sum(x) / n
    d_mean = sum(d_values) / n

    numerator   = sum((x[i] - x_mean) * (d_values[i] - d_mean) for i in range(n))
    denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

    if denominator == 0:
        slope = 0.0
    else:
        slope = numerator / denominator

    # Normalise slope to [-1, 1] — slope of 0.1 per segment = ~1.0
    drift_score = max(-1.0, min(slope / 0.1, 1.0))

    # Inflection count — direction changes
    directions = []
    for i in range(1, n):
        delta = d_values[i] - d_values[i - 1]
        if abs(delta) > 0.01:   # ignore noise below 0.01
            directions.append(1 if delta > 0 else -1)

    inflection_count = 0
    for i in range(1, len(directions)):
        if directions[i] != directions[i - 1]:
            inflection_count += 1

    # Trajectory shape classification
    variance = sum((d - d_mean) ** 2 for d in d_values) / n

    if inflection_count >= 3:
        shape = "volatile"
    elif drift_score > 0.25:
        shape = "drifting"
    elif drift_score < -0.25:
        shape = "converging"
    else:
        shape = "stable"

    return round(drift_score, 4), inflection_count, shape


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def compute_trajectory(
    prompt: str,
    output: str,
    prompt_type: str = "",
) -> TrajectoryResult:
    """
    Compute the distortion trajectory across output segments.

    Args:
        prompt      : the original user prompt
        output      : the full model output
        prompt_type : pipeline taxonomy type

    Returns:
        TrajectoryResult with per-segment scores and trajectory metrics.

    The trajectory is computed independently of the full-output state vector.
    The full-output Ct remains the primary controller input in pipeline.py.
    Trajectory metrics are stored as additional diagnostic signals for:
        - Research dashboard display
        - Results JSON (for analysis and paper figures)
        - Future V4 use as an additional controller input

    For V3 the trajectory feeds into the results JSON and dashboard only.
    It does not directly modify the controller decision — the full-output
    distortion is the controller input. This is intentional: trajectory
    calibration requires evaluation data we don't yet have.
    """
    result = TrajectoryResult()

    # Segment the output
    segments = _split_segments(output)

    # Too short to segment meaningfully — skip
    if len(segments) < 2:
        result.skipped = True
        result.trajectory_shape = "stable"
        return result

    result.segment_count = len(segments)

    # Score each segment independently
    scored_segments = []
    for i, seg in enumerate(segments):
        seg_result = _score_segment(prompt, seg, i, prompt_type=prompt_type)
        scored_segments.append(seg_result)

    result.segments = scored_segments

    # Extract D_combined values for trajectory analysis
    d_values = [s["D_combined"] for s in scored_segments]

    # Aggregate metrics
    result.mean_distortion = round(sum(d_values) / len(d_values), 4)
    result.peak_distortion = round(max(d_values), 4)
    result.peak_segment    = d_values.index(max(d_values))

    # Trajectory analysis
    drift, inflections, shape = _analyse_trajectory(d_values)
    result.drift_score      = drift
    result.inflection_count = inflections
    result.trajectory_shape = shape

    return result


def run_trajectory(
    prompt: str,
    output: str,
    prompt_type: str = "",
) -> TrajectoryResult:
    """
    Public interface. Never raises — returns skipped result on any error.
    """
    try:
        return compute_trajectory(prompt, output, prompt_type=prompt_type)
    except Exception:
        r = TrajectoryResult()
        r.skipped = True
        return r
