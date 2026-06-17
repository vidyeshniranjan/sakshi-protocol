"""
retriever.py — Sakshi-Protocol V3.1

External grounding via two independent sources with cross-source consistency.

V3.1 changes (cleanup):
    - Attribute map removed (replaced by LLM extractor in entity_screener.py)
    - _extract_entity, _build_targeted_query, _clean_query removed
    - llm_query from entity screener is the primary query source
    - Simple prompt cleaning retained as fallback only

Query priority in retrieve_with_consistency:
    1. llm_query from entity screener (LLM-extracted, most accurate)
    2. Simple prompt cleaning fallback (strip question words)

Design principle: retriever is a pure retrieval layer.
Query construction belongs to the entity screener.
"""

import re
import json
import urllib.request
import urllib.parse
from dataclasses import dataclass, field

_UA = {
    "User-Agent": (
        "SakshiProtocol/3.1 "
        "(academic research; https://github.com/vidyeshniranjan/sakshi-protocol)"
    )
}

_TIMEOUT = 6


# =============================================================================
# RETRIEVAL RESULT
# =============================================================================

@dataclass
class RetrievalResult:
    context:               str   = ""
    wikipedia_context:     str   = ""
    wikidata_context:      str   = ""
    sources_found:         list  = field(default_factory=list)
    cross_source_agreement: float = 0.5
    retrieval_success:     bool  = False
    query_used:            str   = ""
    query_refined:         bool  = False
    absence_signal:        bool  = False

    def to_dict(self) -> dict:
        return {
            "sources_found":           self.sources_found,
            "cross_source_agreement":  round(self.cross_source_agreement, 4),
            "retrieval_success":       self.retrieval_success,
            "query_used":              self.query_used,
            "query_refined":           self.query_refined,
            "absence_signal":          self.absence_signal,
            "context_length":          len(self.context),
        }

    def format_for_prompt(self) -> str:
        if not self.retrieval_success:
            return (
                "[Source: None — no external grounding available]\n"
                "No verified external information was found for this query. "
                "If you cannot answer with confidence, say so explicitly."
            )
        parts = []
        if self.wikipedia_context:
            parts.append(f"[Source: Wikipedia]\n{self.wikipedia_context}")
        if self.wikidata_context:
            parts.append(f"[Source: Wikidata]\n{self.wikidata_context}")
        combined = "\n\n".join(parts)
        if self.cross_source_agreement < 0.3:
            combined += (
                "\n\n[Note: Sources showed limited agreement. "
                "Treat this context with additional caution.]"
            )
        return combined


# =============================================================================
# QUERY FALLBACK
# Simple cleaning only — used when llm_query is not available.
# =============================================================================

_QUESTION_WORDS = [
    "what were the specific", "what were the exact", "what were the key",
    "what were the", "what are the", "what is the", "what was the",
    "what were", "what are", "what is", "what was", "what did",
    "what does", "who was", "who is", "who are", "how does",
    "how did", "how is", "how many", "how long", "how old",
    "in what year", "in which", "at what", "explain", "describe",
    "tell me", "according to", "summarise", "summarize",
]


def _clean_query(prompt: str) -> str:
    """Strip question words and punctuation. Used only as llm_query fallback."""
    query = prompt.strip()
    q_lower = query.lower()
    for qw in sorted(_QUESTION_WORDS, key=len, reverse=True):
        if q_lower.startswith(qw):
            query = query[len(qw):].strip()
            break
    return query.replace("?", "").strip()[:100]


def _refine_query(prompt: str) -> str:
    """
    Extract longest capitalised sequence as entity name.
    Used on second retrieval pass when first pass returns nothing.
    """
    _FILTER = {
        "What","Who","How","Why","When","Where","Which","The",
        "According","Explain","Describe","Tell","In","At","By",
        "For","Of","On","To",
        # Name prefixes — incomplete on their own
        "Al","Ibn","Abu","Imam","Sheikh","Mulla","Rabbi",
        "Pandit","Acharya","Swami","Dr","Prof","Sir","St",
    }
    matches = re.findall(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b", prompt)
    matches = [m for m in matches if m not in _FILTER and len(m) > 3]
    if matches:
        return max(matches, key=len)[:80]
    words = [w for w in prompt.split() if len(w) > 3]
    return " ".join(words[:4])


# =============================================================================
# SOURCE RETRIEVERS
# =============================================================================

def _retrieve_wikipedia(query: str) -> str:
    search_url = (
        "https://en.wikipedia.org/w/api.php?"
        + urllib.parse.urlencode({
            "action": "query", "list": "search",
            "srsearch": query, "srlimit": 1, "format": "json"
        })
    )
    try:
        req = urllib.request.Request(search_url, headers=_UA)
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
            data    = json.loads(r.read())
            results = data.get("query", {}).get("search", [])
            if not results:
                return ""
            title = results[0]["title"]
        summary_url = (
            "https://en.wikipedia.org/api/rest_v1/page/summary/"
            + urllib.parse.quote(title)
        )
        req = urllib.request.Request(summary_url, headers=_UA)
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
            page    = json.loads(r.read())
            extract = page.get("extract", "").strip()
            return extract[:1000] if extract else ""
    except Exception:
        return ""


def _retrieve_wikidata(query: str) -> str:
    search_url = (
        "https://www.wikidata.org/w/api.php?"
        + urllib.parse.urlencode({
            "action": "wbsearchentities", "search": query,
            "language": "en", "limit": 3, "format": "json"
        })
    )
    try:
        req = urllib.request.Request(search_url, headers=_UA)
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
            data    = json.loads(r.read())
            results = data.get("search", [])
        if not results:
            return ""
        best        = results[0]
        label       = best.get("label", "")
        description = best.get("description", "")
        aliases     = best.get("aliases", [])
        parts = []
        if label:       parts.append(label)
        if description: parts.append(f"— {description}")
        if aliases:     parts.append(f"(also known as: {', '.join(aliases[:3])})")
        return " ".join(parts) if parts else ""
    except Exception:
        return ""


def _retrieve_duckduckgo(query: str) -> str:
    url = (
        "https://api.duckduckgo.com/?"
        + urllib.parse.urlencode({
            "q": query, "format": "json", "no_html": "1", "skip_disambig": "1"
        })
    )
    try:
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
            data = json.loads(r.read())
        abstract = data.get("AbstractText", "").strip()
        if abstract:
            return abstract[:500]
        answer = data.get("Answer", "").strip()
        return answer[:200] if answer else ""
    except Exception:
        return ""


# =============================================================================
# CROSS-SOURCE CONSISTENCY
# =============================================================================

def _compute_cross_source_agreement(
    wiki_context: str,
    wikidata_context: str,
    query: str,
) -> float:
    if not wiki_context and not wikidata_context:
        return 0.0
    if not wiki_context or not wikidata_context:
        return 0.5

    def sig_words(text):
        return set(
            w.lower() for w in re.findall(r"\b[a-zA-Z]{4,}\b", text)
            if w.lower() not in {"this","that","with","from","have",
                                  "been","were","they","their","which"}
        )

    wiki_words     = sig_words(wiki_context)
    wikidata_words = sig_words(wikidata_context)

    if not wiki_words or not wikidata_words:
        return 0.5

    intersection = wiki_words & wikidata_words
    union        = wiki_words | wikidata_words
    if not union:
        return 0.5

    jaccard   = len(intersection) / len(union)
    agreement = min(jaccard / 0.2, 1.0)
    return round(agreement, 4)


# =============================================================================
# MAIN RETRIEVAL FUNCTION
# =============================================================================

def retrieve_with_consistency(prompt: str, llm_query: str = "") -> RetrievalResult:
    """
    Retrieve context from Wikipedia and Wikidata with consistency check.

    Args:
        prompt:    original user prompt
        llm_query: targeted query from entity screener LLM extractor.
                   When provided, used as primary query.
                   Falls back to simple prompt cleaning when absent.

    Two-pass strategy:
        Pass 1: llm_query (or cleaned prompt) → Wikipedia + Wikidata
        Pass 2: if both empty → refined entity query → Wikipedia + DuckDuckGo
    """
    result = RetrievalResult()

    # Primary query: LLM-extracted (preferred) or simple cleaned fallback
    query = llm_query.strip() if llm_query else _clean_query(prompt)
    result.query_used = query

    wiki_context     = _retrieve_wikipedia(query)
    wikidata_context = _retrieve_wikidata(query)

    # Pass 2: refinement if both sources returned nothing
    if not wiki_context and not wikidata_context:
        refined = _refine_query(prompt)
        if refined and refined != query:
            wiki_context     = _retrieve_wikipedia(refined)
            wikidata_context = _retrieve_wikidata(refined)
            result.query_used    = refined
            result.query_refined = True

        # DuckDuckGo fallback
        if not wiki_context and not wikidata_context:
            ddg = _retrieve_duckduckgo(query)
            if ddg:
                wiki_context = ddg
                result.sources_found.append("DuckDuckGo")

    if wiki_context:     result.sources_found.append("Wikipedia")
    if wikidata_context: result.sources_found.append("Wikidata")

    result.wikipedia_context   = wiki_context
    result.wikidata_context    = wikidata_context
    result.retrieval_success   = bool(wiki_context or wikidata_context)
    result.cross_source_agreement = _compute_cross_source_agreement(
        wiki_context, wikidata_context, query
    )
    result.absence_signal = (not wiki_context and not wikidata_context)
    result.context        = result.format_for_prompt()

    return result


def retrieve(prompt: str) -> str:
    """Backward-compatible interface."""
    return retrieve_with_consistency(prompt).context
