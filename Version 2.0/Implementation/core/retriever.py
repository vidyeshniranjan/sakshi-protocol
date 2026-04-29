import urllib.request
import urllib.parse
import json


# =============================================================================
# Omega (Ω) — External Grounding Mechanism
#
# Design principle: Ω must be genuinely external to the generator model.
# Using the same LLM to "verify" its own outputs is the self-consistency
# problem the Sakshi-Protocol is designed to solve (see Section 3.6).
#
# Primary source: Wikipedia REST API (free, no API key, genuinely external)
# Fallback:       DuckDuckGo Instant Answer API (also free, no API key)
#
# If both external sources fail (no internet, topic not found), the system
# returns an explicit uncertainty signal rather than silently falling back
# to the generator model. This preserves the grounding guarantee.
# =============================================================================


def retrieve_wikipedia(query: str, timeout: int = 5):
    """
    Query the Wikipedia REST API for a page summary.
    Returns the extract string if found, None otherwise.
    """
    search_url = (
        "https://en.wikipedia.org/w/api.php?"
        + urllib.parse.urlencode({
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": 1,
            "format": "json"
        })
    )

    try:
        with urllib.request.urlopen(search_url, timeout=timeout) as r:
            data = json.loads(r.read())
            results = data.get("query", {}).get("search", [])
            if not results:
                return None
            title = results[0]["title"]

        summary_url = (
            "https://en.wikipedia.org/api/rest_v1/page/summary/"
            + urllib.parse.quote(title)
        )
        with urllib.request.urlopen(summary_url, timeout=timeout) as r:
            page = json.loads(r.read())
            extract = page.get("extract", "").strip()
            return extract if extract else None

    except Exception:
        return None


def retrieve_duckduckgo(query: str, timeout: int = 5):
    """
    Query the DuckDuckGo Instant Answer API.
    Returns the abstract text if found, None otherwise.
    """
    url = (
        "https://api.duckduckgo.com/?"
        + urllib.parse.urlencode({
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1
        })
    )

    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            data = json.loads(r.read())
            abstract = data.get("AbstractText", "").strip()
            return abstract if abstract else None

    except Exception:
        return None


def retrieve(prompt: str) -> str:
    """
    Main Omega grounding function.

    Attempts to retrieve genuinely external factual context for the prompt.
    Priority:
      1. Wikipedia REST API
      2. DuckDuckGo Instant Answer API
      3. Explicit uncertainty signal (never falls back to the generator model)

    The returned string is used by the pipeline to construct a grounded
    prompt for regeneration. An explicit uncertainty signal causes the
    pipeline fallback abstain condition to trigger.
    """

    # Clean the prompt into a search query
    query = (
        prompt
        .replace("What were", "").replace("What are", "")
        .replace("What is", "").replace("What did", "")
        .replace("Who was", "").replace("Who is", "")
        .replace("How does", "").replace("How did", "")
        .replace("Explain", "").replace("?", "")
        .strip()
    )

    # Attempt 1: Wikipedia
    result = retrieve_wikipedia(query)
    if result:
        return f"[Source: Wikipedia]\n{result}"

    # Attempt 2: DuckDuckGo
    result = retrieve_duckduckgo(query)
    if result:
        return f"[Source: DuckDuckGo]\n{result}"

    # Attempt 3: No external source found.
    # Return an explicit uncertainty signal. The pipeline's fallback abstain
    # condition will trigger when distortion remains high after regeneration
    # on this context — the correct behaviour when no grounding is available.
    return (
        "[Source: None — no external grounding available]\n"
        "No verified external information was found for this query. "
        "The answer is uncertain and cannot be externally grounded. "
        "If you cannot answer with confidence, say so explicitly."
    )
