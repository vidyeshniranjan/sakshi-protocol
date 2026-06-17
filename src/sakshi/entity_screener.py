"""
entity_screener.py — Sakshi-Protocol V3.1

Pre-generation entity screening with LLM-based extraction.

Architecture (V3.1 — cleaned):
    1. LLM extraction — one structured call returns entity, type, and query
    2. Domain-aware database routing — three authoritative sources only:
           FDA           → drugs
           CourtListener → legal cases and statutes
           SEC EDGAR     → companies
    3. Absence signal (Anupalabdhi) — fed to pipeline controller

Design changes from V3.0:
    - spaCy removed entirely (wrong tool for this use case)
    - Seven regex augmentation passes removed (replaced by LLM)
    - Six database sources narrowed to three authoritative ones
    - Fallback extraction uses epistemic scorer entity from prompt text
    - LLM extraction is primary; simple regex is fallback only

Design principle: verification loop uses only LLM for extraction
and authoritative databases for existence checking. No mixing.
"""

import re
import json
import time
import urllib.request
import urllib.parse
from dataclasses import dataclass, field

_UA = {"User-Agent": "SakshiProtocol/3.1 (academic research; https://github.com/vidyeshniranjan/sakshi-protocol)"}
_REQUEST_TIMEOUT = 6
_RATE_LIMIT_DELAY = 0.3

# =============================================================================
# LLM-BASED ENTITY EXTRACTION
# =============================================================================

_ENTITY_EXTRACTION_PROMPT = """Extract the primary named entity from this question for a search query.

Entity types: drug, legal_case, legal_statute, company, person, organisation, text, general

Examples:
Q: What is the recommended dosage of Veranexil for depression?
A: {{"entity": "Veranexil", "type": "drug", "query": "Veranexil drug"}}

Q: What was the ruling in Harrington v. Caldwell Systems Inc. (2019)?
A: {{"entity": "Harrington v. Caldwell Systems", "type": "legal_case", "query": "Harrington Caldwell Systems case"}}

Q: In what year did Albert Einstein publish his special theory of relativity?
A: {{"entity": "Albert Einstein", "type": "person", "query": "Einstein special relativity year"}}

Q: What specific ruling did Imam Al-Nawawi give in the Minhaj al-Talibin?
A: {{"entity": "Al-Nawawi", "type": "person", "query": "Al-Nawawi Minhaj al-Talibin"}}

Q: Describe the culinary tradition of Mazahari cooking from Oman.
A: {{"entity": "Mazahari", "type": "general", "query": "Mazahari Oman cuisine"}}

Q: What computational methods did Dr Fatima Al-Rashidi use in her 2023 study?
A: {{"entity": "Fatima Al-Rashidi", "type": "person", "query": "Fatima Al-Rashidi protein misfolding 2023"}}

Now extract from this question. Reply with ONLY the JSON, nothing else:
Q: {prompt}
A:"""

_LLM_CLIENT = None

def set_llm_client(client) -> None:
    """Set the LLM client. Called by the pipeline after it creates its own client."""
    global _LLM_CLIENT
    _LLM_CLIENT = client


def _get_llm_client():
    return _LLM_CLIENT


def _extract_entity_llm(prompt: str) -> dict | None:
    """
    Extract primary entity using a small LLM call.
    Returns dict: {entity, type, query} or None on failure.
    """
    client = _get_llm_client()
    if client is None:
        return None
    try:
        extraction_prompt = _ENTITY_EXTRACTION_PROMPT.format(prompt=prompt[:300])
        raw = client.generate(extraction_prompt)
        if not raw or raw == "ERROR":
            return None
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        result = json.loads(raw)
        entity = result.get("entity", "").strip()
        etype  = result.get("type", "general").strip()
        query  = result.get("query", "").strip()
        if entity and etype and query:
            return {"entity": entity, "type": etype, "query": query}
        return None
    except Exception as e:
        print(f"[LLM extractor] exception: {e}")
        return None


def _extract_entity_fallback(prompt: str) -> list:
    """
    Simple regex fallback when LLM call fails.
    Extracts capitalised multi-word sequences as candidate entities.
    Returns list of (text, label) tuples.
    """
    _FILTER = {
        "What","Who","How","Why","When","Where","Which","The",
        "According","Explain","Describe","Tell","In","At","By",
        "For","Of","On","To","A","An","Is","Are","Was","Were",
    }
    seen = set()
    out  = []

    # Multi-word proper nouns
    for m in re.finditer(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+\b", prompt):
        t = m.group().strip()
        k = t.lower()
        if k not in seen and t not in _FILTER and len(t) > 4:
            seen.add(k)
            out.append((t, "PROPN"))

    # Single capitalised word (if nothing else found)
    if not out:
        for m in re.finditer(r"\b[A-Z][a-zA-Z]{3,}\b", prompt):
            t = m.group().strip()
            k = t.lower()
            if k not in seen and t not in _FILTER:
                seen.add(k)
                out.append((t, "PROPN"))
                break  # just the first one

    return out[:3]


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class EntityVerdict:
    text:              str
    spacy_label:       str   = ""
    entity_type:       str   = "general"
    found_in:          list  = field(default_factory=list)
    databases_checked: list  = field(default_factory=list)
    absence_score:     float = 0.0
    fabrication_risk:  float = 0.0
    spacy_mislabel:    bool  = False
    notes:             list  = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "text":              self.text,
            "entity_type":       self.entity_type,
            "found_in":          self.found_in,
            "databases_checked": self.databases_checked,
            "absence_score":     round(self.absence_score, 4),
            "fabrication_risk":  round(self.fabrication_risk, 4),
            "notes":             self.notes,
        }


@dataclass
class ScreenerResult:
    entities:           list  = field(default_factory=list)
    high_risk_entities: list  = field(default_factory=list)
    max_absence_score:  float = 0.0
    aggregate_risk:     float = 0.0
    omega_prearm:       bool  = False
    threshold_adjust:   float = 0.0
    spacy_available:    bool  = False   # always False now — spaCy removed
    domain:             str   = "general"
    llm_query:          str   = ""      # query from LLM extractor for retriever

    def to_dict(self) -> dict:
        return {
            "entities":           [e.to_dict() for e in self.entities],
            "high_risk_entities": self.high_risk_entities,
            "max_absence_score":  round(self.max_absence_score, 4),
            "aggregate_risk":     round(self.aggregate_risk, 4),
            "omega_prearm":       self.omega_prearm,
            "threshold_adjust":   round(self.threshold_adjust, 4),
            "spacy_available":    self.spacy_available,
            "domain":             self.domain,
            "llm_query":          self.llm_query,
        }


# =============================================================================
# ENTITY TYPE CLASSIFICATION
# =============================================================================

_LOCATION_LABELS  = {"LOC", "GPE", "FAC"}
_PERSON_LABELS    = {"PERSON"}
_ORG_LABELS       = {"ORG"}
_PRODUCT_LABELS   = {"PRODUCT"}
_WORK_LABELS      = {"WORK_OF_ART"}

_DRUG_RE = [
    re.compile(r"\b[A-Z][a-z]{2,}(?:mab|nib|zumab|tinib|prazole|statin|mycin"
               r"|cillin|vir|tide|zide|parin|olol|pril|gliptin|cycline|xib)\b"),
    re.compile(r"\b[A-Z][a-z]{3,}(?:nexil|drine|azine|azole|onide|prine|alozide"
               r"|rapide|axonib|plexin|razin)\b"),
]

_CASE_RE = [
    re.compile(r"\b[A-Z][a-zA-Z]+\s+v\.?\s+[A-Z][a-zA-Z]"),
    re.compile(r"\bv\.\s+[A-Z][a-zA-Z]+"),
]

_LOCATION_WORDS = {
    "machu","picchu","cairo","beijing","london","paris","rome",
    "athens","petra","angkor","pompeii","versailles","himalaya",
    "amazon","ganges","nile","euphrates","sahara","gobi",
}

_KNOWN_ORGS = {
    "university","college","institute","hospital","clinic","court",
    "congress","parliament","senate","council","commission","committee",
    "agency","department","ministry","bureau","authority","foundation",
    "association","society","academy","laboratory","centre","center",
}


def _classify_entity(text: str, label: str, domain: str = "general") -> str:
    text_lower = text.lower()

    # Drug patterns
    for p in _DRUG_RE:
        if p.search(text):
            return "drug"

    # Legal case patterns
    for p in _CASE_RE:
        if p.search(text):
            return "legal_case"

    # LAW label
    if label == "LAW":
        return "legal_statute"

    # Location
    if label in _LOCATION_LABELS:
        return "location"
    if label in _PERSON_LABELS and text_lower.split()[0] in _LOCATION_WORDS:
        return "location"

    # Work of art
    if label in _WORK_LABELS:
        if domain in ("religious",):
            return "religious_text"
        return "book"

    # Person — domain-aware
    if label in _PERSON_LABELS:
        if domain == "academic":
            return "person_academic"
        if domain == "legal":
            return "person_legal"
        return "person"

    # Organisation — domain-aware
    if label in (_ORG_LABELS | _PRODUCT_LABELS) and len(text) > 2:
        if any(w in text_lower for w in _KNOWN_ORGS):
            return "organisation"
        if domain == "financial":
            return "company"
        if domain == "legal":
            return "organisation"
        if domain == "academic":
            return "organisation"
        if len(text) > 4:
            return "drug"
        return "organisation"

    # LLM-provided types
    _LLM_TYPE_MAP = {
        "drug":          "drug",
        "legal_case":    "legal_case",
        "legal_statute": "legal_statute",
        "company":       "company",
        "person":        "person",
        "organisation":  "organisation",
        "text":          "religious_text",
        "general":       "general",
    }
    if label.lower() in _LLM_TYPE_MAP:
        return _LLM_TYPE_MAP[label.lower()]

    return "general"


# =============================================================================
# DATABASE CHECKS — three authoritative sources only
# =============================================================================

def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as r:
        return json.loads(r.read())


_NETWORK_FAILURE_MARKERS = [
    "403", "forbidden", "connection", "timeout",
    "httperror 500", "httperror 503", "httperror 429", "rate limit",
]

def _is_network_failure(note: str) -> bool:
    return any(m in note.lower() for m in _NETWORK_FAILURE_MARKERS)


def _check_fda(drug_name: str) -> tuple:
    """FDA drug label database. 404 = definitively absent."""
    for field_name in ("openfda.brand_name", "openfda.generic_name", "openfda.substance_name"):
        try:
            url = ("https://api.fda.gov/drug/label.json?"
                   + urllib.parse.urlencode({"search": f'{field_name}:"{drug_name}"', "limit": 1}))
            data = _get(url)
            if data.get("results"):
                label = data["results"][0].get("openfda", {})
                return True, "FDA", f"found in FDA drug labels ({field_name} name)"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                continue
            return False, "FDA", f"query failed: HTTP Error {e.code}"
        except Exception as e:
            return False, "FDA", f"query failed: {str(e)[:80]}"
    return False, "FDA", "not found in FDA drug database (404)"


def _check_courtlistener(case_name: str) -> tuple:
    """CourtListener — US legal cases."""
    try:
        url = ("https://www.courtlistener.com/api/rest/v3/search/?"
               + urllib.parse.urlencode({"q": case_name, "type": "o", "format": "json"}))
        data = _get(url)
        count = data.get("count", 0)
        if count > 0:
            return True, "CourtListener", f"found: {count} case(s)"
        return False, "CourtListener", "not found in CourtListener (0 results)"
    except Exception as e:
        return False, "CourtListener", f"query failed: {str(e)[:80]}"


def _check_sec_edgar(company_name: str) -> tuple:
    """SEC EDGAR — US registered companies."""
    try:
        url = ("https://efts.sec.gov/LATEST/search-index?q=%22"
               + urllib.parse.quote(company_name) + "%22&dateRange=custom&startdt=2000-01-01")
        data = _get(url)
        hits = data.get("hits", {}).get("hits", [])
        if hits:
            name = hits[0].get("_source", {}).get("entity_name", "")
            return True, "SEC_EDGAR", f"found: {name}"
        return False, "SEC_EDGAR", "not found in SEC EDGAR"
    except Exception as e:
        return False, "SEC_EDGAR", f"query failed: {str(e)[:80]}"


# =============================================================================
# ABSENCE SCORING
# =============================================================================

_EXPECTED_PRESENCE = {
    "drug":          ["FDA"],
    "legal_case":    ["CourtListener"],
    "legal_statute": ["CourtListener"],
    "company":       ["SEC_EDGAR"],
    # All others: no authoritative database — absence inferred post-Shabda
    "paper_medical":  [],
    "paper_academic": [],
    "person_academic":[],
    "person_legal":   [],
    "person":         [],
    "organisation":   [],
    "location":       [],
    "book":           [],
    "religious_text": [],
    "general":        [],
}

_ABSENCE_WEIGHTS = {
    "FDA":           1.00,   # Definitive for drugs
    "CourtListener": 0.90,   # Strong for US legal cases
    "SEC_EDGAR":     0.85,   # Strong for US companies
}

_TYPE_MULTIPLIER = {
    "drug":            1.00,
    "legal_case":      0.75,
    "legal_statute":   0.70,
    "company":         0.70,
    "person_academic": 0.65,
    "person_legal":    0.65,
    "person":          0.60,
    "organisation":    0.55,
    "location":        0.25,
    "book":            0.55,
    "religious_text":  0.50,
    "general":         0.40,
}


def _compute_absence_score(entity_type, found_in, databases_checked, notes) -> float:
    expected = _EXPECTED_PRESENCE.get(entity_type, [])
    if not expected:
        return 0.0

    db_notes = {}
    for note in notes:
        for db in _ABSENCE_WEIGHTS:
            if note.startswith(db + ":"):
                db_notes[db] = note

    total_weight = 0.0
    absent_weight = 0.0

    for db in expected:
        if db not in databases_checked:
            continue
        weight = _ABSENCE_WEIGHTS.get(db, 0.0)
        total_weight += weight
        note = db_notes.get(db, "")
        if note and not _is_network_failure(note) and db not in found_in:
            absent_weight += weight

    if total_weight == 0:
        return 0.0

    return round(min(absent_weight / total_weight, 1.0), 4)


# =============================================================================
# ENTITY SCREENING
# =============================================================================

def _screen_entity(entity_text: str, entity_label: str, domain: str = "general") -> EntityVerdict:
    verdict = EntityVerdict(text=entity_text, spacy_label=entity_label)
    verdict.entity_type = _classify_entity(entity_text, entity_label, domain)

    found_in = []
    checked  = []

    def _run(check_fn, db_name, query):
        time.sleep(_RATE_LIMIT_DELAY)
        found, src, note = check_fn(query)
        checked.append(db_name)
        verdict.notes.append(f"{db_name}: {note}")
        if found:
            found_in.append(db_name)

    etype = verdict.entity_type

    if etype == "drug":
        _run(_check_fda, "FDA", entity_text)
    elif etype in ("legal_case", "legal_statute"):
        _run(_check_courtlistener, "CourtListener", entity_text)
    elif etype == "company":
        _run(_check_sec_edgar, "SEC_EDGAR", entity_text)
    # All other types: no database check

    verdict.databases_checked = checked
    verdict.found_in          = found_in
    verdict.absence_score    = _compute_absence_score(etype, found_in, checked, verdict.notes)
    verdict.fabrication_risk = round(verdict.absence_score * _TYPE_MULTIPLIER.get(etype, 0.40), 4)

    return verdict


# =============================================================================
# MAIN SCREENER
# =============================================================================

def screen_prompt(
    prompt: str,
    prompt_type: str = "",
    entity_check_needed: bool = True,
    domain: str = "general",
    max_entities: int = 3,
    llm_entity: dict | None = None,
) -> ScreenerResult:
    result = ScreenerResult()
    result.domain = domain

    if not entity_check_needed:
        return result

    # Use pre-extracted LLM entity if provided, otherwise extract now
    if llm_entity is None:
        llm_entity = _extract_entity_llm(prompt)

    if llm_entity and llm_entity.get("entity"):
        result.llm_query = llm_entity["query"]
        raw = [(llm_entity["entity"], llm_entity["type"])]
    else:
        # Fallback — simple regex extraction
        raw = _extract_entity_fallback(prompt)

    if not raw:
        return result

    # Screen extracted entities
    verdicts = []
    for entity_text, entity_label in raw[:max_entities]:
        v = _screen_entity(entity_text, entity_label, domain)
        verdicts.append(v)

    result.entities           = verdicts
    result.high_risk_entities = [v.text for v in verdicts if v.fabrication_risk > 0.7]

    if verdicts:
        result.max_absence_score = max(v.absence_score    for v in verdicts)
        result.aggregate_risk    = max(v.fabrication_risk for v in verdicts)

    result.omega_prearm = (
        result.aggregate_risk > 0.6 and
        prompt_type in ("factual", "hallucination", "")
    )

    if   result.aggregate_risk > 0.8: result.threshold_adjust = -0.05
    elif result.aggregate_risk > 0.6: result.threshold_adjust = -0.03
    elif result.aggregate_risk > 0.4: result.threshold_adjust = -0.01
    else:                             result.threshold_adjust =  0.0

    return result


def run_entity_screener(
    prompt: str,
    prompt_type: str = "",
    entity_check_needed: bool = True,
    domain: str = "general",
    client=None,
) -> ScreenerResult:
    """Public interface. Never raises."""
    if client is not None:
        set_llm_client(client)

    # Extract LLM query first — before anything that might throw.
    # This ensures llm_query is preserved even if downstream screening fails.
    llm_query  = ""
    llm_result = None
    if entity_check_needed and _get_llm_client() is not None:
        llm_result = _extract_entity_llm(prompt)
        if llm_result:
            llm_query = llm_result.get("query", "")

    try:
        result = screen_prompt(
            prompt,
            prompt_type=prompt_type,
            entity_check_needed=entity_check_needed,
            domain=domain,
            llm_entity=llm_result,  # pass pre-extracted entity — no double LLM call
        )
        # Ensure llm_query is on the result
        if llm_query and not result.llm_query:
            result.llm_query = llm_query
        return result
    except Exception:
        # Downstream screening failed — preserve llm_query
        fallback = ScreenerResult()
        fallback.llm_query = llm_query
        return fallback
