"""
epistemic_scorer.py — Sakshi-Protocol V3
Pre-generation epistemic load scoring with domain detection.

Deterministic, zero API cost, zero network calls.

V3 expansion: domain detection added across medical, legal, financial,
academic, religious/philosophical, astrological, cultural, geographic,
and literary domains. The detected domain is passed to the entity
screener so it routes entity existence checks to the most relevant
authoritative databases.
"""

import re
from dataclasses import dataclass, field


# =============================================================================
# RISK PATTERNS
# =============================================================================

_FABRICATION_PRONE_PATTERNS = [
    (r"what\s+are\s+the\s+(fda|ema|who|mhra|tga)[\-\s]approved", 0.90),
    (r"what\s+(were|are)\s+the\s+(key\s+)?findings\s+of", 0.85),
    (r"what\s+did\s+the\s+\d{4}\s+\w+\s+study", 0.85),
    (r"according\s+to\s+(?:dr|prof|professor|justice|judge|rabbi|imam|swami)\s+\w+", 0.80),
    (r"what\s+does\s+(?:dr|prof|professor|justice|judge)\s+\w+\s+say", 0.80),
    (r"cite\s+the\s+(specific\s+)?findings", 0.75),
    (r"what\s+are\s+the\s+(clinical\s+)?trial\s+results", 0.80),
    (r"what\s+are\s+the\s+(approved\s+)?uses\s+of", 0.75),
    (r"what\s+(is|are)\s+the\s+(exact\s+|recommended\s+|exact\s+recommended\s+)?dosage", 0.75),
    (r"exact\s+recommended\s+dosage", 0.80),
    (r"contraindications\s+listed\s+(in|on)\s+the", 0.80),
    (r"(fda|ema|mhra)\s+label", 0.75),
    (r"primary\s+contraindications", 0.70),
    (r"name\s+the\s+(specific|exact)\s+", 0.70),
    (r"provide\s+(the\s+)?(exact|specific)\s+", 0.70),
    (r"quote\s+(directly\s+)?from", 0.80),
    (r"what\s+are\s+the\s+side\s+effects\s+of", 0.65),
    (r"what\s+is\s+the\s+(mechanism\s+of\s+action|moa)\s+of", 0.70),
    (r"describe\s+the\s+(methodology|protocol)\s+of", 0.70),
    (r"what\s+(chapter|page|section|verse|shloka|sutra)\s+(does|discusses|says)", 0.75),
    (r"what\s+(is|are)\s+the\s+(holding|ruling|judgment)\s+(in|of)", 0.80),
    (r"what\s+did\s+the\s+court\s+(hold|rule|decide|find)\s+in", 0.80),
    (r"what\s+(is|are)\s+the\s+(provisions?|clauses?|sections?)\s+of", 0.75),
    (r"what\s+does\s+(section|article|clause)\s+\d+", 0.75),
    (r"what\s+(is|are)\s+the\s+(penalties?|punishments?|sentences?)\s+for", 0.70),
    (r"what\s+(is|are)\s+the\s+(annual|quarterly)\s+(revenue|profit|loss|earnings)", 0.70),
    (r"what\s+did\s+\w+\s+report\s+(in|for)\s+(q[1-4]|\d{4})", 0.75),
    (r"what\s+are\s+the\s+(exact\s+)?(lyrics|words|verses)\s+(of|to|from)", 0.80),
    (r"what\s+does\s+(the\s+)?(quran|bible|torah|vedas?|upanishads?|gita)\s+say", 0.70),
    (r"according\s+to\s+(the\s+)?(quran|bible|torah|vedas?|upanishads?|gita)", 0.70),
    (r"what\s+(is|are)\s+the\s+(astrological|zodiac|natal|birth)\s+(chart|position)", 0.65),
    (r"what\s+does\s+\w+\s+(nakshatra|rashi|dasha|graha)\s+(indicate|mean|signify)", 0.65),
    # Cultural/traditional practice fabrications — general domain
    # "Describe the culinary tradition of X", "Describe the traditional healing practice of X"
    # "Describe the traditional textile/boat-building/weaving of X"
    # These request specific descriptions of named traditions that may be fabricated
    (r"describe\s+the\s+(culinary|cooking|food)\s+tradition\s+of", 0.75),
    (r"describe\s+the\s+traditional\s+(healing|medicinal|spiritual|ceremonial)\s+(practice|ritual|ceremony)\s+of", 0.80),
    (r"describe\s+the\s+traditional\s+(textile|weaving|boat|craft|art)(\s+\w+)?\s+(form|technique|tradition|practice)", 0.75),
    (r"describe\s+the\s+traditional\s+\w+[\-\s]building\s+(techniques?|methods?|practices?)", 0.75),
    (r"describe\s+the\s+(breeding|conservation)\s+programme?\s+for", 0.70),
    (r"describe\s+the\s+(traditional|ancient|indigenous|folk)\s+(tradition|practice|ritual|ceremony|custom)", 0.75),
    (r"describe\s+the\s+(specific\s+)?(role|significance|function)\s+of\s+\w+\s+in\s+the", 0.70),
    (r"what\s+(is|are)\s+the\s+(traditional|indigenous|folk|ancient)\s+(method|technique|practice)", 0.70),
    (r"describe\s+the\s+\w+\s+(methodology|framework|index|coefficient|protocol)", 0.75),
    (r"what\s+(were|are)\s+the\s+(key\s+)?policy\s+recommendations?\s+of\s+the", 0.75),
    (r"describe\s+the\s+(gameplay|game\s*play)\s+(mechanics|features|elements)", 0.70),
    (r"what\s+(were|was)\s+the\s+outcome\s+of\s+the\s+\d{4}\s+\w+", 0.75),
    (r"what\s+(were|are)\s+the\s+results\s+of\s+the\s+\d{4}", 0.75),
    (r"describe\s+the\s+(conflict|dispute)\s+resolution\s+(mechanism|process|framework)", 0.75),
]

_FALSE_PREMISE_PATTERNS = [
    (r"given\s+that\s+.{5,50}\s+(?:is|are|was|were)\s+(?:proven|established|confirmed|shown)", 0.85),
    (r"since\s+.{5,50}\s+(?:has\s+been|is\s+now)\s+(?:proven|established|approved)", 0.85),
    (r"as\s+(?:we\s+know|is\s+well\s+known|is\s+established)", 0.70),
    (r"the\s+(?:well[\-\s]known|established|proven)\s+(?:fact|finding|result|ruling|text)\s+that", 0.80),
    (r"following\s+the\s+(?:discovery|finding|publication|ruling|judgment)\s+of", 0.75),
    (r"in\s+light\s+of\s+(?:the\s+)?(?:recent|new|latest)\s+(?:study|research|findings|ruling|case)", 0.70),
    (r"(?:the\s+)?(?:\d{4}\s+)?\w+\s+study\s+(?:showed|found|demonstrated|proved)", 0.80),
    (r"researchers\s+at\s+\w+\s+(?:university|institute|lab)\s+(?:found|showed|proved)", 0.75),
    (r"(?:the\s+)?\w+\s+(?:case|ruling|judgment|decision)\s+(?:established|held|found)", 0.80),
    (r"under\s+(?:section|article|clause)\s+\d+\s+of\s+the\s+\w+\s+act", 0.75),
]

_TEMPORAL_PATTERNS = [
    (r"\b(current(ly)?|latest|recent(ly)?|now|today|this\s+year)\b", 0.50),
    (r"\b(as\s+of\s+(today|now|\d{4}))\b", 0.55),
    (r"\b(up[\-\s]to[\-\s]date|up[\-\s]to[\-\s]the[\-\s]minute)\b", 0.55),
    (r"\b(most\s+recent|newest|state[\-\s]of[\-\s]the[\-\s]art)\b", 0.45),
    (r"\b(at\s+present|presently|in\s+\d{4})\b", 0.50),
    (r"\b(just\s+(released|published|announced|approved|decided|ruled))\b", 0.60),
    (r"\b(new(ly)?\s+(approved|released|published|launched|enacted|passed))\b", 0.60),
    (r"\b(current\s+(law|legislation|regulation|ruling|case\s+law))\b", 0.60),
    (r"\b(current\s+(market|stock|share)\s+price)\b", 0.65),
]

_DOMAIN_PATTERNS = [
    # Medical
    (r"\b(pharmacokinetics|pharmacodynamics|bioavailability|half[\-\s]life)\b", 0.65),
    (r"\b(contraindication|adverse\s+event|off[\-\s]label|indication)\b", 0.65),
    (r"\b(ld50|ic50|ec50|ki\s+value|binding\s+affinity)\b", 0.75),
    (r"\b(clinical\s+trial|phase\s+[iii]+|randomised|double[\-\s]blind)\b", 0.60),
    (r"\b(icd[\-\s]\d+|dsm[\-\s]\d+|snomed|loinc)\b", 0.70),
    # Legal
    (r"\b(jurisdiction|statute|case\s+law|precedent|tort|fiduciary)\b", 0.55),
    (r"\b(plaintiff|defendant|appellant|respondent|habeas\s+corpus|mens\s+rea)\b", 0.60),
    (r"\b(constitutional|unconstitutional|adjudication|jurisprudence)\b", 0.55),
    # Financial
    (r"\b(sec\s+filing|10[\-\s]k|prospectus|derivative|hedge\s+fund)\b", 0.55),
    (r"\b(earnings\s+per\s+share|ebitda|market\s+cap|price[\-\s]to[\-\s]earnings)\b", 0.55),
    # Scientific
    (r"\b(gene\s+expression|mrna|crispr|protein\s+folding|genome)\b", 0.60),
    (r"\b(eigenvalue|topology|manifold|hilbert\s+space|tensor)\b", 0.55),
    (r"\b(doi|isbn|issn|preprint|peer[\-\s]reviewed|impact\s+factor)\b", 0.60),
    # Religious / esoteric
    (r"\b(shloka|sutra|upanishad|vedanta|tantra|mantra|yantra)\b", 0.55),
    (r"\b(nakshatra|rashi|dasha|graha|lagna|bhava|jyotish)\b", 0.60),
    (r"\b(hadith|fatwa|sharia|fiqh|talmud|mishnah|kabbalah)\b", 0.60),
    (r"\b(hermeneutics|exegesis|eschatology|soteriology)\b", 0.55),
    # Historical
    (r"\b(papyrus|cuneiform|manuscript|codex|inscription|epigraphy)\b", 0.55),
    (r"\b(dynasty|regnal|annals|chronicles|hagiography)\b", 0.50),
]

_CONJUNCTIVE_PATTERNS = [
    (r"\band\s+(also|additionally|furthermore|moreover)\b", 0.35),
    (r"\b(as\s+well\s+as|in\s+addition\s+to|along\s+with)\b", 0.35),
    (r"\b(list\s+(all|the|every)|enumerate\s+(all|every))\b", 0.45),
    (r"\b(provide\s+(a\s+)?(complete|comprehensive|full|detailed)\s+list)\b", 0.50),
    (r"\b(include\s+(all|every)\s+(relevant|applicable|possible))\b", 0.45),
    (r"\b(step[\-\s]by[\-\s]step|in\s+order|sequentially|chronologically)\b", 0.30),
]

_COUNTERFACTUAL_PATTERNS = [
    (r"\b(what\s+would\s+have\s+happened|what\s+if)\b", 0.40),
    (r"\b(hypothetically|theoretically|in\s+theory|suppose\s+that)\b", 0.35),
    (r"\b(imagine\s+(that|if)|let'?s\s+say|assume\s+that)\b", 0.35),
    (r"\b(could\s+have|would\s+have|should\s+have|might\s+have)\b", 0.30),
    (r"\b(alternative(ly)?|alternatively|instead\s+of|rather\s+than)\b", 0.25),
]

_NAMED_ENTITY_INDICATORS = [
    r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b",
    r"\b(?:Dr|Prof|Mr|Mrs|Ms|Sir|Justice|Judge|Rabbi|Imam|Swami|Pandit)\.\s*[A-Z]\w+",
    r"\b[A-Z]{2,}\b",
    r"\b\d{4}\b",
    r"\b[A-Z][a-z]+(?:[\-][A-Z][a-z]+)+\b",
]
_NAMED_ENTITY_RE = [re.compile(p) for p in _NAMED_ENTITY_INDICATORS]

_SPECIFICITY_PATTERNS = [
    r"\b\d+\.?\d*\s*(?:mg|ml|kg|lb|mm|cm|km|µg|ng|%|percent)\b",
    r"\b\d+\s*(?:times|fold|x)\b",
    r"\$\d+(?:\.\d+)?(?:\s*(?:million|billion|trillion))?\b",
    r"\b\d+(?:\.\d+)?\s*(?:years?|months?|weeks?|days?|hours?)\b",
    r"\bp\s*[<>=]\s*0\.\d+\b",
    r"\b(?:sensitivity|specificity|accuracy)\s+of\s+\d+",
    r"\b\d+(?:\.\d+)?\s*(?:crore|lakh|rupees?|pounds?|euros?|yen)\b",
    r"\barticle\s+\d+|section\s+\d+|clause\s+\d+\b",
]
_SPECIFICITY_RE = [re.compile(p, re.IGNORECASE) for p in _SPECIFICITY_PATTERNS]


# =============================================================================
# DOMAIN DETECTION
# =============================================================================

_DOMAIN_DETECTION = {
    "medical": [
        r"\b(drug|medication|medicine|pharmaceutical|dosage|treatment|therapy|disease|disorder|syndrome|symptom|diagnosis|prognosis|clinical|patient|hospital|physician|surgeon|prescription|vaccine|antibiotic|chemotherapy|oncology|cardiology|neurology|psychiatry|fda[\-\s]approved|ema[\-\s]approved)\b",
        r"\b(mg|mcg|µg|iv\b|subcutaneous|intramuscular|oral\s+administration|contraindication|adverse\s+event|clinical\s+trial|phase\s+[iii]+)\b",
    ],
    "legal": [
        r"\b(law|legal|court|judge|justice|ruling|judgment|statute|legislation|act|bill|constitution|plaintiff|defendant|attorney|lawyer|counsel|tort|contract|criminal|civil|appeal|jurisdiction|precedent|case\s+law|habeas\s+corpus|supreme\s+court|high\s+court|district\s+court)\b",
        r"\b(v\.\s+[A-Z]|[A-Z][a-z]+\s+v\.\s+[A-Z][a-z]+)\b",
    ],
    "financial": [
        r"\b(stock|share|equity|bond|derivative|option|futures|etf|fund|portfolio|dividend|earnings|revenue|profit|loss|ebitda|ipo|sec|nasdaq|nyse|market\s+cap|valuation|investment|hedge\s+fund|private\s+equity|venture\s+capital|cryptocurrency|bitcoin)\b",
        r"\b(10[\-\s]k|10[\-\s]q|8[\-\s]k|sec\s+filing|annual\s+report|balance\s+sheet|income\s+statement)\b",
    ],
    "academic": [
        r"\b(paper|journal|publication|doi|isbn|issn|preprint|arxiv|peer[\-\s]reviewed|citation|abstract|methodology|literature\s+review|meta[\-\s]analysis|systematic\s+review|conference|proceedings|thesis|dissertation)\b",
        r"\b(university|institute|professor|researcher|academic|scholar|faculty|study|findings|published|authors?)\b",
    ],
    "religious": [
        r"\b(quran|koran|bible|torah|talmud|vedas?|upanishads?|bhagavad\s+gita|gita|mahabharata|ramayana|puranas?|sutras?|tripitaka|dhammapada|guru\s+granth|hadith|sunna|sharia|fiqh|fatwa|pope|bishop|imam|rabbi|priest|monk|swami|pandit|mullah)\b",
        r"\b(shloka|mantra|yantra|tantra|dharma|karma|moksha|nirvana|samsara|brahman|atman|vedanta|advaita|theravada|mahayana|sufi|kabbalah|scripture|gospel|epistle|psalm)\b",
    ],
    "astrological": [
        r"\b(astrology|astrological|horoscope|zodiac|natal\s+chart|birth\s+chart|ascendant|retrograde|conjunction|opposition|trine|sextile)\b",
        r"\b(nakshatra|rashi|lagna|dasha|graha|bhava|yoga|jyotish|vedic\s+astrology|sun\s+sign|moon\s+sign|rising\s+sign|aries|taurus|gemini|cancer|leo|virgo|libra|scorpio|sagittarius|capricorn|aquarius|pisces)\b",
    ],
    "historical": [
        r"\b(ancient|medieval|renaissance|enlightenment|industrial\s+revolution|world\s+war|cold\s+war|empire|dynasty|kingdom|republic|revolution|colonialism|archaeological|excavation|artifact|inscription)\b",
        r"\b(bc|ad|bce|ce|century|millennium|prehistoric|antiquity|classical\s+period|byzantine|ottoman|mughal|roman|greek|egyptian|mesopotamian)\b",
    ],
    "geographical": [
        r"\b(country|nation|city|town|village|region|province|state|continent|ocean|sea|river|mountain|valley|desert|island|peninsula|coordinates|latitude|longitude|capital|geography)\b",
    ],
    "literary": [
        r"\b(novel|poem|play|drama|essay|short\s+story|novella|autobiography|biography|memoir|anthology|genre|narrative|protagonist|antagonist|author|writer|poet|playwright|fiction|non[\-\s]fiction|classic|bestseller)\b",
        r"\b(isbn|publisher|edition|chapter|verse|stanza|canto|prologue|epilogue|preface)\b",
    ],
}

_DOMAIN_RE = {
    domain: [re.compile(p, re.IGNORECASE) for p in patterns]
    for domain, patterns in _DOMAIN_DETECTION.items()
}


def detect_domain(prompt: str) -> str:
    """
    Detect the primary domain of the prompt for database routing.
    Returns the domain with the most pattern hits, or 'general'.
    """
    scores = {}
    for domain, patterns in _DOMAIN_RE.items():
        hits = sum(1 for p in patterns if p.search(prompt))
        if hits > 0:
            scores[domain] = hits
    if not scores:
        return "general"
    return max(scores, key=scores.get)


# =============================================================================
# EPISTEMIC PROFILE
# =============================================================================

@dataclass
class EpistemicProfile:
    risk_score:          float = 0.0
    risk_level:          str   = "low"
    domain:              str   = "general"
    fabrication_prone:   bool  = False
    false_premise:       bool  = False
    temporal_sensitive:  bool  = False
    domain_specialised:  bool  = False
    high_specificity:    bool  = False
    named_entity_count:  int   = 0
    conjunctive_load:    int   = 0
    threshold_adjust:    float = 0.0
    entity_check_needed: bool  = False
    flags:               list  = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "risk_score":          round(self.risk_score, 4),
            "risk_level":          self.risk_level,
            "domain":              self.domain,
            "fabrication_prone":   self.fabrication_prone,
            "false_premise":       self.false_premise,
            "temporal_sensitive":  self.temporal_sensitive,
            "domain_specialised":  self.domain_specialised,
            "high_specificity":    self.high_specificity,
            "named_entity_count":  self.named_entity_count,
            "conjunctive_load":    self.conjunctive_load,
            "threshold_adjust":    round(self.threshold_adjust, 4),
            "entity_check_needed": self.entity_check_needed,
            "flags":               self.flags,
        }


# =============================================================================
# SCORING
# =============================================================================

def _match_patterns(text: str, patterns: list) -> tuple:
    max_weight = 0.0
    hit_count  = 0
    text_lower = text.lower()
    for pattern, weight in patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            max_weight = max(max_weight, weight)
            hit_count += 1
    return max_weight, hit_count


def _count_named_entities(text: str) -> int:
    hits = set()
    for pattern in _NAMED_ENTITY_RE:
        for match in pattern.finditer(text):
            hits.add(match.group().strip())
    return len(hits)


def _count_specificity_markers(text: str) -> int:
    count = 0
    for pattern in _SPECIFICITY_RE:
        count += len(pattern.findall(text))
    return count


def score_prompt(prompt: str, prompt_type: str = "") -> EpistemicProfile:
    profile = EpistemicProfile()
    flags   = []

    # Domain detection — informs entity screener routing
    profile.domain = detect_domain(prompt)
    if profile.domain != "general":
        flags.append(f"domain_detected: {profile.domain}")

    fp_weight,   fp_count   = _match_patterns(prompt, _FABRICATION_PRONE_PATTERNS)
    fpr_weight,  fpr_count  = _match_patterns(prompt, _FALSE_PREMISE_PATTERNS)
    temp_weight, temp_count = _match_patterns(prompt, _TEMPORAL_PATTERNS)
    dom_weight,  dom_count  = _match_patterns(prompt, _DOMAIN_PATTERNS)
    conj_weight, conj_count = _match_patterns(prompt, _CONJUNCTIVE_PATTERNS)
    cf_weight,   cf_count   = _match_patterns(prompt, _COUNTERFACTUAL_PATTERNS)

    if fp_weight   > 0: profile.fabrication_prone  = True; flags.append(f"fabrication_prone_pattern (weight={fp_weight:.2f})")
    if fpr_weight  > 0: profile.false_premise       = True; flags.append(f"false_premise_indicator (weight={fpr_weight:.2f})")
    if temp_weight > 0: profile.temporal_sensitive  = True; flags.append(f"temporal_sensitivity (weight={temp_weight:.2f})")
    if dom_weight  > 0: profile.domain_specialised  = True; flags.append(f"domain_specialised (weight={dom_weight:.2f})")
    if conj_count  > 0: flags.append(f"conjunctive_load={conj_count}")
    if cf_weight   > 0: flags.append(f"counterfactual_framing (weight={cf_weight:.2f})")

    profile.conjunctive_load = conj_count

    ne_count = _count_named_entities(prompt)
    profile.named_entity_count = ne_count
    if ne_count >= 3:
        flags.append(f"high_named_entity_density (count={ne_count})")

    spec_count = _count_specificity_markers(prompt)
    if spec_count > 0:
        profile.high_specificity = True
        flags.append(f"high_specificity_markers (count={spec_count})")

    risk_score = (
        fp_weight  * 0.30 +
        fpr_weight * 0.25 +
        dom_weight * 0.15 +
        temp_weight * 0.10 +
        cf_weight  * 0.05 +
        min(ne_count / 10, 1.0) * 0.08 +
        min(spec_count / 5, 1.0) * 0.07
    )
    risk_score = min(risk_score + min(conj_count * 0.02, 0.08), 1.0)
    profile.risk_score = round(risk_score, 4)

    if risk_score < 0.15:   profile.risk_level = "low"
    elif risk_score < 0.35: profile.risk_level = "medium"
    elif risk_score < 0.60: profile.risk_level = "high"
    else:                   profile.risk_level = "critical"

    if risk_score < 0.15:   threshold_adjust = +0.03
    elif risk_score < 0.35: threshold_adjust = -0.02
    elif risk_score < 0.60: threshold_adjust = -0.05
    else:                   threshold_adjust = -0.08

    if prompt_type in ("reasoning", "ambiguous"):
        threshold_adjust *= 0.5

    profile.threshold_adjust = round(threshold_adjust, 4)

    profile.entity_check_needed = (
        ne_count > 0 and (
            prompt_type in ("factual", "hallucination", "") or
            profile.fabrication_prone or
            profile.false_premise
        )
    )

    profile.flags = flags
    return profile


def run_epistemic_scorer(prompt: str, prompt_type: str = "") -> EpistemicProfile:
    """Public interface. Never raises."""
    try:
        return score_prompt(prompt, prompt_type=prompt_type)
    except Exception:
        return EpistemicProfile()
