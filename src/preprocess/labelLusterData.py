import sys
import os
import json
import re
import string
import pandas as pd
from typing import Any, Dict, Iterable, List, Pattern, Set, Tuple

sys.path.append(os.environ.get('CONVLAB3_REPOSITORY_BASE_PATH'))
from convlab.util.unified_datasets_util import load_ontology


# =========================

# Constants & Regexes

# =========================

INT2WORD = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten",
}

# Generic patterns
PHONE_RE: Pattern = re.compile(r"\b(?:\+?\d[\d\s\-]{6,}\d)\b")
TIME_COLON_RE: Pattern = re.compile(r"\b\d{1,2}[:.]\d{2}\b")
REF_LIKE_RE: Pattern = re.compile(r"\b[A-Z0-9]{6,10}\b")

# Money-ish context to avoid false time flags
MONEY_NEAR_RE: Pattern = re.compile(
    r"(?:pounds?|gbp|£|\bper\b|\bticket\b|\bprice\b|\bcost\b|\beach\b|\bpp\b)",
    re.I,
)

# Domain-agnostic, high-signal keywords → (domain, slot)
GLOBAL_KEYWORD_SLOTS: List[Tuple[str, str, str]] = [
    ("entrance fee", "attraction", "entrance fee"),
    ("postcode", "attraction", "postcode"),
    ("postcode", "restaurant", "postcode"),
    ("reference number", "restaurant", "ref"),
    ("reference number", "hotel", "ref"),
    ("train id", "train", "train id"),
]

# Stopwords for ontology value scanning
ONTOLOGY_STOPWORDS: Set[str] = {
    "yes", "no",
    "hotel", "restaurant", "attraction",
    "place", "location", "free",
    "centre", "center", "north", "south", "east", "west",
    "the", "and", "of", "in",
    "guesthouse", "guest houses", "guest house", "hotel guesthouse",
    "town", "city",
}

WEEKDAYS: Set[str] = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
CUISINE_WORDS: Set[str] = {
    "indian", "chinese", "british", "italian", "french", "japanese",
    "korean", "thai", "turkish", "spanish", "mediterranean", "vietnamese",
    "mexican", "portuguese", "greek", "malaysian", "lebanese", "catalan",
}

# Streets for address heuristics
STREET_WORDS = "road|rd|street|st|lane|ln|avenue|ave|way|row|place|pl|close|cl|drive|dr|court|ct|hill|park|gardens|crescent|square"
EXTRA_ADDR_RE: Pattern = re.compile(rf"\b(?:\d+\s+)?[a-z]+(?:\s+[a-z]+)*\s+(?:{STREET_WORDS})\b")
ADDR_HINT_RE: Pattern = re.compile(rf"\b\d+\s+[a-z]+(?:\s+[a-z]+)*\s+(?:{STREET_WORDS})\b")

# Flexible "[number] [up to 3 words] (restaurants|trains|hotels|guesthouses)"
FLEX_NUM_NOUN = (
    r"\b\d+\s+(?:\w+\s+){0,3}?"
    r"(?:restaurant|restaurants|locations|branches|hotel|hotels|guesthouse|guesthouses|guest houses|train|trains)\b"
)

# Canonical slots we allow to match by name (used with triggers)
CRITICAL_CANONICAL_SLOTS: Dict[str, Set[str]] = {
    "taxi": {"leave at", "arrive by"},
    "train": {"leave at", "arrive by", "train id"},
    "restaurant": set(),
    "hotel": set(),
    "attraction": set(),
    "general": set(),
}

# Slot triggers per domain (used to detect slot mentions when DA doesn't license them)
DOMAIN_SLOT_TRIGGERS: Dict[str, Dict[str, List[str]]] = {
    "taxi": {
        "leave at": ["leave", "pickup time", "pick you up", "pick me up", "when to leave", "when should.*leave"],
        "arrive by": ["arrive by", "arrival time", "when (?:do|would) (?:you|i|we).*arrive", "arrive .* by"],
        "phone": ["contact number", "phone number", r"\b\d{7,}\b"],
        "type": ["car is", "vehicle is", "the car will be", "look out for", "lookout for"],
    },
    "train": {
        "leave at": ["leaves at", "departure time", "departs at", "when.*leave"],
        "arrive by": ["arrives at", "arrival time", "arrive by"],
        "train id": ["train id", r"\btr\d+\b"],
        "price": ["total fee", "price", "cost"],
        "duration": ["travel time", "journey time", "takes .* minutes"],
        "destination": ["to take you to", "where.*to", "destination"],
        "departure": ["from", "leaving from", "departure"],
        "day": ["what day"],
        "choice": [r"\b\d+\s+trains?\b", r"\bthere (?:are|re)\s+\d+\b", FLEX_NUM_NOUN],
    },
    "restaurant": {
        "food": ["serves", "cuisine", "type of cuisine", "type of food", "what.*type.*food"],
        "name": [
            "called", "named", "name is", "closest i can find is",
            "my favorite is", "favourite is", "favorite it", "favourite it",
            "recommend", "may i recommend",
            "i have the", "we have the", "there is a", "there is the",
            "i have", "we have", "there is", "there are",
        ],
        "area": [rf"(?:in|on|at)\s+the\s+(?:centre|center|north|south|east|west)(?:\s+(?:end|area))?", "part of town"],
        "price range": ["cheap", "moderately priced", "expensive", "price range"],
        "postcode": ["post code", "postcode"],
        "ref": ["reference number", r"\b[A-Z0-9]{6,10}\b"],
        "book time": [r"for \d{1,2}:\d{2}", r"at \d{1,2}:\d{2}", "time"],
        "book day": ["day", "what day", "date", "different day"],
        "book people": ["people", "guests", r"for \d+ people"],
        "book stay": ["nights?"],
        "stars": ["star rating", r"\b[1-5]\s*star"],
        "choice": [
            r"\b\d+\s+(?:restaurant|restaurants)\b",
            r"\bthere (?:are|re)\s+\d+\b",
            r"\b\d+\s+(?:locations|branches)\b",
            "two different locations",
            FLEX_NUM_NOUN,
        ],
    },
    "hotel": {
        "area": [rf"(?:in|on|at)\s+the\s+(?:centre|center|north|south|east|west)(?:\s+(?:end|area))?", "part of town"],
        "price range": ["cheap", "moderately priced", "expensive", "price range"],
        "postcode": ["post code", "postcode"],
        "ref": ["reference number", r"\b[A-Z0-9]{6,10}\b"],
        "book time": [r"at \d{1,2}:\d{2}", "time"],
        "book day": ["day", "what day", "date", "different day"],
        "book people": ["people", "guests", r"for \d+ people"],
        "book stay": ["nights?"],
        "stars": ["star request", "how many stars", "star rating", r"\b\d+\s*star"],
        "choice": [
            r"\b\d+\s+(?:hotel|hotels|guesthouse|guesthouses|guest houses)\b",
            r"\bthere (?:are|re)\s+\d+\b",
            r"\babout\s+\d+\b",
            FLEX_NUM_NOUN,
        ],
        "type": ["guesthouse", "guest house", "hostel", "inn", "bed and breakfast", "b and b", "bnb", "hotel"],
    },
    "attraction": {
        "name": ["called", "named", "is called", "recommend", "may i recommend", "is located", "located at", "there is", "there s", "is situated"],
        "type": ["architectural", "entertainment", "multiple sports", "park", "museum", "college", "colleges", "theater", "theatre", "theaters", "theatres"],
        "area": [
            rf"(?:in|on|at)\s+the\s+(?:centre|center|north|south|east|west)(?:\s+(?:end|area))?",
            r"\b(?:different|another)\s+\w*\s*area\b",
            r"\barea\?\b",
            "part of town", "neighborhood", "neighbourhood", "what area", "which area", "what part of town", "which part of town",
        ],
        "postcode": ["post code", "postcode"],
        "entrance fee": ["entrance is free", "free entry", "entrance fee", "free to get in", "free to get in!"],
    },
}

# Synonyms used to satisfy a slot (when DA value is empty) for missing detection
SLOT_SYNONYMS: Dict[str, List[str]] = {
    "food": ["cuisine", "type of cuisine", "type of food", "kind of food", "specific type of cuisine", "what cuisine"],
    "area": ["area", "part of town", "neighborhood", "neighbourhood", "which area", "what area"],
    "name": ["name", "what restaurant", "which restaurant", "restaurant are you looking for", "what is it called", "called", "spelled the name"],
    "book time": ["time", "what time", "different time", "time would you like"],
    "book day": ["day", "what day", "date", "different day"],
    "leave at": ["leave", "when should we leave", "when should i leave", "pickup time"],
    "arrive by": ["arrive", "arrival time", "arrives at", "arrive by"],
    "destination": ["where to", "to where"],
    "departure": ["leaving from", "from where"],
}

# Request-intent phrases per slot (subset of SLOT_SYNONYMS, tuned to questions)
REQUEST_PHRASES_BY_SLOT: Dict[str, List[str]] = {
    "area": ["what area", "which area", "what part of town", "which part of town", r"\b(?:different|another)\b.*\barea\b"],
    "food": ["what cuisine", "which cuisine", "type of cuisine", "what type of food", r"\b(?:different|another)\b.*\bcuisine\b"],
    "name": ["what restaurant", "which restaurant", "what is it called", "spelled the name"],
    "book time": ["what time", "which time", "different time", "time would you like"],
    "day": ["what day", "which day", "date", "different day"],
    "book day": ["what day", "which day", "date", "different day"],
    "book stay": ["how many nights", "how long", "shorter stay", r"\b\d+\s*nights?\b"],
    "leave at": ["what time", "when should we leave", "when should i leave", "pickup time"],
    "arrive by": ["what time", "when do.*arrive", "arrival time", "arrive by"],
    "destination": ["where to", "to where", "where are you going", "where will you be travelling to", "what is your destination"],
    "price range": ["price range", "preference on price", "preferred price range"],
    "departure": ["leaving from", "from where"],
    "stars": ["star request", "how many stars", "star rating", r"\b\d+\s*star"],
}

# Intent phrase lexicon
INTENT2PHRASES_LOCAL: Dict[str, List[str]] = {
    "reqmore": ["anything else", "something else", "anything more", "what else", "would you like more information", "would you like more info"],
    "bye": ["goodbye", "bye", "see you", "have a nice day", "have a great day", "have a great trip"],
    "greet": ["hello", "hi"],
    "deny": ["no", "nope", "not really"],
    "book": [
        r"\b(?:i(?:'| a)m|i(?:'| wi)ll|we(?:'| wi)ll)\s+book\b",
        r"\b(?:booked|booking)\s+(?:successfully|confirmed)\b",
        r"\breservation\s+(?:confirmed|made)\b",
        r"\bbooking\s+(?:confirmed|successful)\b",
    ],
    "offerbook": ["can i book", "may i book", "shall i book", "would you like a ticket", "can i get you a ticket"],
    "request": [r"\bwhat\b", r"\bwhich\b", r"\bwhen\b", r"\bwhere\b"],  # global question words (only when utterance has '?')
}

NAME_EXPOSING_TRIGGERS: List[str] = [
    "called", "named", "name is", "closest i can find is",
    "recommend", "may i recommend",
]

# =========================

# Utility functions

# =========================

def normalize_text(s: Any) -> str:
    """Lowercase, replace punctuation with spaces, collapse whitespace."""
    if s is None:
        return ""

    s = str(s).lower()
    s = re.sub(f"[{re.escape(string.punctuation)}]", " ", s)
    s = re.sub(r"\s+", " ", s)

    return s.strip()


def token_in_text(token: str, text: str) -> bool:
    """
    Robust matching against normalized `text`:
      1) exact word-boundary match
      2) plural-tolerant match (token + s/es)
      3) whitespace-insensitive for multiword tokens
      4) numeric-word bridge (e.g., "5" ↔ "five")
    """

    if not token or not text:
        return False

    def _has_regex(pat: str) -> bool:
        try:
            return re.search(pat, text) is not None

        except re.error:
            return False

    # exact
    if _has_regex(r"\b" + re.escape(token) + r"\b"):
        return True

    # plural tolerant
    if not token.endswith("s"):
        if _has_regex(r"\b" + re.escape(token) + r"(?:s|es)\b"):
            return True

    # multi-word (space-insensitive)
    if " " in token and token.replace(" ", "") in text.replace(" ", ""):
        return True

    # numeric↔word bridge
    if token in INT2WORD and INT2WORD[token] in text:
        return True

    return False


def _flex_time_token_pat(tkn: str) -> Pattern:
    """Make a regex that matches 16:50 / 16.50 / 16 50 from '16 50'."""
    m = re.fullmatch(r"(\d{1,2})\s+(\d{2})", tkn)
    if not m:
        return re.compile(re.escape(tkn), re.I)
    h, mm = m.groups()
    return re.compile(rf"\b{h}[:.\s]?{mm}\b", re.I)


def _near_money_units(cand: str, pat: Pattern, window: int = 12) -> bool:
    """Heuristic: treat time-like tokens as prices if near money-ish words."""
    for m in pat.finditer(cand):
        start, end = m.span()
        right = cand[end:end + window]
        left = cand[max(0, start - window):start]
        if MONEY_NEAR_RE.search(right) or MONEY_NEAR_RE.search(left):
            return True

    return False

def extract_candidate(rec: Dict[str, Any]) -> str:
    """
    Choose the utterance text to analyze:
      1) rec['utterance']
      2) rec['predictions']['utterance'|'prediction'|'generated']
      3) longest of top-level ['prediction','generated','text','gen']
    """

    v = rec.get("utterance")
    if isinstance(v, str) and v.strip():
        return v.strip()
    preds = rec.get("predictions", {})
    if isinstance(preds, dict):
        for k in ("utterance", "prediction", "generated"):
            pv = preds.get(k)
            if isinstance(pv, str) and pv.strip():
                return pv.strip()

    cands = [rec.get(k) for k in ("prediction", "generated", "text", "gen")]
    cands = [c for c in cands if isinstance(c, str) and c.strip()]

    return max(cands, key=len) if cands else ""

# =========================

# Ontology helpers (from ontology object)

# =========================


def build_val2ds_from_ontology_obj(ontology: Dict[str, Any]) -> Dict[str, List[str]]:
    """normalized ontology value -> list['domain-slot']"""
    val2ds: Dict[str, Set[str]] = {}
    for domain_name, domain in (ontology.get("domains") or {}).items():
        for slot_name, slot in (domain.get("slots") or {}).items():
            for v in (slot.get("possible_values") or []):
                if v is None:
                    continue
                vn = normalize_text(v)
                if not vn:
                    continue
                val2ds.setdefault(vn, set()).add(f"{normalize_text(domain_name)}-{normalize_text(slot_name)}")

    return {k: sorted(list(vs)) for k, vs in val2ds.items()}


def build_domain2slots_from_ontology_obj(ontology: Dict[str, Any]) -> Dict[str, List[str]]:
    """normalized domain -> list[normalized slots]"""
    d2s: Dict[str, Set[str]] = {}

    for domain_name, domain in (ontology.get("domains") or {}).items():
        slots = set()
        for slot_name in (domain.get("slots") or {}).keys():
            sn = normalize_text(slot_name)
            if sn:
                slots.add(sn)
        d2s[normalize_text(domain_name)] = slots

    return {d: sorted(list(slots)) for d, slots in d2s.items()}


# =========================

# DA → expected surfaces

# =========================


def collect_expected_alternatives(
    da: Dict[str, Any],
    intent2phrases: Dict[str, List[str]],
    slot_synonyms: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    """
    Per DA triple, build 'alts' (normalized strings) that count as satisfying the triple.
    Returns list of dicts: {'intent','domain','slot','value','alts':[...] }.
    """
    out: List[Dict[str, Any]] = []

    if not isinstance(da, dict):
        return out


    def add_unique(seq: Iterable[str]) -> List[str]:
        seen, acc = set(), []
        for s in seq:
            if s and s not in seen:
                seen.add(s)
                acc.append(s)

        return acc


    for bucket in ("categorical", "non-categorical", "binary"):
        for triple in da.get(bucket, []):
            if not isinstance(triple, dict):
                continue

            intent = (triple.get("intent") or "").strip()
            domain = (triple.get("domain") or "").strip()
            slot = (triple.get("slot") or "").strip()
            value = (triple.get("value") or "").strip()

            alts: List[str] = []
            if value:
                alts.append(normalize_text(value))
            else:
                if slot:
                    alts.append(normalize_text(slot))
                    for s in slot_synonyms.get(slot.lower(), []):
                        if s:
                            alts.append(normalize_text(s))
                    if intent:
                        alts.append(normalize_text(f"{intent} {slot}"))
                else:
                    if intent and intent.lower() in intent2phrases:
                        alts.extend(normalize_text(x) for x in intent2phrases[intent.lower()] if x)
                    if intent and domain:
                        alts.append(normalize_text(f"{intent} {domain}"))
                        alts.append(normalize_text(f"{domain} {intent}"))
            out.append({
                "intent": intent,
                "domain": normalize_text(domain),
                "slot": normalize_text(slot),
                "value": value,
                "alts": add_unique(alts),
            })

    return out


def build_expected_set(acts_with_alts: List[Dict[str, Any]], da: Dict[str, Any]) -> Set[str]:
    """Union of all alts + all raw slot/value tokens for quick membership checks."""
    ex: Set[str] = set()
    for a in acts_with_alts:
        ex.update(a.get("alts", []))
    for bucket in ("categorical", "non-categorical", "binary"):
        for t in da.get(bucket, []):
            if not isinstance(t, dict):
                continue

            v = normalize_text((t.get("value") or "").strip())
            s = normalize_text((t.get("slot") or "").strip())

            if v:
                ex.add(v)
            if s:
                ex.add(s)

    return ex


def da_bookkeeping(da: Dict[str, Any]) -> Tuple[Set[str], Set[str], Set[str], Dict[Tuple[str, str], Set[str]]]:
    """
    Returns:
      - ds_in_da: set of "domain-slot"
      - domains_in_da
      - slots_in_da
      - ds2intents: map (domain, slot) → set[intents]
    """

    ds_in_da: Set[str] = set()
    domains_in_da: Set[str] = set()
    slots_in_da: Set[str] = set()
    ds2intents: Dict[Tuple[str, str], Set[str]] = {}

    for bucket in ("categorical", "non-categorical", "binary"):
        for t in da.get(bucket, []):
            if not isinstance(t, dict):
                continue
            dn = normalize_text(t.get("domain") or "")
            sn = normalize_text(t.get("slot") or "")
            it = normalize_text(t.get("intent") or "")

            if dn:
                domains_in_da.add(dn)
            if sn:
                slots_in_da.add(sn)
            ds_in_da.add(f"{dn}-{sn}")
            if dn and sn and it:
                ds2intents.setdefault((dn, sn), set()).add(it)

    return ds_in_da, domains_in_da, slots_in_da, ds2intents


# =========================

# Detectors

# =========================


def detect_missing(acts_with_alts: List[Dict[str, Any]], cand_norm: str) -> Tuple[int, List[str]]:
    """Count acts whose alts don't appear in the utterance; return (missing_count, detected_mentions)."""
    missing = 0
    detected: List[str] = []
    for a in acts_with_alts:
        alts = a.get("alts", [])
        if not alts:
            continue

        hit = next((alt for alt in alts if alt and token_in_text(alt, cand_norm)), None)
        if hit:
            detected.append(hit)
        else:
            missing += 1

    return missing, detected



def slot_mentioned(domain: str, slot: str, cand_norm: str) -> bool:
    st = normalize_text(slot)
    if st in CRITICAL_CANONICAL_SLOTS.get(domain, set()) and token_in_text(st, cand_norm):
        return True

    for trig in DOMAIN_SLOT_TRIGGERS.get(domain, {}).get(slot, []):
        try:
            if any(ch in trig for ch in ".?*+()[]{}\\|^$"):
                if re.search(trig, cand_norm):
                    return True
            else:
                if token_in_text(normalize_text(trig), cand_norm):
                    return True
        except re.error:
            if trig in cand_norm:
                return True

    return False

# -------------------------

# A) Ontology value redundancies

# -------------------------


def detect_ontology_values(
    cand_norm: str,
    expected_set: Set[str],
    ds2intents: Dict[Tuple[str, str], Set[str]],
    val2ds: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for val_norm, ds_entry in val2ds.items():
        if not val_norm or len(val_norm) < 3 or val_norm in ONTOLOGY_STOPWORDS:
            continue

        if any(token_in_text(val_norm, ev) or token_in_text(ev, val_norm) or ev == val_norm for ev in expected_set):
            continue

        if val_norm in WEEKDAYS:
            if any("request" in intents for intents in ds2intents.values()):
                continue

        if token_in_text(val_norm, cand_norm):
            out.append({"type": "ontology_value", "value": val_norm, "ds": list(ds_entry)})

    return out


def detect_cuisine_fallback(
    cand_norm: str,
    expected_set: Set[str],
    domains_in_da: Set[str],
    slots_in_da: Set[str],
) -> List[Dict[str, Any]]:

    out: List[Dict[str, Any]] = []
    if "restaurant" in domains_in_da and "food" not in slots_in_da:
        for w in CUISINE_WORDS:
            wn = normalize_text(w)
            if any(token_in_text(wn, ev) or token_in_text(ev, wn) or ev == wn for ev in expected_set):
                continue

            if token_in_text(wn, cand_norm):
                out.append({"type": "ontology_value", "value": wn, "ds": ["restaurant-food"]})
    return out


def detect_both_and_name_hint(
    cand_norm: str,
    domains_in_da: Set[str],
    slots_in_da: Set[str],
) -> List[Dict[str, Any]]:

    out: List[Dict[str, Any]] = []
    if "restaurant" in domains_in_da and "name" not in slots_in_da:
        if re.search(r"\bboth\s+[^?!.]{1,80}?\s+and\s+[^?!.]{1,80}\b", cand_norm):
            out.append({"type": "ontology_slot", "domain": "restaurant", "value": "name"})

    return out


# -------------------------

# B) Ontology slot redundancies (triggers)

# -------------------------


def detect_slot_triggers(
    cand_norm: str,
    expected_set: Set[str],
    ds_in_da: Set[str],
    domains_in_da: Set[str],
    slots_in_da: Set[str],
    ds2intents: Dict[Tuple[str, str], Set[str]],
    has_booking_confirmation: bool,
) -> List[Dict[str, Any]]:

    out: List[Dict[str, Any]] = []
    for d in domains_in_da:
        candidate_slots = set(DOMAIN_SLOT_TRIGGERS.get(d, {}).keys()) | CRITICAL_CANONICAL_SLOTS.get(d, set())

        for slot in candidate_slots:
            st = normalize_text(slot)
            if not st or st in {"the", "and", "of"}:
                continue

            slot_is_in_da = (f"{d}-{st}" in ds_in_da) or (st in slots_in_da)
            intents_for_slot = ds2intents.get((d, st), set())
            request_only = slot_is_in_da and intents_for_slot and intents_for_slot == {"request"}
            if slot_is_in_da and not request_only:
                continue

            if d == "taxi" and st in {"arrive by", "leave at"} and has_booking_confirmation:
                continue

            name_exposure = False
            if st == "name":
                for trig in NAME_EXPOSING_TRIGGERS:
                    try:
                        if any(ch in trig for ch in ".?*+()[]{}\\|^$"):
                            if re.search(trig, cand_norm):
                                name_exposure = True
                                break
                        else:
                            if token_in_text(normalize_text(trig), cand_norm):
                                name_exposure = True
                                break
                    except re.error:
                        if trig in cand_norm:
                            name_exposure = True
                            break
            if not name_exposure:
                if any(token_in_text(st, ev) or token_in_text(ev, st) or ev == st for ev in expected_set):
                    continue
            if d == "attraction" and st == "name":
                if "address" in slots_in_da or ADDR_HINT_RE.search(cand_norm):
                    continue
            if slot_mentioned(d, st, cand_norm):
                out.append({"type": "ontology_slot", "domain": d, "value": st})

    return out


def detect_global_keywords(
    cand_norm: str,
    expected_set: Set[str],
    ds_in_da: Set[str],
    slots_in_da: Set[str],
) -> List[Dict[str, Any]]:

    out: List[Dict[str, Any]] = []
    for kw, dom, slot in GLOBAL_KEYWORD_SLOTS:
        kw_n = normalize_text(kw)
        ds_key = f"{normalize_text(dom)}-{normalize_text(slot)}"
        if ds_key in ds_in_da or normalize_text(slot) in slots_in_da:
            continue
        if token_in_text(kw_n, cand_norm):
            out.append({"type": "ontology_slot", "domain": normalize_text(dom), "value": normalize_text(slot)})

    return out


# -------------------------

# C) Pattern extras

# -------------------------


def detect_pattern_extras(
    cand_raw: str,
    cand_norm: str,
    expected_set: Set[str],
    slots_in_da: Set[str],
) -> List[Dict[str, Any]]:

    out: List[Dict[str, Any]] = []
    if "phone" not in slots_in_da:
        for ph in PHONE_RE.findall(cand_raw or ""):
            phn = normalize_text(ph)
            if not any(token_in_text(phn, ev) or token_in_text(ev, phn) or ev == phn for ev in expected_set):
                out.append({"type": "extra_phone", "value": ph})
    if "ref" not in slots_in_da:
        for refm in REF_LIKE_RE.findall(cand_raw or ""):
            rn = normalize_text(refm)
            if rn and not any(token_in_text(rn, ev) or token_in_text(ev, rn) or ev == rn for ev in expected_set):
                out.append({"type": "extra_ref_like", "value": refm})
    if "address" in slots_in_da:
        for addr in EXTRA_ADDR_RE.findall(cand_norm):
            if not any(token_in_text(addr, ev) or token_in_text(ev, addr) or ev == addr for ev in expected_set):
                out.append({"type": "extra_address", "value": addr})

    return out


def detect_extra_time(
    cand_raw: str,
    expected_set: Set[str],
    slots_in_da: Set[str],
) -> List[Dict[str, Any]]:

    out: List[Dict[str, Any]] = []
    expected_time_digits = {re.sub(r"\D", "", ev) for ev in expected_set if ev}
    cand_times = [m.group(0) for m in TIME_COLON_RE.finditer(cand_raw)]
    for tkn in cand_times:
        pat = _flex_time_token_pat(normalize_text(tkn))
        if _near_money_units(cand_raw, pat) or "price" in slots_in_da:
            continue
        tdigits = re.sub(r"\D", "", tkn)
        if tdigits and tdigits not in expected_time_digits:
            out.append({"type": "extra_time", "value": tkn})

    return out


# -------------------------

# D) Intent phrase redundancies

# -------------------------


def detect_intent_phrases(
    cand_raw: str,
    cand_norm: str,
    acts_with_alts: List[Dict[str, Any]],
    redundant_details_so_far: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    out: List[Dict[str, Any]] = []
    intents_in_da = {normalize_text(a.get("intent", "")) for a in acts_with_alts if a.get("intent")}
    is_question = "?" in cand_raw
    slot_request_already = any(d.get("type") == "intent_phrase" and d.get("intent") == "request" for d in redundant_details_so_far)

    for intent, phrases in INTENT2PHRASES_LOCAL.items():
        for ph in phrases:
            if intent == "request" and (not is_question or slot_request_already):
                continue
            phn = normalize_text(ph)
            if not phn or (len(phn) < 4 and len(phn.split()) == 1):
                continue
            try:
                is_regex = any(ch in ph for ch in ".?*+()[]{}\\|^$")
                matched = bool(re.search(ph, cand_norm)) if is_regex else token_in_text(phn, cand_norm)
            except re.error:
                matched = phn in cand_norm
            if intent == "bye" and {"greet", "welcome"} & intents_in_da:
                continue
            if intent == "reqmore" and "bye" in intents_in_da:
                continue
            if matched and (intent not in intents_in_da):
                out.append({"type": "intent_phrase", "intent": intent, "phrase": phn})
                break

    return out


def detect_slot_specific_requests(
    cand_norm: str,
    domains_in_da: Set[str],
    ds2intents: Dict[Tuple[str, str], Set[str]],
    domain2slots: Dict[str, List[str]],
) -> List[Dict[str, Any]]:

    out: List[Dict[str, Any]] = []
    for d in domains_in_da:
        valid_slots = set(domain2slots.get(d, []))
        for slot, phrases in REQUEST_PHRASES_BY_SLOT.items():
            st = normalize_text(slot)
            if st not in valid_slots:
                continue
            intents = ds2intents.get((d, st), set())
            if "request" in intents:
                continue
            if intents and "request" not in intents:
                continue
            for ph in phrases:
                phn = normalize_text(ph)
                try:
                    is_regex = any(ch in ph for ch in ".?*+()[]{}\\|^$")
                    matched = bool(re.search(ph, cand_norm)) if is_regex else token_in_text(phn, cand_norm)
                except re.error:
                    matched = phn in cand_norm
                if matched:
                    out.append({"type": "intent_phrase", "intent": "request", "domain": d, "slot": st, "phrase": phn})
                    break

    return out


# =========================

# Aggregating analyzer (no per-utterance rows)

# =========================

def create_hallucination_labels(
    records: List[Dict[str, Any]],
    val2ds: Dict[str, List[str]],
    domain2slots: Dict[str, List[str]],
    use_original_acts: bool = False,
) -> Tuple[pd.Series, List]:

    labels: List = []
    allRedundantDetails: List = []
    for rec in records:
        # choose DA field
        if use_original_acts:
            da = rec.get("original_dialogue_acts") or {}
        else:
            da = rec.get("dialogue_acts") or {}

        # unconditional empty-DA filtering (parity)
        acts_size = 0
        if isinstance(da, dict):
            acts_size = sum(len(da.get(b, [])) for b in ("categorical", "non-categorical", "binary"))
        if acts_size == 0:
            labels.append(0)
            allRedundantDetails.append([])
            continue

        cand = extract_candidate(rec)
        cand_norm = normalize_text(cand)
        has_booking_confirmation = bool(re.search(r"\b(book(?:ed|ing)?|reservation\s+(?:made|confirmed)|booking\s+(?:confirmed|successful))\b", cand_norm))

        # expected surfaces & bookkeeping
        acts_with_alts = collect_expected_alternatives(da, INTENT2PHRASES_LOCAL, SLOT_SYNONYMS)
        _, _detected_mentions = detect_missing(acts_with_alts, cand_norm)
        expected_set = build_expected_set(acts_with_alts, da)
        ds_in_da, domains_in_da, slots_in_da, ds2intents = da_bookkeeping(da)

        # redundancy ( = hallucination )
        redundant_details: List[Dict[str, Any]] = []
        redundant_details += detect_ontology_values(cand_norm, expected_set, ds2intents, val2ds)
        redundant_details += detect_cuisine_fallback(cand_norm, expected_set, domains_in_da, slots_in_da)
        redundant_details += detect_both_and_name_hint(cand_norm, domains_in_da, slots_in_da)
        redundant_details += detect_slot_triggers(
            cand_norm, expected_set, ds_in_da, domains_in_da, slots_in_da, ds2intents, has_booking_confirmation
        )
        redundant_details += detect_global_keywords(cand_norm, expected_set, ds_in_da, slots_in_da)
        redundant_details += detect_pattern_extras(rec.get("candidate", "") or cand, cand_norm, expected_set,
                                                   slots_in_da)
        redundant_details += detect_extra_time(cand, expected_set, slots_in_da)
        redundant_details += detect_intent_phrases(cand, cand_norm, acts_with_alts, redundant_details)
        redundant_details += detect_slot_specific_requests(cand_norm, domains_in_da, ds2intents, domain2slots)
        allRedundantDetails.append(redundant_details)
        redundant_count = len(redundant_details)

        if redundant_count > 0:
            labels.append(1)
        else:
            labels.append(0)

    return pd.Series(data = labels), allRedundantDetails


def load_data_and_create_hallu_labels(predict_result: str) -> Tuple[pd.Series, List]:
    """
      - predict_result: path to predictions file (JSON array; strict like upstream)
      - returns hallucination labels and list of all redundancies
    """

    # Strict JSON array load (parity)
    records = json.load(open(predict_result, "r", encoding="utf-8"))

    # Diagnostics aggregation
    ontology = load_ontology("multiwoz21")
    val2ds = build_val2ds_from_ontology_obj(ontology)
    domain2slots = build_domain2slots_from_ontology_obj(ontology)
    labels, allRedundancies = create_hallucination_labels(records, val2ds, domain2slots, use_original_acts=False)

    return labels, allRedundancies