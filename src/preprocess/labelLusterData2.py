"""
Script for labeling LusterData based on ConvLab3/convlab/nlgs/analyze_systematic.py
"""

from __future__ import annotations

import json
import sys
import os
import re
import string
import importlib
import types
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Set, Tuple

import yaml
sys.path.append(os.environ.get('CONVLAB3_REPOSITORY_BASE_PATH'))
from convlab.util.unified_datasets_util import load_ontology  # external dep

import spacy
from spacy.matcher import Matcher
from spacy.tokens import Doc

try:
    from tqdm import tqdm
except ImportError:
    print("tqdm not found. Please run: pip install tqdm")


    class tqdm:
        def __init__(self, iterable, *args, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def update(self, *args):
            pass

# =========================
# Globals (Config-dependent)
# =========================

config: types.ModuleType = None
SEMANTIC_TRIGGER_VECTORS: Dict[str, List[Doc]] = {}

# =========================
# Load NLP Model
# =========================
try:
    NLP_MODEL = "/home/kaali100/.local/lib/python3.12/site-packages/en_core_web_lg/en_core_web_lg-3.8.0"
    nlp = spacy.load(NLP_MODEL)
    print(f"Loaded spaCy '{NLP_MODEL}' model.")
    if not nlp.has_pipe("sentencizer"):
        nlp.add_pipe("sentencizer")

    if not Doc.has_extension("domain"):
        Doc.set_extension("domain", default=None)

    if not Doc.has_extension("slots_in_da"):
        Doc.set_extension("slots_in_da", default=set())

except IOError:
    print(f"Error: spaCy model '{NLP_MODEL}' not found.")
    print(f"Please run: python -m spacy download {NLP_MODEL}")
    exit(1)


# =========================
# Config Loading
# =========================

def load_config_and_cache_vectors(config_name: str):
    """
    Dynamically loads the specified config module and caches semantic vectors.
    """
    global config, SEMANTIC_TRIGGER_VECTORS

    print(f"Loading configuration from: {config_name}.py")
    try:
        config = importlib.import_module(config_name)
    except ImportError:
        print(f"Error: Could not import config file '{config_name}.py'.")
        print("Please ensure the file exists and is in the same directory.")
        exit(1)

    print("Caching semantic trigger vectors...")
    for key, lemmas in config.SEMANTIC_SLOT_TRIGGERS.items():
        SEMANTIC_TRIGGER_VECTORS[key] = list(nlp.pipe(lemmas, disable=["parser", "ner"]))
    print("Semantic vectors cached.")


# =========================
# Span-Based Dataclass
# =========================
@dataclass(frozen=True)
class Span:
    start_char: int
    end_char: int
    type: str
    value: str

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def overlaps(self, other: Span) -> bool:
        return not (self.end_char <= other.start_char or other.end_char <= self.start_char)


# =========================
# Utility functions
# =========================
_ontology_cache: Dict[str, Dict[str, List[str]]] = {}
_domain2slots_cache: Dict[str, List[str]] = {}


def read_yaml_config(path: str) -> dict:
    """Reads the main YAML config file for the script."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_json_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read().strip()
        if not text:
            return []
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for ln in f:
            ln = ln.strip();
            if ln: records.append(json.loads(ln))
    return records


def normalize_text(s: Any) -> str:
    s = str(s).lower()
    s = re.sub(r"(\d):(\d)", r"\1 \2", s)
    s = re.sub(f"[{re.escape(string.punctuation)}]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def get_lemmatized_set(phrases: Iterable[str]) -> Set[str]:
    """
    Returns a set of *individual token lemmas* from all phrases.
    """
    lemmatized_set = set()
    docs = nlp.pipe([normalize_text(p) for p in phrases if p], disable=["parser", "ner"])
    for doc in docs:
        for token in doc:
            lemma = token.lemma_
            if token.pos_ == "NUM" and token.text.isdigit():
                lemma = token.text
            if lemma and len(lemma) > 1 and lemma not in config.ONTOLOGY_STOPWORDS:
                lemmatized_set.add(lemma)
    return lemmatized_set


def extract_candidate(rec: Dict[str, Any]) -> str:
    v = rec.get("utterance")
    if isinstance(v, str) and v.strip(): return v.strip()
    preds = rec.get("predictions", {})
    if isinstance(preds, dict):
        for k in ("utterance", "prediction", "generated"):
            pv = preds.get(k);
            if isinstance(pv, str) and pv.strip(): return pv.strip()
    cands = [rec.get(k) for k in ("prediction", "generated", "text", "gen")]
    cands = [c for c in cands if isinstance(c, str) and c.strip()]
    return max(cands, key=len) if cands else ""


def build_val2ds_from_ontology(dataset_name: str) -> Dict[str, List[str]]:
    """
    Caches *single token lemmas* as keys.
    """
    onto = load_ontology(dataset_name)
    val2ds: Dict[str, Set[str]] = {}

    values_to_process = []
    for domain_name, domain in (onto.get("domains") or {}).items():
        for slot_name, slot in (domain.get("slots") or {}).items():
            for v in (slot.get("possible_values") or []):
                if v is None: continue
                values_to_process.append((normalize_text(v), f"{domain_name}-{slot_name}"))

    texts = [v[0] for v in values_to_process]
    docs = nlp.pipe(texts, disable=["parser", "ner"])

    for doc, (original_text, domain_slot) in tqdm(zip(docs, values_to_process), total=len(values_to_process),
                                                  desc="Lemmatizing ontology"):
        for token in doc:
            lemma = token.lemma_
            if token.pos_ == "NUM" and token.text.isdigit():
                lemma = token.text
            if lemma and len(lemma) > 1 and lemma not in config.ONTOLOGY_STOPWORDS:
                val2ds.setdefault(lemma, set()).add(domain_slot)

    return {k: sorted(list(vs)) for k, vs in val2ds.items()}


def build_domain2slots_from_ontology(dataset_name: str) -> Dict[str, List[str]]:
    onto = load_ontology(dataset_name)
    d2s: Dict[str, Set[str]] = {}
    for domain_name, domain in (onto.get("domains") or {}).items():
        slots = set()
        for slot_name in (domain.get("slots") or {}).keys():
            sn = normalize_text(slot_name)
            if sn: slots.add(sn)
        d2s[normalize_text(domain_name)] = slots
    return {d: sorted(list(slots)) for d, slots in d2s.items()}


def get_val2ds(dataset_name: str) -> Dict[str, List[str]]:
    if dataset_name not in _ontology_cache:
        print("\nBuilding lemmatized ontology cache...")
        _ontology_cache[dataset_name] = build_val2ds_from_ontology(dataset_name)
        print("Ontology cache built.")
    return _ontology_cache[dataset_name]


def get_domain2slots(dataset_name: str) -> Dict[str, List[str]]:
    if dataset_name not in _domain2slots_cache:
        _domain2slots_cache[dataset_name] = build_domain2slots_from_ontology(dataset_name)
    return _domain2slots_cache[dataset_name]


def collect_expected_alternatives(da: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Collects raw, normalized phrases and lemmatized synonyms.
    """
    out: List[Dict[str, Any]] = []
    if not isinstance(da, dict): return out

    synonym_lemmas: Dict[str, List[str]] = {}
    for slot, phrases in config.SLOT_SYNONYMS.items():
        synonym_lemmas[slot] = list(get_lemmatized_set(phrases))

    for bucket in ("categorical", "non-categorical", "binary"):
        for triple in da.get(bucket, []):
            if not isinstance(triple, dict): continue
            intent = (triple.get("intent") or "").strip().lower()
            domain = (triple.get("domain") or "").strip().lower()
            slot = (triple.get("slot") or "").strip().lower()
            value = (triple.get("value") or "").strip()

            alts: List[str] = []
            if value: alts.append(normalize_text(value))
            if slot:
                alts.append(normalize_text(slot))
                alts.extend(synonym_lemmas.get(slot, []))
            if intent:
                alts.extend([normalize_text(p) for p in config.INTENT2PHRASES_LOCAL.get(intent, [])])

            out.append({"intent": intent, "domain": domain, "slot": slot, "value": value,
                        "alts": sorted(list(set(a for a in alts if a)))})
    return out


def build_expected_set(acts_with_alts: List[Dict[str, Any]], da: Dict[str, Any]) -> Set[str]:
    ex: Set[str] = set()
    for a in acts_with_alts:
        ex.update(a.get("alts", []))
    for bucket in ("categorical", "non-categorical", "binary"):
        for t in da.get(bucket, []):
            if not isinstance(t, dict): continue
            v = normalize_text((t.get("value") or "").strip())
            s = normalize_text((t.get("slot") or "").strip())
            if v: ex.add(v)
            if s: ex.add(s)
    # Add individual lemmas from the values for robust licensing
    ex.update(get_lemmatized_set(ex))
    return ex


def da_bookkeeping(da: Dict[str, Any]) -> Tuple[Set[str], Set[str], Set[str], Set[str]]:
    ds_in_da: Set[str] = set()
    domains_in_da: Set[str] = set()
    slots_in_da: Set[str] = set()
    intents_in_da: Set[str] = set()

    for bucket in ("categorical", "non-categorical", "binary"):
        for t in da.get(bucket, []):
            if not isinstance(t, dict): continue
            dn = normalize_text(t.get("domain") or "")
            sn = normalize_text(t.get("slot") or "")
            it = normalize_text(t.get("intent") or "")
            if dn: domains_in_da.add(dn)
            if sn: slots_in_da.add(sn)
            if it: intents_in_da.add(it)
            ds_in_da.add(f"{dn}-{sn}")
    return ds_in_da, domains_in_da, slots_in_da, intents_in_da


# =========================
# SYSTEMATIC DETECTORS
# =========================

def license_spans(cand_norm: str, expected_set_norm: Set[str]) -> Set[Span]:
    consumed_spans: Set[Span] = set()
    for exp_phrase in sorted(list(expected_set_norm), key=len, reverse=True):
        if not exp_phrase: continue
        try:
            for m in re.finditer(re.escape(exp_phrase), cand_norm):
                span = Span(m.start(), m.end(), "licensed", exp_phrase)
                if not any(span.overlaps(consumed) for consumed in consumed_spans):
                    consumed_spans.add(span)
        except re.error:
            continue
    return consumed_spans


def detect_ner(doc: Doc, slots_in_da: Set[str]) -> List[Span]:
    spans: List[Span] = []
    for ent in doc.ents:
        if ent.label_ == "DATE" and ent.text.lower() in config.POLITE_NER_DATES:
            continue

        possible_slots = config.SPACY_LABEL_TO_SLOTS.get(ent.label_, set())
        if not possible_slots: continue

        is_licensed = any(s in slots_in_da for s in possible_slots)
        if ent.label_ == "CARDINAL":
            if "stars" in slots_in_da or "book people" in slots_in_da or "choice" in slots_in_da:
                is_licensed = True

        if not is_licensed:
            spans.append(Span(ent.start_char, ent.end_char, f"ner_{ent.label_.lower()}", ent.text))
    return spans


def detect_ontology_values_lemmatized(doc: Doc, val2ds: Dict[str, List[str]], expected_set_lemmatized: Set[str]) -> \
List[Span]:
    spans: List[Span] = []
    for token in doc:
        lemma = token.lemma_
        if not lemma:
            continue

        if token.pos_ == "NUM" and token.text.isdigit():
            lemma = token.text

        if lemma in val2ds and lemma not in expected_set_lemmatized:
            if lemma in config.WEEKDAYS: continue
            spans.append(Span(token.idx, token.idx + len(token.text), "ontology_value", token.text))
    return spans


def detect_patterns(cand_raw: str, cand_norm: str, slots_in_da: Set[str]) -> List[Span]:
    spans: List[Span] = []
    if "phone" not in slots_in_da:
        for m in config.PHONE_RE.finditer(cand_raw):
            spans.append(Span(m.start(), m.end(), "pattern_phone", m.group(0)))
    if "ref" not in slots_in_da:
        for m in config.REF_LIKE_RE.finditer(cand_raw):
            spans.append(Span(m.start(), m.end(), "pattern_ref", m.group(0)))

    licensed_time = any(s in config.AMBIGUOUS_TIME_SLOTS for s in slots_in_da)
    if not licensed_time:
        for m in config.TIME_COLON_RE.finditer(cand_norm):
            start, end = m.span()
            window = 12
            right = cand_norm[end:end + window]
            left = cand_norm[max(0, start - window):start]
            if "gbp" in right or "pound" in right or "gbp" in left or "pound" in left:
                continue
            spans.append(Span(m.start(), m.end(), "pattern_time", m.group(0)))
    return spans


def build_intent_matcher(nlp_vocab: Any, intents_in_da: Set[str]) -> Matcher:
    matcher = Matcher(nlp_vocab)
    for intent, phrases in config.INTENT2PHRASES_LOCAL.items():
        if intent not in intents_in_da:
            for phrase in phrases:
                pattern = [{"LOWER": w} for w in phrase.split()]
                matcher.add(intent, [pattern])
    return matcher


def detect_intent_phrases(doc: Doc, matcher: Matcher) -> List[Span]:
    matched_spans: Set[Span] = set()
    matches = matcher(doc)
    for match_id, start_token, end_token in matches:
        intent_name = doc.vocab.strings[match_id]
        span = doc[start_token:end_token]

        if intent_name == "deny" and span.text.lower() == "no" and len(doc) > 3:
            continue

        if intent_name == "bye" and any(t.lemma_ in {"greet", "welcome"} for t in doc): continue
        if intent_name == "reqmore" and any(t.lemma_ == "bye" for t in doc): continue

        current_span = Span(span.start_char, span.end_char, f"intent_{intent_name}", span.text)
        if not any(current_span.overlaps(s) for s in matched_spans):
            matched_spans.add(current_span)
    return list(matched_spans)


def build_slot_trigger_matcher(nlp_vocab: Any, ds_in_da: Set[str]) -> Matcher:
    matcher = Matcher(nlp_vocab)
    for key, patterns in config.MATCHER_SLOT_TRIGGERS.items():
        domain, slot = key.split('-', 1)
        norm_key = f"{normalize_text(domain)}-{normalize_text(slot)}"
        if norm_key not in ds_in_da:
            matcher.add(key, patterns)
    return matcher


def detect_slot_triggers(doc: Doc, matcher: Matcher) -> List[Span]:
    matched_spans: Set[Span] = set()
    matches = matcher(doc)
    for match_id, start_token, end_token in matches:
        key_name = doc.vocab.strings[match_id]
        span = doc[start_token:end_token]
        current_span = Span(span.start_char, span.end_char, f"trigger_{key_name}", span.text)
        if not any(current_span.overlaps(s) for s in matched_spans):
            matched_spans.add(current_span)
    return list(matched_spans)


def detect_dependency_patterns(doc: Doc, ds_in_da: Set[str]) -> List[Span]:
    spans: Set[Span] = set()

    for token in doc:
        if "choice" not in ds_in_da and token.pos_ == "NUM":
            if token.head.lemma_ in config.CHOICE_NOUNS_LEMMA:
                span = doc[token.i: token.head.i + 1]
                spans.add(Span(span.start_char, span.end_char, "dep_choice", span.text))

        domain = doc._.domain or "general"
        stars_key = f"{domain}-stars"
        if stars_key not in ds_in_da and token.lemma_ == "star":
            for child in token.children:
                if child.dep_ == "nummod":
                    span_start = min(child.i, token.i)
                    span_end = max(child.i, token.i)
                    span = doc[span_start: span_end + 1]
                    spans.add(Span(span.start_char, span.end_char, f"dep_stars", span.text))
                    break

    return list(spans)


def build_request_matcher(nlp_vocab: Any, slots_in_da: Set[str]) -> Matcher:
    """
    Builds a Matcher for all *unlicensed* requestable slot phrases.
    """
    matcher = Matcher(nlp_vocab)
    for slot, patterns in config.REQUEST_PHRASES_BY_SLOT.items():
        if slot not in slots_in_da:
            matcher.add(slot, patterns)
    return matcher


def detect_redundant_requests(doc: Doc, matcher: Matcher) -> List[Span]:
    """
    Runs the request matcher to find keyword-based slot requests.
    """
    matched_spans: Set[Span] = set()
    matches = matcher(doc)
    for match_id, start_token, end_token in matches:
        slot_name = doc.vocab.strings[match_id]
        span = doc[start_token:end_token]

        if slot_name in config.AMBIGUOUS_TIME_SLOTS:
            if any(s in config.AMBIGUOUS_TIME_SLOTS for s in doc._.slots_in_da):
                continue

        current_span = Span(span.start_char, span.end_char, f"request_{slot_name}", span.text)
        if not any(current_span.overlaps(s) for s in matched_spans):
            matched_spans.add(current_span)
    return list(matched_spans)


def detect_semantic_triggers(doc: Doc, slots_in_da: Set[str]) -> List[Span]:
    """
    Detects triggers using word vector similarity.
    """
    spans: List[Span] = []

    unlicensed_slots_to_check: Set[str] = set()
    for slot in config.SEMANTIC_SLOT_TRIGGERS.keys():
        if slot not in slots_in_da:
            unlicensed_slots_to_check.add(slot)

    if not unlicensed_slots_to_check:
        return []

    for token in doc:
        if (token.pos_ not in {"VERB", "NOUN", "ADJ"} or
                token.is_stop or not token.has_vector or not token.vector_norm or
                token.lemma_ in config.ONTOLOGY_STOPWORDS):
            continue

        for slot in unlicensed_slots_to_check:
            trigger_vectors = SEMANTIC_TRIGGER_VECTORS[slot]
            for trigger_vec in trigger_vectors:
                if token.similarity(trigger_vec) > config.SIMILARITY_THRESHOLD:
                    spans.append(Span(token.idx, token.idx + len(token.text), f"sem_{slot}", token.text))
                    break

    return spans


# =========================
# Main analyzer
# =========================

def analyze_records(
        records: List[Dict[str, Any]],
        dataset_name: str,
        filter_empty_acts: bool = True,
        use_original_acts: bool = False,
) -> Tuple[pd.Series, List]:

    val2ds = get_val2ds(dataset_name)

    print(f"Processing {len(records)} candidates with nlp.pipe (NER, Parser, etc.)...")
    candidates = [extract_candidate(rec) for rec in records]
    docs = list(tqdm(nlp.pipe(candidates), total=len(candidates), desc="Running NLP pipeline"))
    print("Candidate processing complete.")

    allRedundantDetails: List = []
    labels: List = []
    for idx, rec in enumerate(tqdm(records, desc="Analyzing records")):
        if use_original_acts:
            da = rec.get("original_dialogue_acts") or {}
        else:
            da = rec.get("dialogue_acts") or {}

        if filter_empty_acts:
            acts_size = sum(len(da.get(b, [])) for b in ("categorical", "non-categorical", "binary"))
            if acts_size == 0:
                allRedundantDetails.append([])
                labels.append(0)
                continue

        cand = candidates[idx]
        cand_norm = normalize_text(cand)
        cand_doc = docs[idx]

        acts_with_alts = collect_expected_alternatives(da)
        expected_set_norm_phrases = build_expected_set(acts_with_alts, da)
        expected_set_lemmatized = get_lemmatized_set(expected_set_norm_phrases)

        ds_in_da, domains_in_da, slots_in_da, intents_in_da = da_bookkeeping(da)

        primary_domain = next(iter(domains_in_da), "general")
        cand_doc._.domain = primary_domain
        cand_doc._.slots_in_da = slots_in_da

        consumed_spans = license_spans(cand_norm, expected_set_norm_phrases)

        intent_matcher = build_intent_matcher(nlp.vocab, intents_in_da)
        slot_trigger_matcher = build_slot_trigger_matcher(nlp.vocab, ds_in_da)
        request_matcher = build_request_matcher(nlp.vocab, slots_in_da)

        all_potential_errors: List[Span] = []
        all_potential_errors.extend(detect_ner(cand_doc, slots_in_da))
        all_potential_errors.extend(detect_ontology_values_lemmatized(cand_doc, val2ds, expected_set_lemmatized))
        all_potential_errors.extend(detect_patterns(cand, cand_norm, slots_in_da))
        all_potential_errors.extend(detect_intent_phrases(cand_doc, intent_matcher))
        all_potential_errors.extend(detect_slot_triggers(cand_doc, slot_trigger_matcher))
        all_potential_errors.extend(detect_dependency_patterns(cand_doc, ds_in_da))
        all_potential_errors.extend(detect_semantic_triggers(cand_doc, slots_in_da))
        all_potential_errors.extend(detect_redundant_requests(cand_doc, request_matcher))

        redundant_details: List[Span] = []
        for err_span in sorted(all_potential_errors, key=lambda s: s.start_char):
            is_consumed = any(err_span.overlaps(consumed) for consumed in consumed_spans)
            if not is_consumed:
                redundant_details.append(err_span)
                consumed_spans.add(err_span)

        allRedundantDetails.append([s.as_dict() for s in redundant_details])
        if len(redundant_details) == 0:
            labels.append(0)
        else:
            labels.append(1)

    return pd.Series(data = labels), allRedundantDetails


def load_data_and_create_hallu_labels(predict_result: str) -> Tuple[pd.Series, List]:
    """
      - predict_result: path to predictions file (JSON array; strict like upstream)
      - returns hallucination labels and list of all redundancies
    """

    # Strict JSON array load (parity)
    records = json.load(open(predict_result, "r", encoding="utf-8"))
    load_config_and_cache_vectors("config_baseline")
    labels, allRedundancies = analyze_records(records, "multiwoz21")

    return labels, allRedundancies