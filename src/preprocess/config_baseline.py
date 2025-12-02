"""
Retrieved from convlab/nlg/

Configuration file for the Systematic Hallucination Analyzer
"""

import re
from typing import Dict, List, Pattern, Set, Tuple

# =========================
# Configuration
# =========================
SIMILARITY_THRESHOLD = 0.7

POLITE_NER_DATES: Set[str] = {"today", "a good day"}

# =========================
# Normalization & Tokenization
# =========================
INT2WORD = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten",
}

# =========================
# Simple Regex Patterns
# =========================
PHONE_RE: Pattern = re.compile(r"\b(?:\+?\d[\d\s\-]{6,}\d)\b")
TIME_COLON_RE: Pattern = re.compile(r"\b\d{1,2}[:.\s]\d{2}\b")
REF_LIKE_RE: Pattern = re.compile(r"\b[A-Z0-9]{6,10}\b")

MONEY_NEAR_RE: Pattern = re.compile(
    r"(?:pounds?|gbp|£|\bper\b|\bticket\b|\bprice\b|\bcost\b|\beach\b|\bpp\b)",
    re.I,
)

STREET_WORDS = "road|rd|street|st|lane|ln|avenue|ave|way|row|place|pl|close|cl|drive|dr|court|ct|hill|park|gardens|crescent|square"
EXTRA_ADDR_RE: Pattern = re.compile(rf"\b(?:\d+\s+)?[a-z]+(?:\s+[a-z]+)*\s+(?:{STREET_WORDS})\b")
ADDR_HINT_RE: Pattern = re.compile(rf"\b\d+\s+[a-z]+(?:\s+[a-z]+)*\s+(?:{STREET_WORDS})\b")

# =========================
# Ontology & Value Stops
# =========================
ONTOLOGY_STOPWORDS: Set[str] = {
    "yes", "no",
    "hotel", "restaurant", "attraction",
    "place", "location", "free",
    "centre", "center", "north", "south", "east", "west",
    "the", "and", "of", "in",
    "guesthouse", "guest", "house", "hotel",
    "town", "city",
}

WEEKDAYS: Set[str] = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}

CUISINE_WORDS: Set[str] = {
    "indian", "chinese", "british", "italian", "french", "japanese",
    "korean", "thai", "turkish", "spanish", "mediterranean", "vietnamese",
    "mexican", "portuguese", "greek", "malaysian", "lebanese", "catalan",
}

# =========================
# Synonyms & Intents
# =========================
SLOT_SYNONYMS: Dict[str, List[str]] = {
    "food": ["cuisine", "food", "what", "kind"],
    "area": ["area", "part", "town", "neighborhood", "which"],
    "name": ["name", "what", "restaurant", "call", "spell"],
    "book time": ["time", "what"],
    "book day": ["day", "what", "date"],
    "leave at": ["leave", "when", "pickup"],
    "arrive by": ["arrive", "arrival"],
    "destination": ["where", "to"],
    "departure": ["leave", "from"],
}

REQUEST_PHRASES_BY_SLOT: Dict[str, List[List[Dict[str, str]]]] = {
    "area": [
        [{"LOWER": "area"}],
        [{"LOWER": "part"}, {"LOWER": "of"}, {"LOWER": "town"}],
        [{"LOWER": "neighborhood"}],
    ],
    "food": [
        [{"LOWER": "cuisine"}],
        [{"LOWER": "type"}, {"LOWER": "of"}, {"LOWER": "food"}],
    ],
    "book time": [
        [{"LOWER": "what"}, {"LOWER": "time"}],
        [{"LOWER": "different"}, {"LOWER": "time"}],
    ],
    "book day": [
        [{"LOWER": "what"}, {"LOWER": "day"}],
        [{"LOWER": "different"}, {"LOWER": "day"}],
    ],
    "book stay": [
        [{"LOWER": "how"}, {"LOWER": "many"}, {"LOWER": "nights"}],
        [{"LOWER": "how"}, {"LOWER": "long"}],
        [{"LOWER": "shorter"}, {"LOWER": "stay"}],
    ],
    "leave at": [
        [{"LOWER": "what"}, {"LOWER": "time"}],
        [{"LOWER": "leave"}],
        [{"LOWER": "depart"}],
    ],
    "arrive by": [
        [{"LOWER": "what"}, {"LOWER": "time"}],
        [{"LOWER": "arrive"}],
    ],
    "destination": [
        [{"LOWER": "destination"}],
        [{"LOWER": "where"}, {"LOWER": "to"}],
    ],
    "departure": [
        [{"LOWER": "departure"}],
        [{"LOWER": "from"}],
    ],
    "price range": [
        [{"LOWER": "price"}, {"LOWER": "range"}],
    ],
    "stars": [
        [{"LOWER": "how"}, {"LOWER": "many"}, {"LOWER": "stars"}],
        [{"LOWER": "star"}, {"LOWER": "rating"}],
    ],
}

INTENT2PHRASES_LOCAL: Dict[str, List[str]] = {
    "reqmore": ["anything else", "something else", "anything more", "what else", "would you like more information", "would you like more info"],
    "bye": ["goodbye", "bye", "see you", "have a nice day", "have a great day", "have a great trip"],
    "greet": ["hello", "hi"],
    "deny": ["no", "nope", "not really"],
    "book": ["i will book", "we will book", "i am book", "booked successfully", "booking confirmed", "reservation confirmed", "reservation made", "booking successful"],
    "offerbook": ["would you like to book", "would you like me to book", "can i book", "may i book", "shall i book", "would you like a ticket", "can i get you a ticket"],
}

# =========================
# Systematic Fix Knowledge
# =========================
GLOBAL_KEYWORD_SLOTS: Dict[str, Tuple[str, str]] = {
    "postcode": ("general", "postcode"),
    "entrance": ("attraction", "entrance fee"),
    "fee": ("attraction", "entrance fee"),
    "reference": ("general", "ref"),
    "trainid": ("train", "train id"),
}

SPACY_LABEL_TO_SLOTS: Dict[str, Set[str]] = {
    "ORG": {"name"},
    "PERSON": {"name"},
    "LOC": {"area", "destination", "departure", "name"},
    "GPE": {"area", "destination", "departure", "name"},
    "TIME": {"book time", "leave at", "arrive by", "duration"},
    "DATE": {"book day", "day"},
    "MONEY": {"price range", "price", "entrance fee"},
    "CARDINAL": {"book people", "choice", "stars"},
}

AMBIGUOUS_TIME_SLOTS: Set[str] = {"book time", "leave at", "arrive by"}

CHOICE_NOUNS_LEMMA: Set[str] = {
    "restaurant", "location", "branch",
    "hotel", "guesthouse", "guest", "house",
    "train",
    "option", "choice",
}

DEPENDENCY_SLOT_RULES: Dict[str, Dict[str, str]] = {
    "hotel-stars": {"lemma": "star", "dep": "nummod"},
    "restaurant-stars": {"lemma": "star", "dep": "nummod"},
}

MATCHER_SLOT_TRIGGERS: Dict[str, List[List[Dict[str, str]]]] = {
    "taxi-type": [
        [{"LOWER": "car"}, {"LOWER": "is"}],
        [{"LOWER": "vehicle"}, {"LOWER": "is"}],
    ],
    "train-leaveat": [
        [{"LOWER": "leaves"}, {"LOWER": "at"}],
        [{"LOWER": "departure"}, {"OP": "*"}, {"LOWER": "time"}],
    ],
    "train-arriveby": [
        [{"LOWER": "arrives"}, {"LOWER": "at"}],
        [{"LOWER": "arrival"}, {"OP": "*"}, {"LOWER": "time"}],
    ],
    "train-duration": [
        [{"LOWER": "travel"}, {"OP": "*"}, {"LOWER": "time"}],
        [{"LOWER": "journey"}, {"OP": "*"}, {"LOWER": "time"}],
    ],
    "restaurant-pricerange": [
        [{"LOWER": "price"}, {"OP": "*"}, {"LOWER": "range"}],
    ],
    "hotel-pricerange": [
        [{"LOWER": "price"}, {"OP": "*"}, {"LOWER": "range"}],
    ],
    "attraction-entrancefee": [
        [{"LOWER": "entrance"}, {"OP": "*"}, {"LOWER": "fee"}],
        [{"LOWER": "free"}, {"OP": "*"}, {"LOWER": "entry"}],
    ],
}

SEMANTIC_SLOT_TRIGGERS: Dict[str, List[str]] = {
    "price": ["price", "cost", "fee", "ticket"],
    "pricerange": ["cheap", "expensive", "moderate", "price"],
    "food": ["serves", "cuisine", "food"],
    "type": ["guesthouse", "hostel", "hotel", "museum", "park", "college", "theater", "entertainment"],
}