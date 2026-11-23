# Member 4
import json
import os
import re
import sys
from typing import Dict, Iterable, List, Set, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INDEX_PATH = os.path.join(PROJECT_ROOT, "output", "positional_index.json")

Span = Tuple[int, int]             
PhraseMatches = Dict[str, List[Span]]  

PHRASE_1 = "online learning"
PHRASE_2 = "student assessment"

EXPECTED = {
    "phrase_1": ["d1", "d2", "d5", "d9"],
    "phrase_2": ["d3", "d5", "d8"],
    "and": ["d5"],
    "and_not": ["d1", "d2", "d9"],
}

ANSWER_KEYS = {
    ("online learning", "student assessment"): EXPECTED,
}

_EDGE_PUNC = re.compile(r"^\W+|\W+$", re.UNICODE)

def normalize_token(token: str) -> str:
    return _EDGE_PUNC.sub("", token.lower())

def tokenize_phrase(phrase: str) -> List[str]:
    return [t for t in (normalize_token(w) for w in phrase.split()) if t]

def normalize_index(index: dict) -> dict:
    clean: Dict[str, Dict[str, List[int]]] = {}
    for term, postings in index.items() :
        key = normalize_token(term)
        if not key:
            continue
        bucket = clean.setdefault(key, {})
        for doc, positions in postings.items():
            bucket.setdefault(doc, []).extend(positions)
    for postings in clean.values():
        for doc in postings:
            postings[doc] = sorted(set(postings[doc]))
    return clean                

def doc_sort_key(doc_id: str):
    """Natural order so that d2 comes before d10 (plain string sort would not)."""
    digits = "".join(ch for ch in doc_id if ch.isdigit())
    return (0, int(digits), doc_id) if digits else (1, 0, doc_id)


def sort_docs(docs: Iterable[str]) -> List[str]:
    """Sort document ids naturally: d1, d2, ..., d10."""
    return sorted(docs, key=doc_sort_key)


def load_index(path: str = DEFAULT_INDEX_PATH) -> dict:
    """Load the positional index JSON file saved by Part A."""
    with open(path, "r", encoding="utf-8") as f:
        return normalize_index(json.load(f))

def phrase_match(index: dict, phrase: str) -> PhraseMatches:
    """
    Find the documents where the words of `phrase` appear next to each other,
    in order (positions p, p+1, p+2, ...).

    Returns {doc_id: [(start_pos, end_pos), ...]} containing only matching
    documents, in natural document order.
    """
    tokens = tokenize_phrase(phrase)
    if not tokens:
        raise ValueError("Phrase must contain at least one word.")

    postings = [index.get(t, {}) for t in tokens]
    if any(not p for p in postings):          
        return {}

    common_docs = set(postings[0])
    for p in postings[1:]:
        common_docs &= set(p)

    last = len(tokens) - 1
    matches: PhraseMatches = {}
    for doc in sort_docs(common_docs):
        later_positions = [set(p[doc]) for p in postings[1:]]   
        spans = [
            (start, start + last)
            for start in sorted(postings[0][doc])
            if all((start + i) in later_positions[i - 1] for i in range(1, last + 1))
        ]
        if spans:
            matches[doc] = spans
    return matches

def boolean_query(set_a: Iterable[str], set_b: Iterable[str], operator: str) -> List[str]:
    """
    Evaluate a Boolean operation between two document collections.
        AND      -> documents in both sets
        AND NOT  -> documents in the first set but not in the second
    """
    op = " ".join(operator.upper().replace("_", " ").split())
    docs_a: Set[str] = set(set_a)
    docs_b: Set[str] = set(set_b)
    if op == "AND":
        return sort_docs(docs_a & docs_b)
    if op == "AND NOT":
        return sort_docs(docs_a - docs_b)
    raise ValueError(
        "Unsupported Boolean operator: {!r} (use 'AND' or 'AND NOT')".format(operator)
    )

def format_phrase_matches(matches: PhraseMatches) -> List[str]:
    """One printable line per matching document: starts and token ranges."""
    lines = []
    for doc in sort_docs(matches):
        spans = matches[doc]
        starts = ", ".join(str(s) for s, _ in spans)
        ranges = ", ".join("{}-{}".format(s, e) if e != s else str(s) for s, e in spans)
        lines.append("  {}: starts {}  (token ranges: {})".format(doc, starts, ranges))
    return lines


def print_phrase_result(phrase: str, matches: PhraseMatches) -> None:
    print('\nPhrase query: "{}"'.format(phrase))
    print("-" * 64)
    if matches:
        print("\n".join(format_phrase_matches(matches)))
        print("  Matching documents: {}".format(", ".join(sort_docs(matches))))
    else:
        print("  No matching documents.")


def print_boolean_result(title: str, docs: List[str]) -> None:
    print("\n" + title)
    print("-" * 64)
    print("  Matching documents: {}".format(", ".join(docs) if docs else "none"))

def run_phrase_boolean_queries(index_path: str = DEFAULT_INDEX_PATH,
                               verbose: bool = True,
                               phrase_1:str = PHRASE_1,
                               phrase_2: str = PHRASE_2) -> dict:
    """
    Run the 4 project queries. Returns:
        phrase_1 / phrase_2 : sorted matching doc lists
        and / and_not       : Boolean result doc lists
        matches_1 / matches_2 : full {doc: [(start, end), ...]} detail
    """
    index = load_index(index_path)

    m1 = phrase_match(index, phrase_1)
    m2 = phrase_match(index, phrase_2)
    docs1, docs2 = set(m1), set(m2)

    and_docs = boolean_query(docs1, docs2, "AND")
    and_not_docs = boolean_query(docs1, docs2, "AND NOT")

    if verbose:
        print("=" * 64)
        print("PART B - PHRASE QUERIES AND BOOLEAN LOGIC")
        print("=" * 64)
        print_phrase_result(phrase_1, m1)
        print_phrase_result(phrase_2, m2)
        print_boolean_result('"{}" AND "{}"'.format(phrase_1, phrase_2), and_docs)
        print_boolean_result('"{}" AND NOT "{}"'.format(phrase_1, phrase_2), and_not_docs)

    return {
        "phrase_1": sort_docs(docs1),
        "phrase_2": sort_docs(docs2),
        "and": and_docs,
        "and_not": and_not_docs,
        "matches_1": m1,
        "matches_2": m2,
        "phrases": (phrase_1, phrase_2),
    }


def check_against_answer_key(results: dict) -> bool:
    """Compare results with the PDF answer key and print OK / FAIL per query."""
    print("\n" + "=" * 64)
    print("CHECK AGAINST ANSWER KEY")
    print("=" * 64)
    pair = tuple(" ".join(tokenize_phrase(p)) for p in results.get("phrases", (PHRASE_1, PHRASE_2)))
    expected_all = ANSWER_KEYS.get(pair)
    if expected_all is None:
        print("  No answer key stored for these phrases - check skipped.")
        print("  (Add an entry to ANSWER_KEYS if expected results are available.)")
        return True
    all_ok = True
    for key, expected in expected_all.items():
        ok = results[key] == expected
        all_ok &= ok
        print("  [{}] {:<8} expected {}  got {}".format(
            "OK  " if ok else "FAIL", key, expected, results[key]))
    print("\n  " + ("All results match the answer key." if all_ok
                    else "Some results do NOT match - check the index."))
    return all_ok


if __name__ == "__main__":
    if len(sys.argv) == 3:
        check_against_answer_key(run_phrase_boolean_queries(phrase_1=sys.argv[1], phrase_2=sys.argv[2]))   
    else:
        check_against_answer_key(run_phrase_boolean_queries())