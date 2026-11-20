# ---------------------------------------------------------
# Test script (Member 6)
# Compares our program's output with the PDF answer key.
# ---------------------------------------------------------

import json
import math
import os

from partB.phrase_boolean import run_phrase_boolean_queries
from partB.similarity_ranking import run_all as run_similarity_ranking
from partB import tf_df_idf
from partB import tfidf_normalization

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
INDEX_FILE = os.path.join(PROJECT_ROOT, "output", "positional_index.json")

TOLERANCE = 0.0001

passed = 0
failed = 0

# helpful functions
def report(name, ok, details=""):
    global passed, failed
    if ok:
        passed += 1
        print("PASS  " + name)
    else:
        failed += 1
        print("FAIL  " + name + "   " + details)


def check_number(name, actual, expected):
    ok = abs(actual - expected) <= TOLERANCE
    report(name, ok, "got " + str(actual) + ", expected " + str(expected))


def check_list(name, actual, expected):
    report(name, actual == expected, "got " + str(actual) + ", expected " + str(expected))



# Answer key: Positional Index (Education version)
EXPECTED_INDEX = {
    "academic":    {"d7": [1], "d10": [1]},
    "analysis":    {"d8": [10], "d10": [3]},
    "and":         {"d4": [6], "d7": [6], "d8": [8]},
    "assessment":  {"d3": [2], "d5": [4], "d8": [2, 4], "d10": [9]},
    "classroom":   {"d4": [1], "d6": [1], "d10": [7]},
    "courses":     {"d5": [6]},
    "data":        {"d8": [9]},
    "digital":     {"d9": [8]},
    "education":   {"d9": [9]},
    "engagement":  {"d2": [5], "d4": [5], "d10": [6]},
    "improve":     {"d3": [4], "d8": [6]},
    "learning":    {"d1": [2], "d2": [2], "d3": [5], "d4": [7], "d5": [2, 9],
                    "d7": [7], "d9": [2, 6], "d10": [8]},
    "measures":    {"d10": [4]},
    "methods":     {"d3": [3], "d6": [3], "d8": [5]},
    "online":      {"d1": [1], "d2": [1], "d5": [1, 5, 8], "d9": [1]},
    "outcomes":    {"d4": [8], "d7": [8], "d9": [7], "d10": [10]},
    "performance": {"d7": [2], "d10": [2]},
    "platforms":   {"d9": [3]},
    "progress":    {"d7": [5]},
    "reflects":    {"d7": [3]},
    "research":    {"d8": [7]},
    "student":     {"d2": [4], "d3": [1], "d4": [4], "d5": [3], "d7": [4],
                    "d8": [1, 3], "d9": [5], "d10": [5]},
    "support":     {"d5": [7], "d6": [4], "d9": [4]},
    "supports":    {"d2": [3], "d4": [3]},
    "systems":     {"d9": [10]},
    "teacher":     {"d6": [5]},
    "teaching":    {"d4": [2], "d6": [2]},
    "training":    {"d6": [6]},
}


# ---------------------------------------------------------
# Test 1: Positional Index (Part A)
# ---------------------------------------------------------
def test_positional_index():
    print("\n--- Positional Index ---")

    if not os.path.exists(INDEX_FILE):
        report("index file exists", False, "not found: " + INDEX_FILE)
        return

    with open(INDEX_FILE, encoding="utf-8") as f:
        actual_index = json.load(f)

    # same number of terms?
    check_number("number of terms", len(actual_index), len(EXPECTED_INDEX))

    # every expected term must be present with the exact same postings
    for term in EXPECTED_INDEX:
        if term not in actual_index:
            report("term '" + term + "'", False, "missing from our index")
        else:
            report("term '" + term + "'", actual_index[term] == EXPECTED_INDEX[term],
                   "got " + str(actual_index[term]) + ", expected " + str(EXPECTED_INDEX[term]))

    # no extra terms that are not in the answer key
    for term in actual_index:
        if term not in EXPECTED_INDEX:
            report("extra term '" + term + "'", False, "not in the answer key")


# ---------------------------------------------------------
# Test 2: Phrase and Boolean queries (Member 4)
# ---------------------------------------------------------
def test_phrase_queries():
    print("\n--- Phrase and Boolean Queries ---")

    results = run_phrase_boolean_queries(INDEX_FILE, verbose=False)

    # matching documents
    check_list("phrase 'online learning'", results["phrase_1"], ["d1", "d2", "d5", "d9"])
    check_list("phrase 'student assessment'", results["phrase_2"], ["d3", "d5", "d8"])
    check_list("AND", results["and"], ["d5"])
    check_list("AND NOT", results["and_not"], ["d1", "d2", "d9"])

    # start/end positions of every match (from the PDF)
    expected_1 = {"d1": [(1, 2)], "d2": [(1, 2)], "d5": [(1, 2), (8, 9)], "d9": [(1, 2)]}
    expected_2 = {"d3": [(1, 2)], "d5": [(3, 4)], "d8": [(1, 2), (3, 4)]}
    report("positions 'online learning'", results["matches_1"] == expected_1,
           "got " + str(results["matches_1"]))
    report("positions 'student assessment'", results["matches_2"] == expected_2,
           "got " + str(results["matches_2"]))

# ---------------------------------------------------------
# Test 3: Raw TF, Weighted TF, DF & IDF (Member 2)
# ---------------------------------------------------------

def test_tf_df_idf():
    print("\n--- TF, Weighted TF, DF & IDF ---")

    check_number("raw_tf terms count", len(tf_df_idf.raw_tf), len(EXPECTED_INDEX))
    check_number("weighted_tf terms count", len(tf_df_idf.weighted_tf), len(EXPECTED_INDEX))
    check_number("df_table terms count", len(tf_df_idf.df_table), len(EXPECTED_INDEX))

    N = 10
    for term, postings in EXPECTED_INDEX.items():
        if term not in tf_df_idf.raw_tf or term not in tf_df_idf.df_table:
            report("tf/idf term '" + term + "'", False, "missing from Member 2 output")
            continue

        for i in range(1, 11):
            doc = f"d{i}"
            exp_tf = len(postings.get(doc, []))
            exp_wtf = (1 + math.log10(exp_tf)) if exp_tf > 0 else 0.0

            act_tf = tf_df_idf.raw_tf[term].get(doc, -1)
            act_wtf = tf_df_idf.weighted_tf[term].get(doc, -1)

            if act_tf != exp_tf:
                report(f"raw_tf '{term}' in {doc}", False, f"got {act_tf}, expected {exp_tf}")
            if abs(act_wtf - exp_wtf) > TOLERANCE:
                report(f"weighted_tf '{term}' in {doc}", False, f"got {act_wtf}, expected {exp_wtf}")

        exp_df = len(postings)
        exp_idf = math.log10(N / exp_df) if exp_df > 0 else 0.0

        act_df = tf_df_idf.df_table[term]["DF"]
        act_idf = tf_df_idf.df_table[term]["IDF"]

        check_number(f"DF '{term}'", act_df, exp_df)
        check_number(f"IDF '{term}'", act_idf, exp_idf)

# ---------------------------------------------------------
# Test 4: TF-IDF, Document Lengths & Normalization (Member 3)
# ---------------------------------------------------------

def test_tfidf_normalization():
    print("\n--- TF-IDF & Normalization ---")

    idf_dict = {term: vals["IDF"] for term, vals in tf_df_idf.df_table.items()}
    act_tfidf = tfidf_normalization.compute_tfidf(tf_df_idf.weighted_tf, idf_dict)
    act_lengths = tfidf_normalization.compute_doc_lengths(act_tfidf)
    act_norm = tfidf_normalization.compute_normalized(act_tfidf, act_lengths)

    N = 10
    docs = [f"d{i}" for i in range(1, 11)]
    exp_tfidf = {}

    for term, postings in EXPECTED_INDEX.items():
        exp_tfidf[term] = {}
        df = len(postings)
        idf = math.log10(N / df) if df > 0 else 0.0
        for doc in docs:
            tf = len(postings.get(doc, []))
            wtf = (1 + math.log10(tf)) if tf > 0 else 0.0
            exp_tfidf[term][doc] = wtf * idf

    exp_lengths = {
        doc: math.sqrt(sum(exp_tfidf[t][doc] ** 2 for t in EXPECTED_INDEX))
        for doc in docs
    }

    for doc in docs:
        check_number(f"doc length '{doc}'", act_lengths.get(doc, -1), exp_lengths[doc])

    tfidf_errors = 0
    norm_errors = 0

    for term in EXPECTED_INDEX:
        for doc in docs:

            if abs(act_tfidf[term][doc] - exp_tfidf[term][doc]) > TOLERANCE:
                tfidf_errors += 1
                report(
                    f"tfidf '{term}' in {doc}",
                    False,
                    f"got {act_tfidf[term][doc]:.4f}, expected {exp_tfidf[term][doc]:.4f}",
                )

            exp_n = (exp_tfidf[term][doc] / exp_lengths[doc]) if exp_lengths[doc] > 0 else 0.0
            if abs(act_norm[term][doc] - exp_n) > TOLERANCE:
                norm_errors += 1
                report(
                    f"normalized '{term}' in {doc}",
                    False,
                    f"got {act_norm[term][doc]:.4f}, expected {exp_n:.4f}",
                )

    if tfidf_errors == 0:
        report("all TF-IDF matrix values (280 cells)", True)
    if norm_errors == 0:
        report("all Normalized TF-IDF matrix values (280 cells)", True)

# ---------------------------------------------------------
# Test 5: TF-IDF, Normalization & Ranking (Members 2, 3, 5)
# ---------------------------------------------------------
def test_similarity_ranking():
    print("\n--- TF-IDF & Similarity Ranking ---")

    all_rankings = run_similarity_ranking(verbose=False, save=False, use_csv=True)

    expected_rankings = {
        "Phrase: online learning": [("d1", 1.0000), ("d5", 0.4476), ("d2", 0.4226), ("d9", 0.1939)],
        "Phrase: student assessment": [("d3", 0.4226), ("d5", 0.3051), ("d8", 0.2731)],
        "online learning AND student assessment": [("d5", 0.5322)],
        "online learning AND NOT student assessment": [("d1", 1.0000), ("d2", 0.4226), ("d9", 0.1939)],
    }

    for query_name, expected_list in expected_rankings.items():
        if query_name not in all_rankings:
            report("query '" + query_name + "'", False, "missing from rankings")
            continue

        actual_list = all_rankings[query_name]

        actual_docs = [doc for doc, score in actual_list]
        expected_docs = [doc for doc, score in expected_list]
        check_list("ranking order '" + query_name + "'", actual_docs, expected_docs)

        for (act_doc, act_score), (exp_doc, exp_score) in zip(actual_list, expected_list):
            if act_doc == exp_doc:
                check_number(
                    "score '" + query_name + "' -> " + exp_doc,
                    act_score,
                    exp_score,
                )

def main():
    test_positional_index()
    test_phrase_queries()
    test_tf_df_idf()
    test_tfidf_normalization()
    test_similarity_ranking()

    print("\n==============================")
    print("Passed: " + str(passed) + "   Failed: " + str(failed))
    print("==============================")


if __name__ == "__main__":
    main()