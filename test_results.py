# ---------------------------------------------------------
# Test script (Member 6)
# Compares our program's output with the PDF answer key.
# ---------------------------------------------------------

import json
import os

from partB.phrase_boolean import run_phrase_boolean_queries

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


def main():
    test_positional_index()
    test_phrase_queries()

    print("\n==============================")
    print("Passed: " + str(passed) + "   Failed: " + str(failed))
    print("==============================")


if __name__ == "__main__":
    main()