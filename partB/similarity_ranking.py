#Member 5
import csv
import json
import math
import os


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
INDEX_PATH = os.path.join(OUTPUT_DIR, "positional_index.json")
NORMALIZED_PATH = os.path.join(OUTPUT_DIR, "normalized_tfidf.csv")  
DF_IDF_PATH = os.path.join(OUTPUT_DIR, "df_idf.csv")               

DOCUMENTS = ["d{}".format(i) for i in range(1, 11)]
N = len(DOCUMENTS)

PHRASE_1 = "online learning"
PHRASE_2 = "student assessment"

EXPECTED = {
    "Phrase: online learning": [("d1", 1.0000), ("d5", 0.4476), ("d2", 0.4226), ("d9", 0.1939)],
    "Phrase: student assessment": [("d3", 0.4226), ("d5", 0.3051), ("d8", 0.2731)],
    "online learning AND student assessment": [("d5", 0.5322)],
    "online learning AND NOT student assessment": [("d1", 1.0000), ("d2", 0.4226), ("d9", 0.1939)],
}


def load_index(path=INDEX_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_idf(index):
    """IDF = log10(N / DF), DF = number of documents containing the term."""
    idf = {}
    for term, postings in index.items():
        df = len(postings)
        idf[term] = math.log10(N / df) if df > 0 else 0.0
    return idf


def compute_normalized(index, idf):
    """Same steps as Members 2 and 3, computed straight from the index."""
    tfidf = {}
    for term, postings in index.items():
        tfidf[term] = {}
        for doc in DOCUMENTS:
            tf = len(postings.get(doc, []))
            w = 1 + math.log10(tf) if tf > 0 else 0.0
            tfidf[term][doc] = w * idf[term]

    lengths = {
        doc: math.sqrt(sum(tfidf[t][doc] ** 2 for t in tfidf)) for doc in DOCUMENTS
    }
    normalized = {}
    for term in tfidf:
        normalized[term] = {
            doc: (tfidf[term][doc] / lengths[doc] if lengths[doc] > 0 else 0.0)
            for doc in DOCUMENTS
        }
    return normalized


def load_normalized_csv(path=NORMALIZED_PATH):
    """Load Member 3's normalized_tfidf.csv (if you prefer to use it)."""
    normalized = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            normalized[row["term"]] = {d: float(row[d]) for d in DOCUMENTS}
    return normalized


def load_idf_csv(path=DF_IDF_PATH):
    """Load Member 2's df_idf.csv (if you prefer to use it)."""
    idf = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            idf[row["term"]] = float(row["IDF"])
    return idf


def phrase_docs(index, phrase):
    tokens = phrase.lower().split()
    postings = [index.get(t, {}) for t in tokens]
    if any(not p for p in postings):
        return []
    docs = set(postings[0])
    for p in postings[1:]:
        docs &= set(p)
    matched = []
    for doc in docs:
        for start in postings[0][doc]:
            if all((start + i) in postings[i][doc] for i in range(1, len(tokens))):
                matched.append(doc)
                break
    return sorted(matched, key=lambda d: int(d[1:]))


def get_matching_docs(index):
    d1 = set(phrase_docs(index, PHRASE_1))
    d2 = set(phrase_docs(index, PHRASE_2))
    key = lambda d: int(d[1:])
    return {
        "Phrase: online learning": sorted(d1, key=key),
        "Phrase: student assessment": sorted(d2, key=key),
        "online learning AND student assessment": sorted(d1 & d2, key=key),
        "online learning AND NOT student assessment": sorted(d1 - d2, key=key),
    }


# ---------------------------------------------------------------- query vector
def query_terms_for(name):
    """AND uses all 4 words; AND NOT uses only the first phrase's words."""
    if name == "Phrase: online learning":
        return PHRASE_1.split()
    if name == "Phrase: student assessment":
        return PHRASE_2.split()
    if name == "online learning AND student assessment":
        return PHRASE_1.split() + PHRASE_2.split()
    if name == "online learning AND NOT student assessment":
        return PHRASE_1.split()
    raise ValueError("Unknown query: " + name)


def build_query_vector(terms, idf):
    """Returns a list of rows: term, tf(=1), weighted tf(=1), idf, tf-idf, normalized."""
    rows = []
    for t in terms:
        rows.append({"term": t, "tf": 1, "w_tf": 1.0, "idf": idf.get(t, 0.0),
                     "tfidf": 1.0 * idf.get(t, 0.0)})
    length = math.sqrt(sum(r["tfidf"] ** 2 for r in rows))
    for r in rows:
        r["normalized"] = r["tfidf"] / length if length > 0 else 0.0
    return rows, length


# ---------------------------------------------------------------- similarity and ranking
def similarity(query_rows, normalized, doc):
    return sum(r["normalized"] * normalized.get(r["term"], {}).get(doc, 0.0)
               for r in query_rows)


def rank_documents(query_rows, normalized, docs):
    scores = [(doc, similarity(query_rows, normalized, doc)) for doc in docs]
    return sorted(scores, key=lambda x: (-x[1], int(x[0][1:])))


# ---------------------------------------------------------------- printing
def print_query_vector(name, rows, length):
    print("\n========== QUERY VECTOR: " + name + " ==========")
    print("{:<13}{:>6}{:>8}{:>10}{:>10}{:>12}".format(
        "Term", "TF", "w-TF", "IDF", "TF-IDF", "Normalized"))
    for r in rows:
        print("{:<13}{:>6}{:>8.4f}{:>10.4f}{:>10.4f}{:>12.4f}".format(
            r["term"], r["tf"], r["w_tf"], r["idf"], r["tfidf"], r["normalized"]))
    print("Query vector length = {:.4f}".format(length))


def print_ranking(name, docs, ranking):
    print("\n========== RANKING: " + name + " ==========")
    print("Matching documents: " + (", ".join(docs) if docs else "none"))
    print("{:<6}{:<12}{}".format("Rank", "Document", "Similarity"))
    for i, (doc, score) in enumerate(ranking, 1):
        print("{:<6}{:<12}{:.4f}".format(i, doc, score))
    print("Ranking: " + " > ".join("{} ({:.4f})".format(d, s) for d, s in ranking))


def check_against_answer_key(all_rankings):
    print("\n" + "=" * 64)
    print("CHECK AGAINST ANSWER KEY")
    print("=" * 64)
    all_ok = True
    for name, expected in EXPECTED.items():
        got = [(d, round(s, 4)) for d, s in all_rankings[name]]
        ok = len(got) == len(expected) and all(
            g[0] == e[0] and abs(g[1] - e[1]) < 1e-4 for g, e in zip(got, expected))
        all_ok &= ok
        print("  [{}] {}".format("OK  " if ok else "FAIL", name))
        if not ok:
            print("        expected: {}".format(expected))
            print("        got     : {}".format(got))
    print("\n  " + ("All results match the answer key." if all_ok
                    else "Some results do NOT match - check the inputs."))
    return all_ok

def save_results(all_vectors, all_rankings):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save query vectors
    with open(os.path.join(OUTPUT_DIR, "query_vectors.csv"), "w",
              newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([
            "query", "term", "TF", "weighted_TF",
            "IDF", "TF-IDF", "normalized", "query_vector_length"
        ])

        for name, (rows, length) in all_vectors.items():
            for r in rows:
                w.writerow([
                    name,
                    r["term"],
                    r["tf"],
                    round(r["w_tf"], 4),
                    round(r["idf"], 4),round(r["tfidf"], 4),
                    round(r["normalized"], 4),round(length, 6)
                ])

    # Save similarity scores
    with open(os.path.join(OUTPUT_DIR, "similarity_scores.csv"), "w",
              newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["query", "document", "similarity"])

        for name, ranking in all_rankings.items():
            for doc, score in ranking:
                w.writerow([name, doc, round(score, 4)])

    # Save rankings
    with open(os.path.join(OUTPUT_DIR, "rankings.csv"), "w",
              newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["query", "rank", "document", "similarity"])

        for name, ranking in all_rankings.items():
            for i, (doc, score) in enumerate(ranking, 1):
                w.writerow([name, i, doc, round(score, 4)])

# ---------------------------------------------------------------- main
def run_all(verbose=True, save=False, use_csv=False):
    """Load data, build the 4 query vectors, score and rank, check, save."""
    index = load_index()
    if use_csv:                       
        idf = load_idf_csv()
        normalized = load_normalized_csv()
    else:                            
        idf = compute_idf(index)
        normalized = compute_normalized(index, idf)

    matching = get_matching_docs(index)

    all_vectors, all_rankings = {}, {}
    for name, docs in matching.items():
        rows, length = build_query_vector(query_terms_for(name), idf)
        ranking = rank_documents(rows, normalized, docs)
        all_vectors[name] = (rows, length)
        all_rankings[name] = ranking
        if verbose:
            print_query_vector(name, rows, length)
            print_ranking(name, docs, ranking)

    if verbose:
        check_against_answer_key(all_rankings)
    if save:
        save_results(all_vectors, all_rankings)
        if verbose:
            print("\nCSV files saved in: " + OUTPUT_DIR)
    return all_rankings

if __name__ == "__main__":
    run_all(save=True)