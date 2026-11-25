#Member 3
import csv
import math
import os
import re


# Formulas (log base 10, N = 10):

#   weighted TF  = 1 + log10(TF) for nonzero TF            
#   IDF          = log10(N / DF)

#   TF-IDF       = weighted TF x IDF = 1 + log10(TF) x log10(N / DF) 
#   length(d)    = sqrt( sum over terms of TF-IDF(t,d)^2 )
#   normalized   = TF-IDF(t,d) / length(d)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
OUTPUT_DIR = DEFAULT_OUTPUT_DIR 


def _doc_order(doc):
    return tuple(
        (0, int(p)) if p.isdigit() else (1, p.lower())
        for p in re.split(r"(\d+)", str(doc))
    )


def _documents_from(matrix):
    for row in matrix.values():
        return sorted(row, key=_doc_order)
    return []


def _paths(output_dir):
    output_dir = output_dir or DEFAULT_OUTPUT_DIR
    return (
        output_dir,
        os.path.join(output_dir, "weighted_tf.csv"),
        os.path.join(output_dir, "df_idf.csv"),
    )


# Load weighted TF data (documents are taken from the CSV header)
def load_weighted_tf(path=None, output_dir=None):
    if path is None:
        path = _paths(output_dir)[1]

    weighted_tf = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        documents = [c for c in (reader.fieldnames or []) if c != "term"]
        for row in reader:
            term = row["term"]
            weighted_tf[term] = {doc: float(row[doc] or 0) for doc in documents}
    return weighted_tf


# Load IDF data
def load_idf(path=None, output_dir=None):
    if path is None:
        path = _paths(output_dir)[2]

    idf = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idf[row["term"]] = float(row["IDF"])
    return idf


#   TF-IDF = weighted TF x IDF
def compute_tfidf(weighted_tf, idf, documents=None):
    documents = documents or _documents_from(weighted_tf)

    tfidf = {}
    for term in weighted_tf:
        tfidf[term] = {}
        for doc in documents:
            tfidf[term][doc] = weighted_tf[term][doc] * idf[term]
    return tfidf


#   length(d) = sqrt( sum over terms of TF-IDF(t,d)^2 )
def compute_doc_lengths(tfidf, documents=None):
    documents = documents or _documents_from(tfidf)

    lengths = {}
    for doc in documents:
        sum_of_squares = sum(tfidf[term][doc] ** 2 for term in tfidf)
        lengths[doc] = math.sqrt(sum_of_squares)
    return lengths


#   normalized = TF-IDF(t,d) / length(d)
def compute_normalized(tfidf, lengths, documents=None):
    documents = documents or _documents_from(tfidf)

    normalized = {}
    for term in tfidf:
        normalized[term] = {}
        for doc in documents:
            if lengths[doc] > 0:
                normalized[term][doc] = tfidf[term][doc] / lengths[doc]
            else:
                normalized[term][doc] = 0.0
    return normalized


def fmt(x):
    return "0" if x == 0 else "{:.4f}".format(x)


# Print Matrix
def print_matrix(title, matrix, documents=None):
    documents = documents or _documents_from(matrix)

    print("\n========== " + title + " ==========")
    print("{:<13}".format("Term") + "".join("{:>8}".format(doc) for doc in documents))

    for term in sorted(matrix):
        row = "{:<13}".format(term)
        for doc in documents:
            row += "{:>8}".format(fmt(matrix[term][doc]))
        print(row)


# Print Document Lengths
def print_lengths(lengths, documents=None):
    documents = documents or sorted(lengths, key=_doc_order)

    print("\n========== DOCUMENT VECTOR LENGTHS ==========")
    print("{:<12}{}".format("Document", "Vector Length"))
    for doc in documents:
        print("{:<12}{:.4f}".format(doc, lengths[doc]))


# Save Matrix to CSV
def save_matrix(filename, matrix, documents=None, output_dir=None):
    documents = documents or _documents_from(matrix)
    output_dir = output_dir or DEFAULT_OUTPUT_DIR

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["term"] + documents)
        for term in sorted(matrix):
            writer.writerow([term] + [round(matrix[term][doc], 4) for doc in documents])


# Save Document Lengths to CSV
def save_lengths(filename, lengths, documents=None, output_dir=None):
    documents = documents or sorted(lengths, key=_doc_order)
    output_dir = output_dir or DEFAULT_OUTPUT_DIR

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["document", "vector_length"])
        for doc in documents:
            writer.writerow([doc, round(lengths[doc], 4)])


def run_all(output_dir=None, verbose=True, save=False):
    output_dir = output_dir or DEFAULT_OUTPUT_DIR

    weighted_tf = load_weighted_tf(output_dir=output_dir)
    idf = load_idf(output_dir=output_dir)

    documents = _documents_from(weighted_tf)

    tfidf = compute_tfidf(weighted_tf, idf, documents)
    lengths = compute_doc_lengths(tfidf, documents)
    normalized = compute_normalized(tfidf, lengths, documents)

    if verbose:
        print_matrix("TF-IDF MATRIX", tfidf, documents)
        print_lengths(lengths, documents)
        print_matrix("NORMALIZED TF-IDF MATRIX", normalized, documents)

    if save:
        save_matrix("tf-idf.csv", tfidf, documents, output_dir)
        save_lengths("doc_lengths.csv", lengths, documents, output_dir)
        save_matrix("normalized_tfidf.csv", normalized, documents, output_dir)

        if verbose:
            print("\nCSV files saved in: " + output_dir)

    return tfidf, lengths, normalized


if __name__ == "__main__":
    run_all()