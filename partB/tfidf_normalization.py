#Member 3
import csv
import math
import os

# Formulas (log base 10, N = 10):

#   weighted TF  = 1 + log10(TF) for nonzero TF            
#   IDF          = log10(N / DF)

#   TF-IDF       = weighted TF x IDF = 1 + log10(TF) x log10(N / DF) 
#   length(d)    = sqrt( sum over terms of TF-IDF(t,d)^2 )
#   normalized   = TF-IDF(t,d) / length(d)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

WEIGHTED_TF_PATH = os.path.join(OUTPUT_DIR, "weighted_tf.csv")

DF_IDF_PATH = os.path.join(OUTPUT_DIR, "df_idf.csv")

DOCUMENTS = ["d{}".format(i) for i in range(1, 11)]

# Load weighted TF data 
def load_weighted_tf(path=WEIGHTED_TF_PATH):
    weighted_tf = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            term = row["term"]
            weighted_tf[term] = {}
            for doc in DOCUMENTS:
                weighted_tf[term][doc] = float(row[doc])
    return weighted_tf

# Load IDF data 
def load_idf(path=DF_IDF_PATH):

    idf = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            term = row["term"]
            idf[term] = float(row["IDF"])

    return idf


#   TF-IDF       = weighted TF x IDF = 1 + log10(TF) x log10(N / DF) 
def compute_tfidf(weighted_tf, idf):

    tfidf = {}
    for term in weighted_tf:
        tfidf[term] = {}
        for doc in DOCUMENTS:
            tfidf[term][doc] = ( weighted_tf[term][doc] * idf[term] ) # TF-IDF = weighted TF x IDF
            
    return tfidf



#   length(d)    = sqrt( sum over terms of TF-IDF(t,d)^2 )
def compute_doc_lengths(tfidf):

    lengths = {}
    for doc in DOCUMENTS:
        sum_of_squares = sum(
            tfidf[term][doc] ** 2
            for term in tfidf
        )
        lengths[doc] = math.sqrt(sum_of_squares)

    return lengths


#   normalized   = TF-IDF(t,d) / length(d)
def compute_normalized(tfidf, lengths):

    normalized = {}
    for term in tfidf:
        normalized[term] = {}
        for doc in DOCUMENTS:
            if lengths[doc] > 0:
                normalized[term][doc] = (
                    tfidf[term][doc] / lengths[doc]
                )
            else:
                normalized[term][doc] = 0.0

    return normalized

def fmt(x):

    return "0" if x == 0 else "{:.4f}".format(x)

# Print Matrix

def print_matrix(title, matrix):

    print("\n========== " + title + " ==========")
    print("{:<13}".format("Term") + "".join("{:>8}".format(doc) for doc in DOCUMENTS))
    
    for term in sorted(matrix):
        
        row = "{:<13}".format(term)
        
        for doc in DOCUMENTS:
            row += "{:>8}".format(fmt(matrix[term][doc]))

        print(row)


# Print Document Lengths

def print_lengths(lengths):

    print("\n========== DOCUMENT VECTOR LENGTHS ==========")
    print("{:<12}{}".format("Document", "Vector Length"))

    for doc in DOCUMENTS:
        print("{:<12}{:.4f}".format(doc, lengths[doc]))


# Save Matrix to CSV

def save_matrix(filename, matrix):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:

        writer = csv.writer(f)
        writer.writerow(["term"] + DOCUMENTS)
        for term in sorted(matrix):
            writer.writerow(
                [term] +
                [round(matrix[term][doc], 4) for doc in DOCUMENTS]
            )


# Save Document Lengths to CSV

def save_lengths(filename, lengths):

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:

        writer = csv.writer(f)
        writer.writerow(["document", "vector_length"])
        for doc in DOCUMENTS:
            writer.writerow([
                doc,
                round(lengths[doc], 4)
            ])


# Run all steps: load weighted TF, load IDF, compute TF-IDF, compute document lengths, normalize TF-IDF, print and save results

def run_all(verbose=True, save=False):

    weighted_tf = load_weighted_tf()
    idf = load_idf()

    tfidf = compute_tfidf(weighted_tf, idf)
    lengths = compute_doc_lengths(tfidf)
    normalized = compute_normalized(tfidf, lengths)

    # Display results
    if verbose:

        print_matrix("TF-IDF MATRIX", tfidf)
        print_lengths(lengths)
        print_matrix("NORMALIZED TF-IDF MATRIX", normalized)

    # Save results
    if save:

        save_matrix("tf-idf.csv", tfidf)
        save_lengths("doc_lengths.csv", lengths)
        save_matrix("normalized_tfidf.csv", normalized)

        if verbose:
            print("\nCSV files saved in: " + OUTPUT_DIR)

if __name__ == "__main__":

    run_all()