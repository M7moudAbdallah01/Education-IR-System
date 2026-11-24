import json
import math
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_PATH = BASE_DIR / "output" / "positional_index.json"

with open(INDEX_PATH, "r", encoding="utf-8") as file:
    positional_index = json.load(file)

documents = [f"d{i}" for i in range(1, 11)]
N = len(documents)

raw_tf = {}
weighted_tf = {}
df_table = {}

for term, doc_positions in positional_index.items():

    raw_tf[term] = {}
    weighted_tf[term] = {}

    for doc in documents:

        positions = doc_positions.get(doc, [])
        tf = len(positions)
        raw_tf[term][doc] = tf
        if tf > 0:
            weighted_tf[term][doc] = 1 + math.log10(tf)
        else:
            weighted_tf[term][doc] = 0

    
    df = sum(
        1 for doc in documents
        if raw_tf[term][doc] > 0
    )

    idf = math.log10(N / df) if df > 0 else 0

    df_table[term] = {
        "DF": df,
        "IDF": idf
    }


print("\n========== RAW TF MATRIX ==========")

for term, values in raw_tf.items():
    print(term, values)

print("\n========== WEIGHTED TF MATRIX ==========")

for term, values in weighted_tf.items():
    formatted = {
        doc: round(value, 4)
        for doc, value in values.items()
    }
    print(term, formatted)

print("\n========== DF AND IDF ==========")

for term, values in df_table.items():
    print(
        term,
        "DF =", values["DF"],
        "IDF =", round(values["IDF"], 4)
    )

#
#import csv
#
# =====================================
# 7. Export Results to CSV
# =====================================
#
#OUTPUT_DIR = BASE_DIR / "output"
#OUTPUT_DIR.mkdir(exist_ok=True)
#
#documents_columns = ["term"] + documents
#
#
# def save_matrix(filename, matrix):
#
#    with open(
#        OUTPUT_DIR / filename,
#        "w",
#        newline="",
#        encoding="utf-8-sig"
#    ) as file:
#
#        writer = csv.writer(file)
#
#        writer.writerow(documents_columns)
#
#        for term, values in matrix.items():
#
#            row = [term]
#
#            for doc in documents:
#                row.append(round(values[doc], 4))
#
#            writer.writerow(row)
#
#
#save_matrix("raw_tf.csv", raw_tf)
#
#
#save_matrix("weighted_tf.csv", weighted_tf)
#
#
## Save DF and IDF
#with open(
#    OUTPUT_DIR / "df_idf.csv",
#    "w",
#    newline="",
#    encoding="utf-8-sig"
#) as file:
#
#    writer = csv.writer(file)
#
#    writer.writerow(["term", "DF", "IDF"])
#
#    for term, values in df_table.items():
#
#        writer.writerow([
#            term,
#            values["DF"],
#            round(values["IDF"], 4)
#        ])
#
#
#print("\nCSV files exported successfully!")