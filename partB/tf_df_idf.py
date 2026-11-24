import json
import math
import csv
import re
from pathlib import Path

# =====================================
# 1. Project Paths and Settings
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "Dataset"
INDEX_PATH = BASE_DIR / "output" / "positional_index.json"

OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Choose: "auto", "files", or "index"
DATA_SOURCE = "auto"


# =====================================
# 2. Helper Functions
# =====================================

def natural_sort_key(value):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    ]


def get_dataset_files():

    if not DATASET_DIR.exists():
        return []

    return sorted(
        DATASET_DIR.glob("*.txt"),
        key=lambda file: natural_sort_key(file.stem)
    )


def build_index_from_files(files):
    """Build an in-memory positional index from TXT files."""

    index = {}

    for file_path in files:

        doc_name = file_path.stem

        text = file_path.read_text(encoding="utf-8")

        # Tokenize text into lowercase terms
        terms = re.findall(r"\b\w+\b", text.lower())

        for position, term in enumerate(terms, start=1):

            if term not in index:
                index[term] = {}

            if doc_name not in index[term]:
                index[term][doc_name] = []

            index[term][doc_name].append(position)

    return index


def load_positional_index():

    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Positional Index not found: {INDEX_PATH}"
        )

    with open(INDEX_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# =====================================
# 3. Select Data Source
# =====================================

dataset_files = get_dataset_files()

if DATA_SOURCE == "files":

    if not dataset_files:
        raise FileNotFoundError(
            f"No TXT files found in: {DATASET_DIR}"
        )

    positional_index = build_index_from_files(dataset_files)

    documents = [file.stem for file in dataset_files]


elif DATA_SOURCE == "index":

    positional_index = load_positional_index()

    indexed_documents = {
        doc
        for doc_positions in positional_index.values()
        for doc in doc_positions
    }

    if dataset_files:
        documents = [file.stem for file in dataset_files]

    else:
        documents = sorted(
            indexed_documents,
            key=natural_sort_key
        )


elif DATA_SOURCE == "auto":

    if dataset_files:

        positional_index = build_index_from_files(dataset_files)

        documents = [file.stem for file in dataset_files]

    else:

        positional_index = load_positional_index()

        indexed_documents = {
            doc
            for doc_positions in positional_index.values()
            for doc in doc_positions
        }

        documents = sorted(
            indexed_documents,
            key=natural_sort_key
        )


else:
    raise ValueError(
        "DATA_SOURCE must be 'auto', 'files', or 'index'."
    )


if not documents:
    raise ValueError("No documents were found!")


# =====================================
# 4. Sort Terms Alphabetically
# =====================================

positional_index = dict(
    sorted(positional_index.items(), key=lambda item: item[0].lower())
)

N = len(documents)


# =====================================
# 5. Calculate Raw TF, Weighted TF,
#    DF and IDF
# =====================================

raw_tf = {}
weighted_tf = {}
df_table = {}

for term, doc_positions in positional_index.items():

    raw_tf[term] = {}
    weighted_tf[term] = {}

    for doc in documents:

        positions = doc_positions.get(doc, [])

        tf = len(positions)

        # Raw TF
        raw_tf[term][doc] = tf

        # Weighted TF = 1 + log10(TF)
        if tf > 0:
            weighted_tf[term][doc] = 1 + math.log10(tf)

        else:
            weighted_tf[term][doc] = 0

    # Document Frequency
    df = sum(
        1 for doc in documents
        if raw_tf[term][doc] > 0
    )

    # Inverse Document Frequency
    idf = math.log10(N / df) if df > 0 else 0

    df_table[term] = {
        "DF": df,
        "IDF": idf
    }


# =====================================
# 6. Print RAW TF Matrix
# =====================================

print("\n========== RAW TF MATRIX ==========")

for term, values in raw_tf.items():

    print(term, values)


# =====================================
# 7. Print WEIGHTED TF Matrix
# =====================================

print("\n========== WEIGHTED TF MATRIX ==========")

for term, values in weighted_tf.items():

    formatted_values = {
        doc: round(value, 4)
        for doc, value in values.items()
    }

    print(term, formatted_values)


# =====================================
# 8. Print DF and IDF
# =====================================

print("\n========== DF AND IDF ==========")

for term, values in df_table.items():

    print(
        f"{term} DF = {values['DF']} "
        f"IDF = {values['IDF']:.4f}"
    )


# =====================================
# 9. Export Results to CSV
# =====================================

## def save_matrix(filename, matrix):
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
#        writer.writerow(["Term"] + documents)
#
#        for term, values in matrix.items():
#
#            row = [term]
#
#            for doc in documents:
#
#                row.append(f"{values[doc]:.4f}")
#
#            writer.writerow(row)
#
#
## Export Raw TF
#save_matrix("raw_tf.csv", raw_tf)
#
## Export Weighted TF
#save_matrix("weighted_tf.csv", weighted_tf)
#
#
## Export DF and IDF
#with open(
#    OUTPUT_DIR / "df_idf.csv",
#    "w",
#    newline="",
#    encoding="utf-8-sig"
#) as file:
#
#    writer = csv.writer(file)
#
#    writer.writerow(["Term", "DF", "IDF"])
#
#    for term, values in df_table.items():
#
#        writer.writerow([
#            term,
#            values["DF"],
#            f"{values['IDF']:.4f}"
#        ])
#
#
## =====================================
## 10. Final Message
## =====================================
#
#print("\nAll calculations completed successfully!")
#print("CSV files exported successfully!")
#