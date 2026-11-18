import glob
import json
import os
import re
import sys

from pyspark.sql import SparkSession

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "dataSet")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "output", "positional_index.json")

def doc_number(doc_id):

    return int(re.sub(r"\D", "", doc_id))


def tokenize(record):
    
    doc_id, text = record
    for position, term in enumerate(text.lower().split(), start=1):
        yield (term, doc_id), position


# Spark pipeline

def build_positional_index(sc, file_paths):
    
    # (file path, full text) -> (doc_id, text)
    documents = (sc.wholeTextFiles(",".join(file_paths))
                   .map(lambda kv: (os.path.splitext(os.path.basename(kv[0]))[0], kv[1])))

    # ((term, doc_id), position)
    term_positions = documents.flatMap(tokenize)

    # ((term, doc_id), [sorted positions])
    positions_per_doc = term_positions.groupByKey().mapValues(sorted)

    # (term, [(doc_id, [positions]), ...])
    index = (positions_per_doc
             .map(lambda kv: (kv[0][0], (kv[0][1], kv[1])))
             .groupByKey()
             .mapValues(lambda postings: sorted(postings, key=lambda p: doc_number(p[0])))
             .sortByKey())

    return documents, index

def display_index(index_items):

    width = max(len(term) for term, _ in index_items) + 2
    print("=" * 72)
    print("POSITIONAL INDEX  (term -> document: token positions)")
    print("=" * 72)
    for term, postings in index_items:
        entry = "; ".join("{}: {}".format(doc, ", ".join(map(str, pos))) for doc, pos in postings)
        print(term.ljust(width) + entry)
    print("=" * 72)



def save_index(index_items, output_path):

    index_dict = {term: {doc: pos for doc, pos in postings} for term, postings in index_items}
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out:
        json.dump(index_dict, out, indent=2)


def main():
    file_paths = sorted(glob.glob(os.path.join(DATA_DIR, "d*.txt")),
                        key=lambda p: doc_number(os.path.basename(p)))
    if not file_paths:
        sys.exit("No document files (d*.txt) were found in: " + DATA_DIR)
    file_paths = [os.path.abspath(p) for p in file_paths]

    spark = (SparkSession.builder
             .appName("PositionalIndex")
             .master("local[*]")
             .getOrCreate())
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")

    documents, index = build_positional_index(sc, file_paths)
    index_items = index.collect()          

    display_index(index_items)
    print("Documents processed : {}".format(documents.count()))
    print("Distinct terms      : {}".format(len(index_items)))

    save_index(index_items, OUTPUT_FILE)
    print("Positional index saved to: {}".format(OUTPUT_FILE))

    spark.stop()


if __name__ == "__main__":
    main()