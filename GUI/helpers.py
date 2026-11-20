import csv
import json
import os
import tkinter as tk
from tkinter import ttk

# ---------- colors / font
SIDEBAR, BTN, BTN_HOVER, BTN_ACTIVE = "#5E3F24", "#8B5E34", "#A8743F", "#C08A4E"
BG, BORDER = "#F5EBDD", "#B08B63"
DARK, SOFT = "#5E3F24", "#8B6B4A"
RED, RED_HOVER = "#A8432F", "#C2543C"
FONT = "Segoe UI"

INDEX_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "output", "positional_index.json")


def fmt(x):
    return "0" if x == 0 else "{:.4f}".format(x)


# ---------- reading the files in the output folder (nothing is calculated here)
OUT = os.path.dirname(INDEX_PATH)


def doc_order(d):
    return int(d[1:])          # d2 before d10


def read_matrix(name, cast):
    """Read a 'term,d1..d10' CSV -> {term: {doc: value}}."""
    with open(os.path.join(OUT, name), encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        docs = reader.fieldnames[1:]
        return {row["term"]: {d: cast(row[d]) for d in docs} for row in reader}


def load_data():
    with open(INDEX_PATH, encoding="utf-8") as f:
        index = json.load(f)                                   # positional_index.json

    raw_tf = read_matrix("raw_tf.csv", int)                    # raw_tf.csv
    weighted_tf = read_matrix("weighted_tf.csv", float)        # weighted_tf.csv
    tfidf = read_matrix("tf-idf.csv", float)                   # tf-idf.csv
    normalized = read_matrix("normalized_tfidf.csv", float)    # normalized_tfidf.csv

    df, idf = {}, {}
    with open(os.path.join(OUT, "df_idf.csv"), encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):                          # df_idf.csv
            df[row["term"]] = int(row["DF"])
            idf[row["term"]] = float(row["IDF"])

    lengths = {}
    with open(os.path.join(OUT, "doc_lengths.csv"), encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):                          # doc_lengths.csv
            lengths[row["document"]] = float(row["vector_length"])

    # query results made by partB/similarity_ranking.py
    queries = {}
    with open(os.path.join(OUT, "query_vectors.csv"), encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):                          # query_vectors.csv
            q = queries.setdefault(row["query"], {"vector": [], "length": 0, "ranking": []})
            q["vector"].append((row["term"], float(row["weighted_TF"]), float(row["TF-IDF"]),
                                float(row["normalized"])))
            q["length"] = float(row["query_vector_length"])
    with open(os.path.join(OUT, "rankings.csv"), encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):                          # rankings.csv (already ranked)
            queries[row["query"]]["ranking"].append((row["document"], float(row["similarity"])))

    terms = sorted(raw_tf)
    docs = sorted(lengths, key=doc_order)
    return {"queries": queries, "index": index, "terms": terms, "docs": docs, "N": len(docs),
            "raw_tf": raw_tf, "weighted_tf": weighted_tf, "df": df, "idf": idf,
            "tfidf": tfidf, "lengths": lengths, "normalized": normalized}


# ---------- small widgets
def setup_style(root):
    s = ttk.Style(root)
    s.theme_use("clam")
    s.configure("Treeview", rowheight=22, font=(FONT, 10), background="white",
                fieldbackground="white", foreground="#3b2a18")
    s.configure("Treeview.Heading", font=(FONT, 10, "bold"), background=BTN,
                foreground="white", relief="flat", padding=5)
    s.map("Treeview.Heading", background=[("active", BTN_HOVER)])
    s.map("Treeview", background=[("selected", "#E3C9A3")], foreground=[("selected", "#3b2a18")])


def make_table(parent, columns, widths, height=10):
    """Table with scrollbars. Returns (frame, tree)."""
    frame = tk.Frame(parent, bg="white", highlightbackground=BORDER, highlightthickness=1)
    tree = ttk.Treeview(frame, columns=columns, show="headings", height=height)
    ys = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    xs = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
    for i, (name, w) in enumerate(zip(columns, widths)):
        tree.heading(name, text=name)
        tree.column(name, width=w, minwidth=40, anchor="w" if i == 0 else "center")
    tree.tag_configure("odd", background="#FBF5EC")
    tree.grid(row=0, column=0, sticky="nsew")
    ys.grid(row=0, column=1, sticky="ns")
    xs.grid(row=1, column=0, sticky="ew")
    frame.rowconfigure(0, weight=1)
    frame.columnconfigure(0, weight=1)
    return frame, tree


def fill(tree, rows):
    tree.delete(*tree.get_children())
    for i, row in enumerate(rows):
        tree.insert("", "end", values=row, tags=("odd",) if i % 2 else ())


def title(parent, text, subtitle):
    tk.Label(parent, text=text, font=(FONT, 18, "bold"), bg=BG, fg=DARK,
             anchor="w").pack(fill="x", padx=24, pady=(14, 0))
    tk.Label(parent, text=subtitle, font=(FONT, 10), bg=BG, fg=SOFT, anchor="w",
             wraplength=1150, justify="left").pack(fill="x", padx=24, pady=(0, 8))
