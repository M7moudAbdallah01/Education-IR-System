import tkinter as tk
from helpers import BG, DARK, SOFT, FONT, fmt, doc_order, make_table, fill, title


# 1. Positional Index
def positional_page(parent, data):
    page = tk.Frame(parent, bg=BG)
    title(page, "Positional Index",
          "Each term with the documents it appears in and its 1-based token positions.")

    # search bar
    bar = tk.Frame(page, bg=BG)
    bar.pack(fill="x", padx=24, pady=(0, 8))
    tk.Label(bar, text="Search term:", bg=BG, fg=DARK, font=(FONT, 10, "bold")).pack(side="left")
    search = tk.StringVar()
    tk.Entry(bar, textvariable=search, font=(FONT, 11), width=28, relief="solid", bd=1).pack(side="left", padx=8)
    count = tk.Label(bar, text="", bg=BG, fg=SOFT, font=(FONT, 10))
    count.pack(side="left", padx=10)

    frame, tree = make_table(page, ("Term", "Document IDs and positions"), (170, 640))
    frame.pack(fill="both", expand=True, padx=24, pady=(0, 20))

    def refresh(*_):
        text = search.get().strip().lower()
        rows = []
        for term in data["terms"]:
            if text in term:
                postings = data["index"][term]
                line = "; ".join("{}: {}".format(d, ", ".join(map(str, postings[d])))
                                 for d in sorted(postings, key=doc_order))
                rows.append((term, line))
        fill(tree, rows)
        count.config(text="{} term(s)".format(len(rows)))

    search.trace_add("write", refresh)
    refresh()
    return page


# 2. TF Matrix
def tf_page(parent, data):
    page = tk.Frame(parent, bg=BG)
    title(page, "Term Frequency (TF)",
          "Raw TF = occurrences of the term in the document.   Weighted TF = 1 + log10(TF).")

    docs = data["docs"]
    mode = tk.StringVar(value="raw")
    bar = tk.Frame(page, bg=BG)
    bar.pack(fill="x", padx=24, pady=(0, 6))
    for text, value in (("Raw TF", "raw"), ("Weighted TF", "weighted")):
        tk.Radiobutton(bar, text=text, value=value, variable=mode, command=lambda: refresh(),
                       bg=BG, fg=DARK, selectcolor="white", activebackground=BG,
                       font=(FONT, 10, "bold")).pack(side="left", padx=(0, 16))

    frame, tree = make_table(page, ["Term"] + docs, [130] + [80] * len(docs))
    frame.pack(fill="both", expand=True, padx=24, pady=(0, 20))

    def refresh():
        raw = mode.get() == "raw"
        matrix = data["raw_tf"] if raw else data["weighted_tf"]
        rows = [[t] + [str(matrix[t][d]) if raw else fmt(matrix[t][d]) for d in docs]
                for t in data["terms"]]
        fill(tree, rows)

    refresh()
    return page


# 3. DF & IDF
def df_idf_page(parent, data):
    page = tk.Frame(parent, bg=BG)
    title(page, "Document Frequency & IDF",
          "DF = number of documents containing the term.   IDF = log10(N / DF),  N = {}.".format(data["N"]))

    frame, tree = make_table(page, ("Term", "DF", "IDF = log10(N / DF)"), (220, 120, 240))
    frame.pack(fill="both", expand=True, padx=24, pady=(0, 20))
    fill(tree, [(t, data["df"][t], fmt(data["idf"][t])) for t in data["terms"]])
    return page


# 4. TF-IDF
def matrix_rows(data, matrix):
    return [[t] + [fmt(matrix[t][d]) for d in data["docs"]] for t in data["terms"]]


def tfidf_page(parent, data):
    page = tk.Frame(parent, bg=BG)
    title(page, "TF-IDF Matrix", "TF-IDF = Weighted TF x IDF.")
    docs = data["docs"]
    frame, tree = make_table(page, ["Term"] + docs, [130] + [80] * len(docs))
    frame.pack(fill="both", expand=True, padx=24, pady=(0, 20))
    fill(tree, matrix_rows(data, data["tfidf"]))
    return page


# 5. Normalized TF-IDF
def normalized_page(parent, data):
    page = tk.Frame(parent, bg=BG)
    title(page, "Normalized TF-IDF",
          "Each TF-IDF weight divided by its document vector length (L2 norm).")
    docs = data["docs"]
    widths = [130] + [80] * len(docs)

    # document vector lengths (small table at the bottom)
    tk.Label(page, text="Document vector lengths", bg=BG, fg=SOFT,
             font=(FONT, 10, "bold")).pack(side="bottom", anchor="w", padx=24)
    lframe, ltree = make_table(page, ["Document"] + docs, [130] + [80] * len(docs), height=1)
    lframe.pack(side="bottom", fill="x", padx=24, pady=(2, 16))
    fill(ltree, [["Length"] + ["{:.4f}".format(data["lengths"][d]) for d in docs]])

    frame, tree = make_table(page, ["Term"] + docs, widths)
    frame.pack(fill="both", expand=True, padx=24, pady=(0, 10))
    fill(tree, matrix_rows(data, data["normalized"]))
    return page
