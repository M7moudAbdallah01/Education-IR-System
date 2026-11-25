import tkinter as tk
from helpers import BG, DARK, SOFT, FONT, fmt, doc_order, make_table, fill, title


def positional_page(parent, data):
    page = tk.Frame(parent, bg=BG)
    title(page, "Positional Index", "Term → document → 1-based token positions.")

    toolbar = tk.Frame(page, bg=BG)
    toolbar.pack(fill="x", padx=24, pady=(0, 8))
    tk.Label(toolbar, text="Filter term:", bg=BG, fg=DARK, font=(FONT, 10)).pack(side="left")

    query = tk.StringVar()
    tk.Entry(toolbar, textvariable=query, width=30, font=(FONT, 11)).pack(side="left", padx=8)
    count_label = tk.Label(toolbar, bg=BG, fg=SOFT, font=(FONT, 9))
    count_label.pack(side="left")

    frame, tree = make_table(page, ("Term", "Document IDs and positions"), (180, 850))
    frame.pack(fill="both", expand=True, padx=24, pady=(0, 18))

    def refresh(*_):
        search_text = query.get().strip().lower()
        rows = []
        for term in data.get("terms", []):
            if search_text in term.lower():
                postings = data.get("index", {}).get(term, {})
                details = "; ".join(
                    f"{document}: {', '.join(map(str, positions))}"
                    for document, positions in sorted(postings.items(),
                                                    key=lambda item: doc_order(item[0]))
                )
                rows.append((term, details))
        fill(tree, rows)
        count_label.config(text=f"{len(rows)} term(s)")

    query.trace_add("write", refresh)
    refresh()
    return page


def _matrix_page(parent, data, page_name, subtitle, matrix):
    page = tk.Frame(parent, bg=BG)
    title(page, page_name, subtitle)
    documents = data.get("docs", [])
    frame, tree = make_table(page, ["Term"] + documents, [160] + [100] * len(documents))
    frame.pack(fill="both", expand=True, padx=24, pady=(0, 18))
    fill(tree, [[term] + [fmt(matrix.get(term, {}).get(document, 0)) for document in documents]
                for term in data.get("terms", [])])
    return page


def tf_page(parent, data):
    page = tk.Frame(parent, bg=BG)
    title(page, "Term Frequency",
          "Raw TF counts and weighted TF = 1 + log10(TF) for TF > 0.")

    toolbar = tk.Frame(page, bg=BG)
    toolbar.pack(fill="x", padx=24, pady=(0, 4))
    mode = tk.StringVar(value="raw")
    documents = data.get("docs", [])

    frame, tree = make_table(page, ["Term"] + documents, [160] + [100] * len(documents))
    frame.pack(fill="both", expand=True, padx=24, pady=12)

    def refresh():
        matrix = data.get("raw_tf" if mode.get() == "raw" else "weighted_tf", {})
        fill(tree, [[term] + [fmt(matrix.get(term, {}).get(document, 0)) for document in documents]
                    for term in data.get("terms", [])])

    for label, value in (("Raw TF", "raw"), ("Weighted TF", "weighted")):
        tk.Radiobutton(toolbar, text=label, variable=mode, value=value, command=refresh,
                       bg=BG, fg=DARK, activebackground=BG, font=(FONT, 10)).pack(
            side="left", padx=(0, 12))

    refresh()
    return page


def df_idf_page(parent, data):
    page = tk.Frame(parent, bg=BG)
    title(page, "Document Frequency & IDF",
          f"DF = number of documents containing the term; IDF = log10(N / DF), N = {data.get('N', 0)}.")
    frame, tree = make_table(page, ("Term", "DF", "IDF"), (300, 150, 220))
    frame.pack(fill="both", expand=True, padx=24, pady=15)
    fill(tree, [(term, data.get("df", {}).get(term, 0), fmt(data.get("idf", {}).get(term, 0)))
                for term in data.get("terms", [])])
    return page


def tfidf_page(parent, data):
    return _matrix_page(parent, data, "TF-IDF Matrix", "TF-IDF = Weighted TF × IDF.",
                        data.get("tfidf", {}))


def normalized_page(parent, data):
    page = tk.Frame(parent, bg=BG)
    title(page, "Normalized TF-IDF",
          "Each TF-IDF weight is divided by the L2 length of its document vector.")

    documents = data.get("docs", [])
    matrix = data.get("normalized", {})
    frame, tree = make_table(page, ["Term"] + documents, [160] + [100] * len(documents))
    frame.pack(fill="both", expand=True, padx=24, pady=(8, 5))
    fill(tree, [[term] + [fmt(matrix.get(term, {}).get(document, 0)) for document in documents]
                for term in data.get("terms", [])])

    tk.Label(page, text="Document vector lengths", bg=BG, fg=DARK,
             font=(FONT, 10, "bold")).pack(anchor="w", padx=24)

    table_height = max(1, min(5, len(documents)))
    lengths_frame, lengths_tree = make_table(page, ["Document", "Vector length"],
                                             [260, 200], height=table_height)
    lengths_frame.pack(fill="x", padx=24, pady=(4, 15))
    fill(lengths_tree, [(document, fmt(data.get("lengths", {}).get(document, 0)))
                        for document in documents])
    return page