import tkinter as tk
from helpers import (BG, BTN, BTN_HOVER, BTN_ACTIVE, BORDER, DARK, SOFT, FONT, fmt, doc_order,
                     make_table, fill, title)

PHRASE_1, PHRASE_2 = "online learning", "student assessment"


def spans(data, phrase):
    """{doc: [(start, end)]} - where the phrase sits in each document (from the positional index)."""
    words = phrase.split()
    postings = [data["index"][w] for w in words]
    result = {}
    for doc in postings[0]:
        found = [(s, s + len(words) - 1) for s in postings[0][doc]
                 if all(doc in postings[i] and s + i in postings[i][doc] for i in range(1, len(words)))]
        if found:
            result[doc] = found
    return result


def ranges_text(found):
    return ", ".join("{}-{}".format(s, e) for s, e in found) if found else "-"


def button(parent, text, command):
    b = tk.Button(parent, text=text, command=command, bg=BTN, fg="white", relief="flat", bd=0,
                  activebackground=BTN_HOVER, activeforeground="white", cursor="hand2",
                  font=(FONT, 10, "bold"), padx=14, pady=7)
    b.normal_color = BTN
    b.bind("<Enter>", lambda e: b.config(bg=BTN_HOVER))
    b.bind("<Leave>", lambda e: b.config(bg=b.normal_color))
    return b


def result_panel(parent, data, match_columns, match_widths):
    """Summary + 3 tables + ranking line. Returns (panel, show(summary, match_rows, query_name))."""
    panel = tk.Frame(parent, bg=BG)
    summary = tk.Label(panel, bg=BG, fg=DARK, font=(FONT, 11, "bold"), anchor="w", justify="left",
                       wraplength=1150)
    summary.pack(fill="x", pady=(4, 8))

    grid = tk.Frame(panel, bg=BG)
    grid.pack(fill="both", expand=True)
    grid.columnconfigure((0, 1, 2), weight=1, uniform="c")
    grid.rowconfigure(1, weight=1)
    blocks = [("Matching documents", match_columns, match_widths),
              ("Query vector", ("Term", "Weighted TF", "TF-IDF", "Normalized"), (90, 90, 80, 90)),
              ("Similarity scores", ("Rank", "Document", "Similarity"), (60, 90, 100))]
    trees = []
    for col, (name, cols, widths) in enumerate(blocks):
        tk.Label(grid, text=name, bg=BG, fg=SOFT, font=(FONT, 10, "bold")).grid(row=0, column=col, sticky="w", padx=4)
        frame, tree = make_table(grid, cols, widths, height=6)
        frame.grid(row=1, column=col, sticky="nsew", padx=4)
        trees.append(tree)

    length_label = tk.Label(panel, bg=BG, fg=SOFT, font=(FONT, 10), anchor="w")
    length_label.pack(fill="x", pady=(8, 0))
    ranking_label = tk.Label(panel, bg="white", fg=DARK, font=(FONT, 12, "bold"), anchor="w",
                             padx=12, pady=8, highlightbackground=BORDER, highlightthickness=1)
    ranking_label.pack(fill="x", pady=(6, 0))

    def show(text, match_rows, name):
        q = data["queries"][name]
        summary.config(text=text)
        fill(trees[0], match_rows)
        fill(trees[1], [(t, fmt(w), fmt(x), fmt(n)) for t, w, x, n in q["vector"]])
        fill(trees[2], [(i + 1, d, fmt(s)) for i, (d, s) in enumerate(q["ranking"])])
        length_label.config(text="Query vector length = {}".format(q["length"]))
        ranking_label.config(text="Ranking (highest similarity first):  " +
                             " > ".join(d for d, _ in q["ranking"]))

    return panel, show


def query_buttons(page, label, items):
    """Row of query buttons. items = [(button text, function)]. Highlights the chosen one."""
    row = tk.Frame(page, bg=BG)
    row.pack(fill="x", padx=24, pady=(0, 8))
    tk.Label(row, text=label, bg=BG, fg=DARK, font=(FONT, 10, "bold")).pack(side="left")
    buttons = []

    def choose(i):
        for j, b in enumerate(buttons):
            b.normal_color = BTN_ACTIVE if j == i else BTN
            b.config(bg=b.normal_color)
        items[i][1]()

    for i, (text, _) in enumerate(items):
        b = button(row, text, lambda i=i: choose(i))
        b.pack(side="left", padx=6)
        buttons.append(b)
    return choose


# ---------------------------------------------------------------- Phrase Search
def phrase_page(parent, data):
    page = tk.Frame(parent, bg=BG)
    title(page, "Phrase Search",
          "Exact, contiguous phrase matching using token positions, then ranking the matching documents by similarity.")
    panel, show = result_panel(page, data, ("Document", "Start positions", "Token ranges"), (90, 110, 130))

    def run(phrase):
        found = spans(data, phrase)
        docs = [d for d, _ in data["queries"]["Phrase: " + phrase]["ranking"]]
        rows = [(d, ", ".join(str(s) for s, _ in found[d]), ranges_text(found[d]))
                for d in sorted(docs, key=doc_order)]
        show('Phrase query: "{}"  -  {} matching document(s): {}'.format(
            phrase, len(docs), ", ".join(sorted(docs, key=doc_order))), rows, "Phrase: " + phrase)

    choose = query_buttons(page, "Assigned queries:",
                           [('"{}"'.format(p), lambda p=p: run(p)) for p in (PHRASE_1, PHRASE_2)])
    panel.pack(fill="both", expand=True, padx=24, pady=(0, 16))
    choose(0)
    return page


# ---------------------------------------------------------------- Boolean Queries
def boolean_page(parent, data):
    page = tk.Frame(parent, bg=BG)
    title(page, "Boolean Queries",
          "AND = documents matching both phrases.   AND NOT = documents matching the first phrase but not the second.")
    panel, show = result_panel(page, data, ("Document", '"online learning"', '"student assessment"'), (90, 130, 140))

    def docs_of(name):
        return sorted((d for d, _ in data["queries"][name]["ranking"]), key=doc_order)

    def run(operator):
        name = "online learning {} student assessment".format(operator)
        result = docs_of(name)
        m1, m2 = spans(data, PHRASE_1), spans(data, PHRASE_2)
        rows = [(d, ranges_text(m1.get(d)), ranges_text(m2.get(d))) for d in result]
        text = ('"{}" {} "{}"\n"{}" matches: {}   |   "{}" matches: {}   |   Result: {}'.format(
            PHRASE_1, operator, PHRASE_2,
            PHRASE_1, ", ".join(docs_of("Phrase: " + PHRASE_1)),
            PHRASE_2, ", ".join(docs_of("Phrase: " + PHRASE_2)), ", ".join(result)))
        show(text, rows, name)

    choose = query_buttons(page, "Assigned queries:",
                           [("online learning AND student assessment", lambda: run("AND")),
                            ("online learning AND NOT student assessment", lambda: run("AND NOT"))])
    panel.pack(fill="both", expand=True, padx=24, pady=(0, 16))
    choose(0)
    return page
