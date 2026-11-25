import math
import re
import tkinter as tk

from helpers import (
    BG, BTN, BTN_HOVER, BTN_ACTIVE, BORDER, DARK, SOFT, FONT,
    fmt, doc_order, make_table, fill, title
)


PHRASE_1 = "online learning"
PHRASE_2 = "student assessment"


def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def phrase_matches(index, phrase):
    words = tokens(phrase)
    if not words:
        return {}

    postings = [index.get(word, {}) for word in words]
    if any(not posting for posting in postings):
        return {}

    common_documents = set(postings[0])
    for posting in postings[1:]:
        common_documents.intersection_update(posting)

    matches = {}
    for document in common_documents:
        starts = [
            start for start in postings[0][document]
            if all(
                start + offset in postings[offset].get(document, [])
                for offset in range(1, len(words))
            )
        ]
        if starts:
            matches[document] = [
                (start, start + len(words) - 1) for start in starts
            ]
    return matches


def evaluate_query(data, query, mode):
    query_text = query.strip()
    if not query_text:
        raise ValueError("Please enter a search query.")

    index = data.get("index", {})

    if mode == "AND NOT":
        parts = re.split(r"\s+AND\s+NOT\s+", query_text, flags=re.IGNORECASE)
        if len(parts) != 2 or not all(tokens(part) for part in parts):
            raise ValueError("Enter two phrases in this format: phrase AND NOT phrase.")
        first_matches = phrase_matches(index, parts[0])
        second_matches = phrase_matches(index, parts[1])
        documents = sorted(set(first_matches) - set(second_matches), key=doc_order)
        query_terms = tokens(parts[0])

    elif mode == "AND":
        parts = re.split(r"\s+AND\s+", query_text, flags=re.IGNORECASE)
        if len(parts) != 2 or not all(tokens(part) for part in parts):
            raise ValueError("Enter two phrases in this format: phrase AND phrase.")
        first_matches = phrase_matches(index, parts[0])
        second_matches = phrase_matches(index, parts[1])
        documents = sorted(set(first_matches).intersection(second_matches), key=doc_order)
        query_terms = tokens(parts[0]) + tokens(parts[1])

    elif mode == "Phrase":
        matches = phrase_matches(index, query_text)
        documents = sorted(matches, key=doc_order)
        query_terms = tokens(query_text)

    elif mode == "Term":
        query_terms = tokens(query_text)
        if len(query_terms) != 1:
            raise ValueError("Term mode accepts exactly one word.")
        postings = index.get(query_terms[0], {})
        documents = sorted(postings, key=doc_order)
        matches = {
            doc: [(position, position) for position in positions]
            for doc, positions in postings.items()
        }

    else:
        raise ValueError(f"Unsupported search mode: {mode}")

    term_counts = {}
    for term in query_terms:
        term_counts[term] = term_counts.get(term, 0) + 1

    idf_values = data.get("idf", {})
    query_weights = {
        term: (1 + math.log10(freq)) * idf_values.get(term, 0)
        for term, freq in term_counts.items()
    }
    query_length = math.sqrt(sum(weight * weight for weight in query_weights.values()))
    normalized_query = {
        term: weight / query_length if query_length else 0
        for term, weight in query_weights.items()
    }

    normalized_matrix = data.get("normalized", {})
    ranking = sorted(
        (
            (
                doc,
                sum(
                    normalized_query.get(term, 0)
                    * normalized_matrix.get(term, {}).get(doc, 0)
                    for term in normalized_query
                )
            )
            for doc in documents
        ),
        key=lambda item: (-item[1], doc_order(item[0]))
    )

    query_vector = [
        (term, term_counts[term], query_weights[term], normalized_query[term])
        for term in term_counts
    ]
    return documents, query_vector, query_length, ranking


def _button(parent, text, command, active=False):
    normal = BTN_ACTIVE if active else BTN
    button = tk.Button(
        parent,
        text=text,
        command=command,
        bg=normal,
        fg="white",
        activebackground=BTN_HOVER,
        activeforeground="white",
        relief="flat",
        bd=0,
        font=(FONT, 9, "bold"),
        padx=12,
        pady=7,
        cursor="hand2"
    )
    button.normal_color = normal
    button.bind("<Enter>", lambda _event: button.config(bg=BTN_HOVER))
    button.bind("<Leave>", lambda _event: button.config(bg=button.normal_color))
    return button


def _positions_text(found):
    if not found:
        return "-"
    return ", ".join(
        str(start) if start == end else f"{start}-{end}"
        for start, end in found
    )


def _result_tables(parent, boolean=False):
    area = tk.Frame(parent, bg=BG)
    area.pack(fill="both", expand=True, padx=20, pady=(6, 14))

    for column in range(3):
        area.columnconfigure(column, weight=1, uniform="result_columns")
    area.rowconfigure(1, weight=1)

    match_columns = (
        ("Document", f'"{PHRASE_1}"', f'"{PHRASE_2}"')
        if boolean else ("Document", "Positions / ranges")
    )
    match_widths = (100, 150, 150) if boolean else (110, 230)

    specs = [
        ("Matching documents", match_columns, match_widths),
        ("Query vector", ("Term", "TF", "TF-IDF", "Normalized"), (85, 55, 85, 95)),
        ("Ranking", ("Rank", "Document", "Cosine similarity"), (55, 95, 130)),
    ]

    trees = []
    for column, (heading, columns, widths) in enumerate(specs):
        tk.Label(
            area, text=heading, bg=BG, fg=SOFT,
            font=(FONT, 10, "bold"), anchor="w"
        ).grid(row=0, column=column, sticky="w", padx=5, pady=(0, 4))

        frame, tree = make_table(area, columns, widths, height=9)
        frame.grid(row=1, column=column, sticky="nsew", padx=5)
        trees.append(tree)

    return trees


def _fill_results(data, trees, query_text, mode, summary):
    documents, vector, query_length, ranking = evaluate_query(data, query_text, mode)

    if mode == "Phrase":
        found = phrase_matches(data.get("index", {}), query_text)
        rows = [
            (doc, _positions_text(found.get(doc, [])))
            for doc in documents
        ]
        fill(trees[0], rows)

    else:
        parts = re.split(
            r"\s+AND(?:\s+NOT)?\s+",
            query_text.strip(),
            flags=re.IGNORECASE
        )
        first = phrase_matches(data.get("index", {}), parts[0])
        second = phrase_matches(data.get("index", {}), parts[1])
        rows = [
            (
                doc,
                _positions_text(first.get(doc, [])),
                _positions_text(second.get(doc, []))
            )
            for doc in documents
        ]
        fill(trees[0], rows)

    fill(trees[1], [
        (term, frequency, fmt(weight), fmt(normalized))
        for term, frequency, weight, normalized in vector
    ])
    fill(trees[2], [
        (rank, document, fmt(score))
        for rank, (document, score) in enumerate(ranking, start=1)
    ])

    summary.config(
        text=(
            f'{len(documents)} matching document(s)  |  '
            f'Query vector length = {fmt(query_length)}'
        ),
        fg=DARK
    )


def phrase_page(parent, data):
    page = tk.Frame(parent, bg=BG)
    title(
        page,
        "Phrase Search",
        "Choose one of the assigned phrases or enter your own phrase to search."
    )

    summary = tk.Label(
        page, text="Choose a phrase or enter a custom phrase.",
        bg=BG, fg=DARK, font=(FONT, 10, "bold"), anchor="w"
    )
    summary.pack(fill="x", padx=24, pady=(2, 8))

    preset_row = tk.Frame(page, bg=BG)
    preset_row.pack(fill="x", padx=24, pady=(0, 8))
    tk.Label(
        preset_row, text="Assigned phrases:",
        bg=BG, fg=DARK, font=(FONT, 10, "bold")
    ).pack(side="left", padx=(0, 8))

    def run(phrase):
        try:
            _fill_results(data, trees, phrase, "Phrase", summary)
        except (ValueError, KeyError, TypeError) as error:
            summary.config(text=str(error), fg="#A8432F")
            for tree in trees:
                fill(tree, [])

    for phrase in (PHRASE_1, PHRASE_2):
        _button(
            preset_row,
            f'"{phrase}"',
            lambda value=phrase: run(value)
        ).pack(side="left", padx=5)

    custom_row = tk.Frame(page, bg=BG)
    custom_row.pack(fill="x", padx=24, pady=(0, 8))
    tk.Label(
        custom_row, text="Custom phrase:",
        bg=BG, fg=DARK, font=(FONT, 10, "bold")
    ).pack(side="left")

    custom_phrase = tk.StringVar()
    entry = tk.Entry(custom_row, textvariable=custom_phrase, font=(FONT, 11), width=42)
    entry.pack(side="left", padx=8)

    _button(
        custom_row, "Search",
        lambda: run(custom_phrase.get())
    ).pack(side="left")

    entry.bind("<Return>", lambda _event: run(custom_phrase.get()))

    trees = _result_tables(page, boolean=False)
    run(PHRASE_1)
    return page


def boolean_page(parent, data):
    page = tk.Frame(parent, bg=BG)
    title(
        page,
        "Boolean Queries",
        "Run the assigned Boolean queries or enter two custom phrases and choose AND / AND NOT."
    )

    summary = tk.Label(
        page, text="Choose an assigned query or create your own.",
        bg=BG, fg=DARK, font=(FONT, 10, "bold"), anchor="w"
    )
    summary.pack(fill="x", padx=24, pady=(2, 8))

    preset_row = tk.Frame(page, bg=BG)
    preset_row.pack(fill="x", padx=24, pady=(0, 8))
    tk.Label(
        preset_row, text="Assigned queries:",
        bg=BG, fg=DARK, font=(FONT, 10, "bold")
    ).pack(side="left", padx=(0, 8))

    def run(first_phrase, operator, second_phrase):
        query_text = f"{first_phrase.strip()} {operator} {second_phrase.strip()}"
        try:
            _fill_results(data, trees, query_text, operator, summary)
        except (ValueError, KeyError, TypeError) as error:
            summary.config(text=str(error), fg="#A8432F")
            for tree in trees:
                fill(tree, [])

    _button(
        preset_row,
        f'"{PHRASE_1}" AND "{PHRASE_2}"',
        lambda: run(PHRASE_1, "AND", PHRASE_2)
    ).pack(side="left", padx=5)

    _button(
        preset_row,
        f'"{PHRASE_1}" AND NOT "{PHRASE_2}"',
        lambda: run(PHRASE_1, "AND NOT", PHRASE_2)
    ).pack(side="left", padx=5)

    custom_box = tk.Frame(page, bg=BG)
    custom_box.pack(fill="x", padx=24, pady=(0, 8))

    tk.Label(
        custom_box, text="First phrase:",
        bg=BG, fg=DARK, font=(FONT, 10, "bold")
    ).grid(row=0, column=0, sticky="w", padx=(0, 6), pady=3)

    first_value = tk.StringVar(value=PHRASE_1)
    first_entry = tk.Entry(custom_box, textvariable=first_value, font=(FONT, 10), width=25)
    first_entry.grid(row=0, column=1, sticky="w", padx=(0, 12), pady=3)

    tk.Label(
        custom_box, text="Operator:",
        bg=BG, fg=DARK, font=(FONT, 10, "bold")
    ).grid(row=0, column=2, sticky="w", padx=(0, 6), pady=3)

    operator_value = tk.StringVar(value="AND")
    operator_menu = tk.OptionMenu(custom_box, operator_value, "AND", "AND NOT")
    operator_menu.config(
        bg=BTN, fg="white", activebackground=BTN_HOVER,
        activeforeground="white", relief="flat", bd=0,
        font=(FONT, 9, "bold"), highlightthickness=0
    )
    operator_menu["menu"].config(font=(FONT, 9))
    operator_menu.grid(row=0, column=3, sticky="w", padx=(0, 12), pady=3)

    tk.Label(
        custom_box, text="Second phrase:",
        bg=BG, fg=DARK, font=(FONT, 10, "bold")
    ).grid(row=0, column=4, sticky="w", padx=(0, 6), pady=3)

    second_value = tk.StringVar(value=PHRASE_2)
    second_entry = tk.Entry(custom_box, textvariable=second_value, font=(FONT, 10), width=25)
    second_entry.grid(row=0, column=5, sticky="w", padx=(0, 8), pady=3)

    _button(
        custom_box, "Run Custom Query",
        lambda: run(first_value.get(), operator_value.get(), second_value.get())
    ).grid(row=0, column=6, sticky="w", pady=3)

    first_entry.bind(
        "<Return>",
        lambda _event: run(first_value.get(), operator_value.get(), second_value.get())
    )
    second_entry.bind(
        "<Return>",
        lambda _event: run(first_value.get(), operator_value.get(), second_value.get())
    )

    trees = _result_tables(page, boolean=True)
    run(PHRASE_1, "AND", PHRASE_2)
    return page
