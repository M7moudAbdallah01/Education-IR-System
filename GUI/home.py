import os
import tkinter as tk

from helpers import (
    BG, BORDER, DARK, SOFT, FONT,
    BTN, BTN_HOVER
)


def home_page(parent, data, on_change=None):
    """Home page: summary of the dataset that is currently loaded."""
    page = tk.Frame(parent, bg=BG)

    card = tk.Frame(
        page,
        bg="white",
        highlightbackground=BORDER,
        highlightthickness=2
    )
    card.place(relx=0.5, rely=0.5, anchor="center", width=760, height=440)

    tk.Label(
        card,
        text="Education Information Retrieval System",
        font=(FONT, 19, "bold"),
        bg="white",
        fg=DARK,
        wraplength=680
    ).pack(pady=(30, 12))

    tk.Label(
        card,
        text="Current dataset",
        font=(FONT, 11, "bold"),
        bg="white",
        fg=SOFT
    ).pack()

    dataset_path = data.get("dataset_path") or "Default Dataset (project output folder)"
    tk.Label(
        card,
        text=dataset_path,
        font=(FONT, 9),
        bg="white",
        fg=DARK,
        wraplength=680,
        justify="center"
    ).pack(pady=(5, 12))

    tk.Frame(card, bg=BORDER, height=1).pack(fill="x", padx=45)

    stats = tk.Frame(card, bg="white")
    stats.pack(pady=17)

    stat_items = (
        ("Documents", data.get("N", len(data.get("docs", [])))),
        ("Distinct terms", len(data.get("terms", []))),
    )

    for label, value in stat_items:
        item = tk.Frame(stats, bg="white")
        item.pack(side="left", padx=30)

        tk.Label(
            item, text=str(value), font=(FONT, 20, "bold"),
            bg="white", fg=DARK
        ).pack()

        tk.Label(
            item, text=label, font=(FONT, 10),
            bg="white", fg=SOFT
        ).pack()

    output_dir = data.get("output_dir", "")
    output_label = (
        "Results folder: " + os.path.basename(os.path.normpath(output_dir))
        if output_dir else "Results folder: not specified"
    )

    tk.Label(
        card, text=output_label, font=(FONT, 9),
        bg="white", fg=SOFT, wraplength=680
    ).pack(pady=(0, 12))

    tk.Label(
        card,
        text=(
            "Choose a page from the navigation menu to explore the "
            "Positional Index, term matrices, and search results."
        ),
        font=(FONT, 10), bg="white", fg=SOFT,
        wraplength=680, justify="center"
    ).pack(pady=(5, 14))

    change_button = tk.Button(
        card,
        text="Change dataset",
        command=on_change if on_change else (lambda: None),
        bg=BTN,
        fg="white",
        activebackground=BTN_HOVER,
        activeforeground="white",
        relief="flat",
        bd=0,
        cursor="hand2",
        font=(FONT, 10, "bold"),
        padx=20,
        pady=9
    )
    change_button.pack()

    return page