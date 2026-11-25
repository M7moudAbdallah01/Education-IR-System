import tkinter as tk

from helpers import BG, BORDER, DARK, SOFT, FONT, BTN, BTN_HOVER, BTN_ACTIVE


def _action_button(parent, text, command, color):
    button = tk.Button(
        parent,
        text=text,
        command=command,
        bg=color,
        fg="white",
        activebackground=BTN_HOVER,
        activeforeground="white",
        relief="flat",
        bd=0,
        cursor="hand2",
        font=(FONT, 11, "bold"),
        padx=22,
        pady=12,
        width=24,
    )
    button.bind("<Enter>", lambda e: button.config(bg=BTN_HOVER))
    button.bind("<Leave>", lambda e: button.config(bg=color))
    return button

def start_page(parent, on_default, on_browse):
    """
    First screen of the application: welcome text and the two ways to start.
    It is shown without the sidebar.
    """
    page = tk.Frame(parent, bg=BTN)

    card = tk.Frame(
        page, bg="white",
        highlightbackground=BORDER, highlightthickness=2
    )
    card.place(relx=0.5, rely=0.5, anchor="center", width=820, height=520)

    tk.Label(
        card, text="Education Information Retrieval System",
        font=(FONT, 22, "bold"), bg="white", fg=DARK, wraplength=740
    ).pack(pady=(40, 8))

    tk.Label(
        card, text="IS 313 - Information Storage and Retrieval",
        font=(FONT, 11, "bold"), bg="white", fg=SOFT
    ).pack()

    tk.Frame(card, bg=BORDER, height=1).pack(fill="x", padx=60, pady=22)

    tk.Label(
        card,
        text=(
            "Welcome! This system builds a Positional Index with PySpark and "
            "lets you explore term frequencies, IDF, TF-IDF matrices, phrase "
            "and Boolean queries, and cosine-similarity rankings.\n\n"
            "Choose how you want to start:"
        ),
        font=(FONT, 11), bg="white", fg=DARK, wraplength=700, justify="center"
    ).pack(pady=(0, 22))

    actions = tk.Frame(card, bg="white")
    actions.pack()

    default_col = tk.Frame(actions, bg="white")
    default_col.pack(side="left", padx=18)
    _action_button(default_col, "Use default Dataset", on_default, BTN_ACTIVE).pack()
    tk.Label(
        default_col,
        text="Open the results already saved\nin the output folder.",
        font=(FONT, 9), bg="white", fg=SOFT, justify="center"
    ).pack(pady=(8, 0))

    browse_col = tk.Frame(actions, bg="white")
    browse_col.pack(side="left", padx=18)
    _action_button(browse_col, "Browse for new dataset", on_browse, BTN).pack()
    tk.Label(
        browse_col,
        text="Select a folder of .txt files and\nrun the full pipeline on it.",
        font=(FONT, 9), bg="white", fg=SOFT, justify="center"
    ).pack(pady=(8, 0))

    return page
