import tkinter as tk
from helpers import BG, BORDER, DARK, SOFT, FONT


def home_page(parent, data):
    page = tk.Frame(parent, bg=BG)
    card = tk.Frame(page, bg="white", highlightbackground=BORDER, highlightthickness=2,
                    width=520, height=340)
    card.place(relx=0.5, rely=0.5, anchor="center")
    card.pack_propagate(False)

    tk.Label(card, text="Welcome to Education IR System", font=(FONT, 18, "bold"),
             bg="white", fg=DARK).pack(pady=(45, 15))
    tk.Label(card, text="Use the menu on the left\nto explore the system",
             font=(FONT, 11), bg="white", fg=SOFT).pack(pady=(0, 25))
    tk.Frame(card, bg=BORDER, height=1).pack(fill="x", padx=60)
    tk.Label(card, text="Documents: {}      Distinct terms: {}".format(data["N"], len(data["terms"])),
             font=(FONT, 10, "bold"), bg="white", fg=DARK).pack(pady=18)
    return page
