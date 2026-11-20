import tkinter as tk
from tkinter import messagebox

from helpers import (BG, SIDEBAR, BTN, BTN_HOVER, BTN_ACTIVE, RED, RED_HOVER, FONT,
                     load_data, setup_style)
from home import home_page
from pages import positional_page, tf_page, df_idf_page, tfidf_page, normalized_page

WIDTH, HEIGHT = 1180, 700


def center(win, w, h):
    """Put the window exactly in the middle of the screen (shrinks on small screens)."""
    win.update_idletasks()
    sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
    w, h = min(w, int(sw * 0.92)), min(h, int(sh * 0.88))
    win.geometry("{}x{}+{}+{}".format(w, h, (sw - w) // 2, (sh - h) // 2))


def menu_button(parent, text, command, color, hover):
    b = tk.Button(parent, text=text, command=command, bg=color, fg="white", relief="flat", bd=0,
                  activebackground=hover, activeforeground="white", cursor="hand2",
                  font=(FONT, 11, "bold"), pady=10)
    b.bind("<Enter>", lambda e: b.config(bg=hover))
    b.bind("<Leave>", lambda e: b.config(bg=b.normal_color))
    b.normal_color = color
    return b


def main():
    try:
        data = load_data()
    except FileNotFoundError as error:
        messagebox.showerror("Missing file", "Run Part A and Part B first. Not found:\n" + str(error.filename))
        return

    root = tk.Tk()
    root.title("Education IR System")
    root.configure(bg=BG)
    root.minsize(980, 600)
    center(root, WIDTH, HEIGHT)
    setup_style(root)

    # left menu
    side = tk.Frame(root, bg=SIDEBAR, width=230)
    side.pack(side="left", fill="y")
    side.pack_propagate(False)
    tk.Label(side, text="IR SYSTEM", font=(FONT, 18, "bold"), bg=SIDEBAR, fg="white").pack(pady=(30, 20))

    # right content: all pages stacked, we raise the one we want
    content = tk.Frame(root, bg=BG)
    content.pack(side="left", fill="both", expand=True)
    content.rowconfigure(0, weight=1)
    content.columnconfigure(0, weight=1)

    pages = {"Home": home_page(content, data)}
    menu = [("Positional Index", positional_page), ("TF Matrix", tf_page), ("DF & IDF", df_idf_page),
            ("TF-IDF", tfidf_page), ("Normalized TF-IDF", normalized_page)]
    for name, build in menu:
        pages[name] = build(content, data)
    for page in pages.values():
        page.grid(row=0, column=0, sticky="nsew")

    buttons = {}

    def show(name):
        pages[name].tkraise()
        for n, b in buttons.items():
            b.normal_color = BTN_ACTIVE if n == name else BTN
            b.config(bg=b.normal_color)

    for name, _ in menu:
        b = menu_button(side, name, lambda n=name: show(n), BTN, BTN_HOVER)
        b.pack(fill="x", padx=24, pady=5)
        buttons[name] = b

    menu_button(side, "Exit", root.destroy, RED, RED_HOVER).pack(side="bottom", fill="x", padx=24, pady=26)

    pages["Home"].tkraise()
    root.mainloop()


if __name__ == "__main__":
    main()
