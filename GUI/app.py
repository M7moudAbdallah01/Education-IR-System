import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from helpers import (
    BG, SIDEBAR, BTN, BTN_HOVER, BTN_ACTIVE, RED, RED_HOVER, FONT, DARK, SOFT,
    DEFAULT_OUTPUT_DIR, load_data, setup_style
)
from runner import (
    list_txt_files, make_output_dir, run_pipeline, remove_incomplete_output
)

from start_page import start_page
from home import home_page
from pages import (
    positional_page,
    tf_page,
    df_idf_page,
    tfidf_page,
    normalized_page
)

from queries import phrase_page, boolean_page
from tests_page import tests_page


WIDTH, HEIGHT = 1500, 940


def center(win, w, h):
    """Put the window in the center of the screen."""
    win.update_idletasks()

    sw = win.winfo_screenwidth()
    sh = win.winfo_screenheight()

    w = min(w, int(sw * 0.96))
    h = min(h, int(sh * 0.92))

    x = (sw - w) // 2
    y = (sh - h) // 2

    win.geometry("{}x{}+{}+{}".format(w, h, x, y))


def menu_button(parent, text, command, color, hover):
    """Create a styled sidebar button."""

    button = tk.Button(
        parent,
        text=text,
        command=command,
        bg=color,
        fg="white",
        relief="flat",
        bd=0,
        activebackground=hover,
        activeforeground="white",
        cursor="hand2",
        font=(FONT, 11, "bold"),
        pady=10
    )

    button.normal_color = color

    button.bind("<Enter>", lambda event: button.config(bg=hover))
    button.bind("<Leave>", lambda event: button.config(bg=button.normal_color))

    return button


def main():
    root = tk.Tk()
    root.title("Education IR System")
    root.configure(bg=BG)
    root.minsize(1000, 620)
    center(root, WIDTH, HEIGHT)
    setup_style(root)

    # -----------------------------------------------------------------
    # Two top-level screens: the start screen (no sidebar) and the
    # workspace (sidebar on the left + the pages on the right).
    # -----------------------------------------------------------------
    start_holder = tk.Frame(root, bg=BG)
    workspace = tk.Frame(root, bg=BG)

    side = tk.Frame(workspace, bg=SIDEBAR, width=230)
    side.pack(side="left", fill="y")
    side.pack_propagate(False)

    tk.Label(
        side, text="IR SYSTEM", font=(FONT, 18, "bold"),
        bg=SIDEBAR, fg="white"
    ).pack(pady=(30, 20))

    content = tk.Frame(workspace, bg=BG)
    content.pack(side="left", fill="both", expand=True)
    content.rowconfigure(0, weight=1)
    content.columnconfigure(0, weight=1)

    menu = [
        ("Positional Index", positional_page),
        ("TF Matrix", tf_page),
        ("DF & IDF", df_idf_page),
        ("TF-IDF", tfidf_page),
        ("Normalized TF-IDF", normalized_page),
        ("Phrase Search", phrase_page),
        ("Boolean Queries", boolean_page),
        ("Tests", tests_page)
    ]

    pages = {}
    buttons = {}

    def show_start():
        workspace.pack_forget()
        start_holder.pack(fill="both", expand=True)

    def show_workspace():
        start_holder.pack_forget()
        workspace.pack(fill="both", expand=True)

    def show(name):
        """Display the selected page and update the sidebar."""
        pages[name].tkraise()
        for page_name, button in buttons.items():
            button.normal_color = BTN_ACTIVE if page_name == name else BTN
            button.config(bg=button.normal_color)

    def rebuild_pages(new_data):
        """Recreate every page so all of them use only the newly loaded dataset."""
        nonlocal pages

        # Destroy every widget in the content area, including any page/table
        # widget that may not be referenced by the current pages dictionary.
        for child in content.winfo_children():
            child.destroy()

        pages = {"Home": home_page(content, new_data, on_change=show_start)}

        for name, build_page in menu:
            pages[name] = build_page(content, new_data)

        for page in pages.values():
            page.grid(row=0, column=0, sticky="nsew")

        show("Home")

    def open_dataset(output_dir, dataset_path, is_default):
        """Load a dataset's result files, then enter the workspace (Home page)."""
        new_data = load_data(output_dir=output_dir, dataset_path=dataset_path)
        new_data["is_default"] = is_default
        rebuild_pages(new_data)
        show_workspace()
        show("Home")

    # -----------------------------------------------------------------
    # Use default dataset: only read and display what is in output/
    # -----------------------------------------------------------------
    def use_default_dataset():
        try:
            open_dataset(DEFAULT_OUTPUT_DIR, "", True)
        except FileNotFoundError as error:
            messagebox.showerror(
                "Default dataset not found",
                "The output folder does not contain all the generated result files.\n\n"
                "Run main.py first, or use 'Browse for new dataset'.\n\n" + str(error),
                parent=root
            )
        except Exception as error:
            messagebox.showerror(
                "Loading Error",
                "Could not load the default dataset.\n\n" + str(error),
                parent=root
            )

    # -----------------------------------------------------------------
    # Browse for new dataset: run main (full pipeline) in a worker thread
    # while a "please wait" window is shown, then open the Home page.
    # -----------------------------------------------------------------
    def show_progress_window(dataset_name):
        win = tk.Toplevel(root)
        win.title("Processing")
        win.configure(bg="white")
        win.resizable(False, False)
        win.transient(root)
        win.protocol("WM_DELETE_WINDOW", lambda: None)  # cannot be closed

        tk.Label(
            win, text="Processing the dataset...",
            font=(FONT, 14, "bold"), bg="white", fg=DARK
        ).pack(padx=50, pady=(28, 6))

        tk.Label(
            win, text=dataset_name, font=(FONT, 10, "bold"),
            bg="white", fg=SOFT, wraplength=380
        ).pack(padx=30)

        tk.Label(
            win,
            text=("Running the full pipeline with PySpark:\n"
                  "Positional Index, TF / IDF, TF-IDF, queries, ranking.\n"
                  "Please wait, this may take a minute."),
            font=(FONT, 10), bg="white", fg=SOFT, justify="center"
        ).pack(padx=30, pady=(10, 14))

        bar = ttk.Progressbar(win, mode="indeterminate", length=320)
        bar.pack(padx=40, pady=(0, 28))
        bar.start(12)

        win.update_idletasks()
        x = root.winfo_rootx() + (root.winfo_width() - win.winfo_width()) // 2
        y = root.winfo_rooty() + (root.winfo_height() - win.winfo_height()) // 2
        win.geometry("+{}+{}".format(max(x, 0), max(y, 0)))
        win.grab_set()
        win.focus_set()
        return win, bar

    def browse_dataset():
        selected_folder = filedialog.askdirectory(
            parent=root,
            title="Select a folder containing TXT documents",
            mustexist=True
        )
        if not selected_folder:
            return

        txt_files = list_txt_files(selected_folder)
        if not txt_files:
            messagebox.showerror(
                "No documents found",
                "The selected folder does not contain any .txt files.",
                parent=root
            )
            return

        selected_folder = os.path.abspath(selected_folder)
        output_folder = make_output_dir(selected_folder)
        existed_before = os.path.isdir(output_folder)

        progress, bar = show_progress_window(os.path.basename(selected_folder))
        state = {"error": None, "done": False}

        def worker():
            try:
                run_pipeline(selected_folder, output_folder)
            except BaseException as error:  # also catches SystemExit
                state["error"] = error
            finally:
                state["done"] = True

        def finish():
            bar.stop()
            progress.grab_release()
            progress.destroy()

            if state["error"] is None:
                try:
                    open_dataset(output_folder, selected_folder, False)
                    return
                except Exception as error:
                    state["error"] = error

            error = state["error"]
            remove_incomplete_output(output_folder, existed_before)
            messagebox.showerror(
                "Dataset Processing Error",
                "The selected folder could not be processed.\n\n"
                "Make sure it contains TXT files and that PySpark/Java are installed.\n\n"
                "Details: {}: {}".format(type(error).__name__, error),
                parent=root
            )

        def poll():
            if state["done"]:
                finish()
            else:
                root.after(150, poll)

        threading.Thread(target=worker, daemon=True).start()
        root.after(150, poll)

    # -----------------------------------------------------------------
    # Sidebar buttons
    # -----------------------------------------------------------------
    home_btn = menu_button(
        side, "Home", lambda: show("Home"), BTN_ACTIVE, BTN_HOVER
    )
    home_btn.pack(fill="x", padx=24, pady=5)
    buttons["Home"] = home_btn

    for name, build_page in menu:
        button = menu_button(
            side, name, lambda n=name: show(n), BTN, BTN_HOVER
        )
        button.pack(fill="x", padx=24, pady=5)
        buttons[name] = button

    menu_button(side, "Exit", root.destroy, RED, RED_HOVER).pack(
        side="bottom", fill="x", padx=24, pady=(6, 26)
    )
    menu_button(side, "Change Dataset", show_start, BTN, BTN_HOVER).pack(
        side="bottom", fill="x", padx=24, pady=6
    )

    # The application opens on the start (welcome) screen.
    start_page(
        start_holder,
        on_default=use_default_dataset,
        on_browse=browse_dataset
    ).pack(fill="both", expand=True)

    show_start()
    root.mainloop()


if __name__ == "__main__":
    main()