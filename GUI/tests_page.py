import os
import re
import subprocess
import sys
import tkinter as tk

from helpers import BG, DARK, FONT, make_table, title
from queries import button

# project root (the folder that contains test_results.py)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GREEN, RED_TEXT = "#2E7D32", "#C62828"

# test_results.py prints lines like:  "PASS  name"  or  "FAIL  name   got ..., expected ..."
LINE = re.compile(r"^(PASS|FAIL)\s{2}(.*)$")


def run_test_script():
    """Run test_results.py exactly as it is and read what it prints.
    Returns a list of (status, test name, details)."""
    proc = subprocess.run([sys.executable, "test_results.py"], cwd=ROOT,
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    rows = []
    for line in proc.stdout.splitlines():
        m = LINE.match(line)
        if not m:
            continue
        status, rest = m.groups()
        name, _, details = rest.partition("   ")
        rows.append((status, name.strip(), details.strip()))
    return rows, proc


def tests_page(parent, data):
    page = tk.Frame(parent, bg=BG)
    title(page, "Tests", "Compares the system output with the answer key (PDF).")

    bar = tk.Frame(page, bg=BG)
    bar.pack(fill="x", padx=24, pady=(0, 8))
    summary = tk.Label(bar, text="Press 'Run Tests'", bg=BG, fg=DARK, font=(FONT, 11, "bold"))
    only_failed = tk.BooleanVar(value=False)
    state = {"rows": []}

    frame, tree = make_table(page, ("Status", "Test", "Details"), (80, 380, 600))
    tree.tag_configure("pass", foreground=GREEN)
    tree.tag_configure("fail", foreground=RED_TEXT)
    frame.pack(fill="both", expand=True, padx=24, pady=(0, 20))

    def show():
        tree.delete(*tree.get_children())
        for status, name, details in state["rows"]:
            if only_failed.get() and status == "PASS":
                continue
            tree.insert("", "end", values=(status, name, details),
                        tags=("pass" if status == "PASS" else "fail",))

    def run():
        summary.config(text="Running...", fg=DARK)
        page.update_idletasks()
        rows, proc = run_test_script()
        state["rows"] = rows
        if not rows:       # script crashed before printing any result
            summary.config(text="Could not run the tests (see details)", fg=RED_TEXT)
            tree.delete(*tree.get_children())
            tree.insert("", "end", values=("FAIL", "test_results.py", proc.stderr.strip()[-300:]),
                        tags=("fail",))
            return
        failed = sum(1 for r in rows if r[0] == "FAIL")
        summary.config(text="Passed: {}    Failed: {}".format(len(rows) - failed, failed),
                       fg=GREEN if failed == 0 else RED_TEXT)
        show()

    button(bar, "Run Tests", run).pack(side="left")
    tk.Checkbutton(bar, text="Show failed only", variable=only_failed, command=show,
                   bg=BG, fg=DARK, activebackground=BG, font=(FONT, 10)).pack(side="left", padx=14)
    summary.pack(side="left", padx=10)
    return page