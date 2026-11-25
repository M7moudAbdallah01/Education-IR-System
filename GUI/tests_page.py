import re
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from helpers import BG, DARK, FONT, make_table, fill, title


def tests_page(parent, data):
    page = tk.Frame(parent, bg=BG)
    title(page, "Tests", "The supplied answer-key tests apply only to the original Default Dataset.")

    is_default = data.get("is_default", not data.get("dataset_path"))
    if not is_default:
        tk.Label(page, text="Answer-key tests are disabled for custom datasets because "
                            "their expected answers belong to the Default Dataset.",
                 bg=BG, fg="#A8432F", font=(FONT, 11, "bold"), wraplength=900,
                 justify="left").pack(anchor="w", padx=24, pady=20)
        return page

    project_root = Path(__file__).resolve().parent.parent
    summary = tk.Label(page, text="Press Run Tests to execute the project test suite.",
                       bg=BG, fg=DARK, font=(FONT, 11, "bold"), anchor="w")
    summary.pack(anchor="w", padx=24, pady=8)

    table_frame, tree = make_table(page, ("Status", "Test", "Details"), (90, 360, 650))
    table_frame.pack(fill="both", expand=True, padx=24, pady=10)

    run_button = tk.Button(page, text="Run Tests", bg="#8B5E34", fg="white",
                           activebackground="#A8743F", activeforeground="white", relief="flat",
                           font=(FONT, 10, "bold"), cursor="hand2")
    run_button.pack(anchor="w", padx=24)

    def parse_results(stdout):
        rows = []
        for line in stdout.splitlines():
            match = re.match(r"^(PASS|FAIL)\s{2}(.*)$", line)
            if not match:
                continue
            status, remainder = match.groups()
            test_name, separator, details = remainder.partition("   ")
            rows.append((status, test_name.strip(), details.strip() if separator else ""))
        return rows

    def finish_test_run(rows, return_code, stderr):
        fill(tree, rows)
        passed = sum(row[0] == "PASS" for row in rows)
        failed = sum(row[0] == "FAIL" for row in rows)
        if rows:
            summary.config(text=f"Passed: {passed}    Failed: {failed}",
                           fg="#A8432F" if failed else DARK)
        elif return_code == 0:
            summary.config(text="The test process finished, but no PASS/FAIL lines were found.",
                           fg=DARK)
        else:
            detail = stderr.strip().splitlines()
            message = detail[-1] if detail else f"Process exit code: {return_code}"
            summary.config(text=f"Test execution failed: {message}", fg="#A8432F")
        run_button.config(state="normal")

    def run_tests():
        run_button.config(state="disabled")
        summary.config(text="Running tests...", fg=DARK)
        fill(tree, [])

        def worker():
            try:
                process = subprocess.run([sys.executable, "test_results.py"], cwd=project_root,
                                         capture_output=True, text=True, encoding="utf-8",
                                         errors="replace", timeout=300, check=False)
                result = (parse_results(process.stdout), process.returncode, process.stderr)
            except subprocess.TimeoutExpired:
                result = ([], 1, "The test process exceeded the 5-minute time limit.")
            except Exception as error:
                result = ([], 1, str(error))
            page.after(0, lambda: finish_test_run(*result))

        threading.Thread(target=worker, daemon=True).start()

    run_button.config(command=run_tests)
    return page