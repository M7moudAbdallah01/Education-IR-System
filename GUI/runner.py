"""
Pipeline runner used by the GUI (no Tkinter code here).

When the user imports a new dataset the GUI calls run_pipeline(), which calls
main.main(dataset_dir, output_dir). main.py then runs every step in order
(Spark positional index -> TF/DF/IDF -> TF-IDF & normalization -> phrase &
Boolean queries -> similarity & ranking) and saves all results inside
output_dir, a new sub-folder of the project's "output" folder.
"""
import hashlib
import importlib
import os
import shutil
import sys

from helpers import PROJECT_ROOT, DEFAULT_OUTPUT_DIR


def list_txt_files(folder):
    """Return the .txt files found directly inside the selected folder."""
    try:
        return sorted(
            name for name in os.listdir(folder)
            if name.lower().endswith(".txt")
            and os.path.isfile(os.path.join(folder, name))
        )
    except OSError:
        return []


def make_output_dir(dataset_dir):
    """
    Build the path of the results folder for a dataset:
        output/<dataset folder name>_<short hash of its full path>
    The hash keeps two different datasets with the same folder name apart.
    """
    dataset_dir = os.path.abspath(dataset_dir)
    name = os.path.basename(os.path.normpath(dataset_dir)) or "dataset"
    safe_name = "".join(c if (c.isalnum() or c in "-_") else "_" for c in name)
    short_hash = hashlib.sha1(dataset_dir.encode("utf-8")).hexdigest()[:6]
    return os.path.join(DEFAULT_OUTPUT_DIR, "{}_{}".format(safe_name, short_hash))


def _fresh_import_main():
    """
    Import main.py again from scratch. Some pipeline modules calculate their
    results when they are imported, so cached modules from a previous run
    would silently reuse the old dataset.
    """
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    for name in list(sys.modules):
        if name == "main" or name.split(".")[0] in ("partA", "partB"):
            del sys.modules[name]

    return importlib.import_module("main")


def run_pipeline(dataset_dir, output_dir):
    """
    Run the whole project pipeline on dataset_dir and save the results in
    output_dir. Raises an exception with a readable message on failure.
    """
    if not list_txt_files(dataset_dir):
        raise ValueError("The selected folder does not contain any .txt files.")

    os.makedirs(output_dir, exist_ok=True)

    # pythonw / packaged apps have no console: keep print() from crashing.
    devnull = None
    if sys.stdout is None or sys.stderr is None:
        devnull = open(os.devnull, "w", encoding="utf-8")
        sys.stdout = sys.stdout or devnull
        sys.stderr = sys.stderr or devnull

    try:
        pipeline = _fresh_import_main()
        pipeline.main(dataset_dir=dataset_dir, output_dir=output_dir)
    except SystemExit as exit_signal:
        # Some modules stop the program with sys.exit("message").
        raise RuntimeError(str(exit_signal.code or "The pipeline stopped unexpectedly."))
    finally:
        if devnull is not None:
            devnull.close()


def remove_incomplete_output(output_dir, existed_before):
    """Delete a results folder that this run created but could not finish."""
    if not existed_before:
        shutil.rmtree(output_dir, ignore_errors=True)
