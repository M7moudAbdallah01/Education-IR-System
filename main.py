#Member 6: runs everything in order

# ---------------------------------------------------------
# Main Integration Program (Member 6)
# Runs Part A (Positional Index with Spark), followed by
# every step of Part B in exact logical order.
# ---------------------------------------------------------

import os
import sys

from partA.positional_index_spark import main as run_part_a
from partB import tfidf_normalization
from partB.phrase_boolean import run_phrase_boolean_queries
from partB.similarity_ranking import run_all as run_similarity_ranking

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
INDEX_FILE = os.path.join(PROJECT_ROOT, "output", "positional_index.json")


def main():
    print("=========================================================")
    print("   EDUCATION INFORMATION RETRIEVAL SYSTEM - PIPELINE")
    print("=========================================================")

    # -----------------------------------------------------
    # Step 1: Part A - Build & Save Positional Index (Spark)
    # -----------------------------------------------------
    print("\n>>> STEP 1: Running Part A (Positional Index via Spark)...")
    try:
        run_part_a()
    except Exception as exc:
        if os.path.exists(INDEX_FILE):
            print(f"Spark skipped ({type(exc).__name__}); using existing positional_index.json.")
        else:
            raise

    # -----------------------------------------------------
    # Step 2: Part B (Member 2) - Raw TF, Weighted TF, DF & IDF
    # -----------------------------------------------------
    print("\n>>> STEP 2: Running Part B - Member 2 (TF, Weighted TF, DF & IDF)...")
    from partB import tf_df_idf  # Executes Member 2's calculations & saves CSVs

    # -----------------------------------------------------
    # Step 3: Part B (Member 3) - TF-IDF & Normalization
    # -----------------------------------------------------
    print("\n>>> STEP 3: Running Part B - Member 3 (TF-IDF & Normalization)...")
    tfidf_normalization.run_all(verbose=True, save=True)

    # -----------------------------------------------------
    # Step 4: Part B (Member 4) - Phrase & Boolean Queries
    # -----------------------------------------------------
    print("\n>>> STEP 4: Running Part B - Member 4 (Phrase & Boolean Queries)...")
    run_phrase_boolean_queries(verbose=True)

    # -----------------------------------------------------
    # Step 5: Part B (Member 5) - Query Vectors, Similarity & Ranking
    # -----------------------------------------------------
    print("\n>>> STEP 5: Running Part B - Member 5 (Cosine Similarity & Ranking)...")
    run_similarity_ranking(verbose=True, save=True, use_csv=True)

    print("\n=========================================================")
    print("   ALL PIPELINE STEPS COMPLETED SUCCESSFULLY!")
    print("=========================================================")


if __name__ == "__main__":
    main()