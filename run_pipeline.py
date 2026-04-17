"""
run_pipeline.py — Full Pipeline Runner
========================================
Runs all steps in order:
  1. Treebank PP-attachment analysis
  2. PCFG training
  3. CKY parsing + evaluation on ambiguous test set

Usage:
    python run_pipeline.py

Prerequisites:
  - UD treebank files in data/  (CoNLL-U format)
  - PTB constituency tree files in data/  (converted from UD)

See README.md for setup instructions.
"""

from step1_treebank_analysis import main as run_step1
from step2_pcfg_training     import main as run_step2
from step4_5_evaluation      import main as run_step4_5


def main():
    print("\n" + "#" * 60)
    print("# PP-ATTACHMENT CROSS-LINGUAL STUDY — FULL PIPELINE")
    print("#" * 60)

    print("\n>>> STEP 1: Treebank baseline analysis")
    baseline_results = run_step1()

    print("\n>>> STEP 2: PCFG training")
    grammars = run_step2()

    print("\n>>> STEP 3-5: CKY parsing + evaluation")
    records, summaries = run_step4_5()

    print("\n" + "#" * 60)
    print("# PIPELINE COMPLETE")
    print("#" * 60)
    print("\nOutputs:")
    print("  models/  — trained PCFG grammars (.pkl)")
    print("  results/ — detailed and summary results (.json, .csv)")


if __name__ == "__main__":
    main()
