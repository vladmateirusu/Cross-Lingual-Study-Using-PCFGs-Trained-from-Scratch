from step1_treebank_analysis import main as run_step1
from step2_pcfg_training     import main as run_step2
from step4_5_evaluation      import main as run_step4_5


def main():
    print("Full pipleine")

    print("\n>>> Step 1: Treebank baseline analysis")
    baseline_results = run_step1()

    print("\n>>> Step 2: PCFG training")
    grammars = run_step2()

    print("\n>>> Step 3-5: CKY parsing + evaluation")
    # Pass baseline_results so Table 2 includes both columns side-by-side
    records, summaries = run_step4_5(baseline_results=baseline_results)

    print("# Pipline completed")
    print("\nOutputs written to:")
    print("  models/results_summary.csv          — VP%/NP% per language")
    print("  results/results_per_sentence.csv    — per-sentence attachment + margin")
    print("  results/results_detailed.json       — full output with scores")
    print("  results/table2_comparison.csv       — Table 2: baseline vs. parser")


if __name__ == "__main__":
    main()
