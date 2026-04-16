# PP-Attachment Cross-Lingual Study — Code

COMP 550 Final Project, Winter 2026

## Project Structure

```
pp_project/
│
├── run_pipeline.py          # Master runner (runs all steps)
│
├── step1_treebank_analysis.py   # Step 1: UD treebank PP-attachment baseline
├── step2_pcfg_training.py       # Step 2: PCFG training (MLE + parent annot. + markov)
├── step3_cky_parser.py          # Step 3: Probabilistic CKY parser
├── step4_5_evaluation.py        # Step 4 & 5: Test set + evaluation
│
├── data/                    # Treebank files (you must download these)
│   ├── en_ewt-ud-train.conllu
│   ├── ja_gsd-ud-train.conllu
│   ├── ar_padt-ud-train.conllu
│   ├── en_trees_train.txt   # PTB constituency trees (converted from UD)
│   ├── ja_trees_train.txt
│   └── ar_trees_train.txt
│
├── models/                  # Saved PCFG grammars (auto-created)
│   ├── english_grammar.pkl
│   ├── japanese_grammar.pkl
│   └── arabic_grammar.pkl
│
└── results/                 # Evaluation output (auto-created)
    ├── results_detailed.json
    ├── results_summary.csv
    └── results_per_sentence.csv
```

## Setup

### 1. Install dependencies

```bash
pip install conllu nltk
```

### 2. Download UD Treebanks

Go to https://universaldependencies.org/ and download:
- **English**: `UD_English-EWT`  → `en_ewt-ud-train.conllu`
- **Japanese**: `UD_Japanese-GSD` → `ja_gsd-ud-train.conllu`
- **Arabic**: `UD_Arabic-PADT`   → `ar_padt-ud-train.conllu`

Place all `.conllu` files in the `data/` directory.

### 3. Convert UD to PTB constituency trees (for Step 2)

Step 2 (PCFG training) requires **constituency trees** in PTB bracket format.
UD treebanks use dependency format, so you must convert them.

**Option A — Stanford NLP toolkit** (recommended):
```bash
# Download: https://stanfordnlp.github.io/CoreNLP/
java -cp stanford-parser.jar \
  edu.stanford.nlp.trees.ud.UniversalDependenciesToConstituentConverter \
  data/en_ewt-ud-train.conllu > data/en_trees_train.txt
```

**Option B — udapi** (Python):
```bash
pip install udapi
udapy -s read.Conllu files=data/en_ewt-ud-train.conllu \
      ud.Convert2Pd write.Treex > tmp.treex
# Then convert .treex to PTB format using a script
```

**Option C** — If you already have Penn Treebank (for English), use it directly
and skip conversion. Place PTB-format trees (one per line) in `data/en_trees_train.txt`.

### 4. Run the full pipeline

```bash
python run_pipeline.py
```

Or run individual steps:

```bash
python step1_treebank_analysis.py   # Baseline attachment rates
python step2_pcfg_training.py       # Train grammars
python step4_5_evaluation.py        # Evaluate on test set
```

## What each step does

| Step | File | Description |
|------|------|-------------|
| 1 | `step1_treebank_analysis.py` | Counts VP vs NP attachment in raw UD treebanks |
| 2 | `step2_pcfg_training.py` | Trains PCFGs with MLE + parent annotation + markovization + CNF |
| 3 | `step3_cky_parser.py` | Probabilistic CKY parser; extracts PP-attachment from parse tree |
| 4+5 | `step4_5_evaluation.py` | Runs parser on 30 ambiguous sentences/language; reports attachment rates and margins |

## Notes on the test sentences (Step 4)

The 30 English sentences are hand-crafted following the paper's methodology.
The Japanese and Arabic sentences are **romanized/transliterated placeholders**
and must be replaced with properly tokenized native-script sentences verified
by a native speaker or sourced from naturally occurring treebank examples.

When replacing, make sure each sentence:
1. Follows [Subject] [Verb] [Object NP] [PP] structure
2. Has a PP semantically compatible with BOTH the verb and the noun
3. Has been verified by a native/near-native speaker

## Output

After running the pipeline, results are in `results/`:

- `results_summary.csv` — VP%, NP%, avg margin per language (Table 2 in paper)
- `results_per_sentence.csv` — attachment decision for each sentence
- `results_detailed.json` — full output with log-probs and glosses
