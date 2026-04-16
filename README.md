# Does Language Typology Shape PP-Attachment Preferences?
### A Cross-Lingual Study Using PCFGs Trained from Scratch
**COMP 550 — Winter 2026**  
Vlad Rusu, Howe Wu, Makram Kerbage

---

## Research Question

Do PCFGs trained from scratch on typologically different languages (English, Japanese, Arabic) resolve PP-attachment ambiguity in systematically different ways, and do these differences correlate with word-order typology?

## Project Structure

```
├── data/
│   ├── raw/            # Original UD treebanks (English Penn UD, Japanese GSD, Arabic PADT)
│   └── processed/      # Binarized/cleaned trees ready for PCFG training
├── src/
│   ├── pcfg/           # Grammar extraction, MLE estimation, parent annotation, markovization
│   ├── parsing/        # Probabilistic CKY parser
│   ├── evaluation/     # PP-attachment classification and cross-lingual analysis
│   └── utils/          # Treebank loading, tree manipulation helpers
├── test_set/           # Manually constructed PP-ambiguous sentences per language
├── results/
│   ├── parses/         # Raw parse outputs
│   └── analysis/       # VP- vs NP-attachment rates, typology correlation
├── notebooks/          # Exploratory analysis and visualization
└── report/             # Final paper
```

## Pipeline

1. **Train PCFGs** — Extract rules from UD treebanks via MLE. Apply parent annotation + horizontal Markovization (Johnson, 1998).
2. **Build test set** — Construct PP-ambiguous sentences in each language where a PP can attach to either VP or NP.
3. **Parse & analyze** — Run probabilistic CKY on each test sentence. Compare VP- vs NP-attachment rates across languages; correlate with head-directionality and case-marking typology.

## Data Sources

| Language | Treebank | Word Order | Case Marking |
|----------|----------|------------|--------------|
| English  | Penn UD  | SVO        | No           |
| Japanese | GSD      | SOV        | Yes          |
| Arabic   | PADT     | VSO        | Yes          |

All treebanks are from [Universal Dependencies](https://universaldependencies.org/).

## Setup

```bash
pip install -r requirements.txt
```

## References

- Johnson, M. (1998). The effect of alternative tree representations on tree bank grammars.
- Jin, L., Oh, B.-D., & Schuler, W. (2021). Character-based PCFG Induction for Modeling the Syntactic Acquisition of Morphologically Rich Languages. EMNLP 2021.
- Schwartz, L., Aikawa, T., & Quirk, C. (2003). Disambiguation of English PP attachment using multilingual aligned data. MT Summit IX.
