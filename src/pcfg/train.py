"""Top-level PCFG training script: load treebank -> annotate -> estimate MLE -> save grammar."""

import argparse
import pickle
from pathlib import Path

from src.utils.treebank import load_conllu, conllu_to_nltk_tree
from src.pcfg.annotations import parent_annotate, markovize
from src.pcfg.grammar import estimate_mle, build_pcfg


def train(conllu_path: str, output_path: str, v: int = 1, h: int = 2):
    print(f"Loading treebank: {conllu_path}")
    token_lists = load_conllu(conllu_path)

    trees = []
    for tl in token_lists:
        tree = conllu_to_nltk_tree(tl)
        if tree is None:
            continue
        if v > 1:
            tree = parent_annotate(tree)
        tree = markovize(tree, h=h)
        trees.append(tree)

    print(f"  Extracted {len(trees)} trees")
    prob_rules = estimate_mle(trees)
    grammar = build_pcfg(prob_rules)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(grammar, f)
    print(f"  Grammar saved to {output_path} ({len(prob_rules)} rules)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("conllu", help="Path to .conllu treebank file")
    parser.add_argument("output", help="Path to save pickled PCFG")
    parser.add_argument("--v", type=int, default=1, help="Vertical markov order (1=parent annotation)")
    parser.add_argument("--h", type=int, default=2, help="Horizontal markov order")
    args = parser.parse_args()
    train(args.conllu, args.output, v=args.v, h=args.h)
