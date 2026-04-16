"""
Step 2: PCFG Training
----------------------
Trains a separate PCFG for each language using:
  - Maximum Likelihood Estimation (MLE)
  - Parent annotation  (Johnson 1998)
  - Horizontal markovization (bigram decomposition of RHS)
  - Binarization + CNF conversion (required for CKY)

Input : Constituency trees converted from UD treebanks using the Stanford
        converter. Trees should be in PTB bracketed format, one tree per line:
          data/en_trees_train.txt
          data/ja_trees_train.txt
          data/ar_trees_train.txt

Output: Pickled PCFG grammar objects saved to:
          models/en_grammar.pkl
          models/ja_grammar.pkl
          models/ar_grammar.pkl

Note on UD -> constituency conversion:
  Use the Stanford NLP toolkit converter:
    java -cp stanford-parser.jar edu.stanford.nlp.trees.ud.UniversalDependenciesToConstituentConverter
  Or use the 'udapi' Python library:
    pip install udapi
    udapy write.Treex ud.Convert < input.conllu | treex write.Penn > output.txt
"""

import os
import re
import pickle
from collections import defaultdict


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TREE_FILES = {
    "English":  "data/en_trees_train.txt",
    "Japanese": "data/ja_trees_train.txt",
    "Arabic":   "data/ar_trees_train.txt",
}

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

MARKOV_ORDER = 2   # horizontal Markov order (bigram = 2)


# ---------------------------------------------------------------------------
# Tree data structure
# ---------------------------------------------------------------------------

class Tree:
    """Simple recursive tree structure."""

    def __init__(self, label, children=None):
        self.label    = label
        self.children = children if children is not None else []

    def is_leaf(self):
        return len(self.children) == 0

    def is_preterminal(self):
        return (len(self.children) == 1 and self.children[0].is_leaf())

    def __repr__(self):
        if self.is_leaf():
            return self.label
        return f"({self.label} {' '.join(repr(c) for c in self.children)})"


# ---------------------------------------------------------------------------
# PTB-format tree parser
# ---------------------------------------------------------------------------

def parse_ptb(text):
    """
    Parse a PTB bracketed string into a Tree.
    E.g.: (S (NP (DT The) (NN cat)) (VP (VBZ sits)))
    """
    tokens = re.findall(r'\(|\)|[^\s()]+', text)
    pos    = [0]

    def _parse():
        tok = tokens[pos[0]]
        pos[0] += 1

        if tok == '(':
            label    = tokens[pos[0]]; pos[0] += 1
            children = []
            while tokens[pos[0]] != ')':
                children.append(_parse())
            pos[0] += 1  # consume ')'
            return Tree(label, children)
        else:
            # bare terminal
            return Tree(tok)

    # Handle top-level wrapping like (ROOT ...)
    tree = _parse()
    # Unwrap ROOT if present
    if tree.label in ("ROOT", "TOP") and len(tree.children) == 1:
        tree = tree.children[0]
    return tree


def load_trees(path):
    """Load all trees from a PTB-format file (one tree per line)."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Tree file not found: {path}\n"
            f"Please convert UD treebank to PTB constituency format first."
        )
    trees = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                trees.append(parse_ptb(line))
            except (IndexError, KeyError):
                continue   # skip malformed trees
    return trees


# ---------------------------------------------------------------------------
# Parent annotation  (Johnson 1998)
# ---------------------------------------------------------------------------

def annotate_parents(tree, parent_label=None):
    """
    Augment each non-terminal with its parent:
      NP under VP becomes NP^VP
    Modifies tree in-place and returns it.
    """
    if parent_label is not None and not tree.is_leaf():
        tree.label = f"{tree.label}^{parent_label}"

    for child in tree.children:
        if not child.is_leaf():
            annotate_parents(child, tree.label)

    return tree


# ---------------------------------------------------------------------------
# Horizontal markovization  (Johnson 1998)
# ---------------------------------------------------------------------------

def markovize(tree, order=2):
    """
    Decompose rules with >2 children into binary rules using a
    right-branching binarization with Markov-order history markers.

    E.g. (NP A B C D) with order=2 becomes:
      (NP A (NP|<A-B> B (NP|<B-C> C D)))

    Modifies tree in-place and returns it.
    """
    if tree.is_leaf():
        return tree

    # Recurse first
    for child in tree.children:
        markovize(child, order)

    # Binarize if needed
    if len(tree.children) > 2:
        _binarize_right(tree, order)

    return tree


def _binarize_right(tree, order):
    """Right-binarize a node with >2 children."""
    children = tree.children

    # Build right-branching chain from right to left
    # history = last `order-1` sibling labels
    history = [c.label for c in children[-(order - 1):]] if order > 1 else []

    # Start from second-to-last pair and work leftward
    right = children[-1]
    for i in range(len(children) - 2, 0, -1):
        left   = children[i]
        marker = f"{tree.label}|<{'-'.join(history[:order-1])}>"
        right  = Tree(marker, [left, right])
        # Shift history window
        history = [left.label] + history[: order - 2]

    tree.children = [children[0], right]


# ---------------------------------------------------------------------------
# CNF conversion
# ---------------------------------------------------------------------------

def to_cnf(tree):
    """
    Convert a tree to Chomsky Normal Form (CNF):
      - Remove unary chains (except preterminals -> terminals)
      - Binarize any remaining nodes with >2 children
    Modifies tree in-place and returns it.
    """
    if tree.is_leaf():
        return tree

    # Collapse unary non-terminal chains (not preterminals)
    while (len(tree.children) == 1
           and not tree.children[0].is_leaf()
           and not tree.children[0].is_preterminal()):
        child        = tree.children[0]
        tree.label   = f"{tree.label}+{child.label}"
        tree.children = child.children

    # Recurse
    tree.children = [to_cnf(c) for c in tree.children]

    # Binarize (should be rare after markovization)
    if len(tree.children) > 2:
        _binarize_right(tree, order=2)

    return tree


# ---------------------------------------------------------------------------
# Rule extraction and MLE
# ---------------------------------------------------------------------------

class PCFG:
    """
    Probabilistic Context-Free Grammar.
    Stores rules as:
      unary_rules  : {A -> w}  (preterminal -> terminal)
      binary_rules : {A -> B C}
    Probabilities computed via MLE.
    """

    def __init__(self):
        # Raw counts
        self._rule_counts = defaultdict(lambda: defaultdict(int))
        self._lhs_counts  = defaultdict(int)

        # Compiled probability tables (filled by compile())
        self.unary  = {}   # {(A, w)      -> log_prob}
        self.binary = {}   # {(A, B, C)   -> log_prob}
        self.nonterminals = set()
        self.start = None

    # ---- Training ----

    def count_rules(self, tree):
        """Recursively extract and count all rules from a tree."""
        if tree.is_leaf():
            return

        lhs  = tree.label
        rhs  = tuple(c.label for c in tree.children)
        self._rule_counts[lhs][rhs] += 1
        self._lhs_counts[lhs]       += 1
        self.nonterminals.add(lhs)

        for child in tree.children:
            self.count_rules(child)

    def compile(self, start="S"):
        """
        Convert raw counts to log-probabilities and separate into
        unary (preterminal) and binary rule tables.
        """
        import math
        self.start = start

        for lhs, rhs_counts in self._rule_counts.items():
            total = self._lhs_counts[lhs]
            for rhs, cnt in rhs_counts.items():
                log_p = math.log(cnt / total)
                if len(rhs) == 1:
                    # Unary: preterminal -> terminal
                    self.unary[(lhs, rhs[0])] = log_p
                elif len(rhs) == 2:
                    self.binary[(lhs, rhs[0], rhs[1])] = log_p
                else:
                    # Should not happen after CNF conversion
                    print(f"  [WARNING] Non-binary rule kept: {lhs} -> {rhs}")

        print(f"  Grammar compiled: {len(self.unary)} unary rules, "
              f"{len(self.binary)} binary rules, "
              f"{len(self.nonterminals)} non-terminals.")


# ---------------------------------------------------------------------------
# Full training pipeline
# ---------------------------------------------------------------------------

def train_grammar(tree_path, lang, markov_order=2):
    """Load trees, apply transformations, extract PCFG."""
    print(f"\n  Loading trees from {tree_path} ...")
    trees = load_trees(tree_path)
    print(f"  Loaded {len(trees)} trees.")

    grammar = PCFG()

    for tree in trees:
        # 1. Parent annotation
        annotate_parents(tree, parent_label=None)
        # 2. Horizontal markovization
        markovize(tree, order=markov_order)
        # 3. CNF conversion
        to_cnf(tree)
        # 4. Count rules
        grammar.count_rules(tree)

    # Determine start symbol (most common root label)
    start = max(grammar._lhs_counts, key=grammar._lhs_counts.get)
    print(f"  Inferred start symbol: {start}")
    grammar.compile(start=start)

    return grammar


def main():
    print("=" * 60)
    print("STEP 2: PCFG Training")
    print("=" * 60)

    grammars = {}

    for lang, path in TREE_FILES.items():
        print(f"\n--- Training {lang} PCFG ---")
        try:
            grammar = train_grammar(path, lang, markov_order=MARKOV_ORDER)
            grammars[lang] = grammar

            # Save to disk
            out_path = os.path.join(MODEL_DIR, f"{lang.lower()}_grammar.pkl")
            with open(out_path, "wb") as f:
                pickle.dump(grammar, f)
            print(f"  Saved to {out_path}")

        except FileNotFoundError as e:
            print(f"  [SKIPPED] {e}")

    return grammars


if __name__ == "__main__":
    main()
