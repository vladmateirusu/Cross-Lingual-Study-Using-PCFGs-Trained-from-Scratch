#Step 2: PCFG Training

import os
import re
import pickle
from collections import defaultdict
import math
import copy


#configuration
TREE_FILES = {
    "English":  "data/en_trees_train.txt",
    "Japanese": "data/ja_trees_train.txt",
    "Arabic":   "data/ar_trees_train.txt",
}

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

MARKOV_ORDER = 2   # horizontal Markov order (bigram = 2)


#tree data structure
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


#PTB-format tree parser
def parse_ptb(text):
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
            pos[0] += 1  
            return Tree(label, children)
        else:
            return Tree(tok)

    tree = _parse()

    if tree.label in ("ROOT", "TOP") and len(tree.children) == 1:
        tree = tree.children[0]
    return tree


def load_trees(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"file not found: {path}\n"
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


#the following three functions are to simply imrpove the grammar, annotate_parents, markovize and binarize_right
#Parent annotation, from Johnson 1998 It helps your PCFG learn better rules:
def annotate_parents(tree, parent_label=None):
    original_label = tree.label   # save before mutating
 
    if parent_label is not None and not tree.is_leaf():
        tree.label = f"{tree.label}^{parent_label}"
 
    for child in tree.children:
        if not child.is_leaf():
            annotate_parents(child, original_label)   # original, not annotated
 
    return tree

#Horizontal markovization 
#decompose rules with >2 children into binary rules using a right-branching binarization with Markov-order history markers.
def markovize(tree, order=2):
    if tree.is_leaf():
        return tree

    # Recurse first
    for child in tree.children:
        markovize(child, order)

    # Binarize if needed
    if len(tree.children) > 2:
        _binarize_right(tree, order)

    return tree

#Right-binarize a node with >2 children
def _binarize_right(tree, order):
    children = tree.children

    # Build right-branching chain from right to left
    history = [c.label for c in children[-(order - 1):]] if order > 1 else []

    # Start from second-to-last pair and work leftward
    right = children[-1]
    for i in range(len(children) - 2, 0, -1):
        left   = children[i]
        marker = f"{tree.label}|<{'-'.join(history[:order-1])}>"
        right  = Tree(marker, [left, right])
        history = [left.label] + history[: order - 2]

    tree.children = [children[0], right]


#CNF conversion, Chomsky Normal Form
def to_cnf(tree):
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


#Rule extraction and MLE 
class PCFG:

    def __init__(self):
        self._rule_counts = defaultdict(lambda: defaultdict(int))
        self._lhs_counts  = defaultdict(int)
        self._root_counts = defaultdict(int)   # actual root labels per tree
 
        self.lexical      = {}   # {(A, word) -> log_prob}
        self.unary        = {}   # {(A, B)    -> log_prob}  NT -> NT
        self.binary       = {}   # {(A, B, C) -> log_prob}
        self.nonterminals = set()
        self.terminals    = set()
        self.start        = None

#extract and count all rules from a tree
    def count_rules(self, tree, is_root=False):
        if tree.is_leaf():
            self.terminals.add(tree.label)
            return
 
        lhs = tree.label
        rhs = tuple(c.label for c in tree.children)
        self._rule_counts[lhs][rhs] += 1
        self._lhs_counts[lhs]       += 1
        self.nonterminals.add(lhs)
 
        if is_root:
            self._root_counts[lhs] += 1
 
        for child in tree.children:
            self.count_rules(child, is_root=False)

#Convert raw counts to log-probabilities
#Separates rules into lexical, unary NT, and binary tables
#Start symbol is inferred from observed root labels, not overall LHS freq
    def compile(self):
        if self._root_counts:
            self.start = max(self._root_counts, key=self._root_counts.get)
        else:
            self.start = "S"
        print(f"  Start symbol: {self.start}")
 
        # Raw unary NT rules before closure 
        raw_unary = {}   # {(A, B) -> prob}
 
        for lhs, rhs_counts in self._rule_counts.items():
            total = self._lhs_counts[lhs]
            for rhs, cnt in rhs_counts.items():
                prob  = cnt / total
                log_p = math.log(prob)
 
                if len(rhs) == 1:
                    child = rhs[0]
                    if child in self.nonterminals:
                        raw_unary[(lhs, child)] = prob
                    else:
                        self.lexical[(lhs, child)] = log_p
 
                elif len(rhs) == 2:
                    self.binary[(lhs, rhs[0], rhs[1])] = log_p
 
                else:
                    print(f"   Non-binary rule after CNF: {lhs} -> {rhs}")
 
        # Unary closure: compute the total probability of reaching B from A
        # through any chain of unary NT steps A->X->...->B.
        closed = dict(raw_unary) 
 
        nts = list(self.nonterminals)
        for _ in range(len(nts)):   
            new_closed = dict(closed)
            for (a, b), p_ab in closed.items():
                for (b2, c), p_bc in raw_unary.items():
                    if b2 == b:
                        key = (a, c)
                        new_closed[key] = new_closed.get(key, 0.0) + p_ab * p_bc
            if new_closed == closed:
                break
            closed = new_closed
 
        # Store closed unary rules as log-probs
        self.unary = {k: math.log(v) for k, v in closed.items() if v > 0}
 
        print(f"  Grammar compiled: "
              f"{len(self.lexical)} lexical rules, "
              f"{len(self.unary)} unary NT rules (after closure), "
              f"{len(self.binary)} binary rules, "
              f"{len(self.nonterminals)} non-terminals.")


#training pipeline
#Load trees, apply transformations, extract PCFG
def train_grammar(tree_path, lang, markov_order=2):
    trees = load_trees(tree_path)
    print(f"  Loaded {len(trees)} trees.")

    grammar = PCFG()

    for tree in trees:
        # Deep-copy so original trees are never mutated
        t = copy.deepcopy(tree)
        annotate_parents(t, parent_label=None)
        markovize(t, order=markov_order)
        to_cnf(t)
        grammar.count_rules(t, is_root=True)

    grammar.compile()

    return grammar


def main():
    print("STEP 2: PCFG Training")

    grammars = {}

    for lang, path in TREE_FILES.items():
        print(f"\nTraining {lang} PCFG ")
        try:
            grammar = train_grammar(path, lang, markov_order=MARKOV_ORDER)
            grammars[lang] = grammar

            # Save to disk
            out_path = os.path.join(MODEL_DIR, f"{lang.lower()}_grammar.pkl")
            with open(out_path, "wb") as f:
                pickle.dump(grammar, f)
            print(f"  Saved to {out_path}")

        except FileNotFoundError as e:
            print(f"error: {e}")

    return grammars


if __name__ == "__main__":
    main()
