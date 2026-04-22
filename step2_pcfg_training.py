#Step 2: PCFG Training

import os
import re
import pickle
from collections import defaultdict
import math
import conllu


#configuration
TREE_FILES = {
    "English":  "data/en_ewt-ud-train.conllu",
    "Japanese": "data/ja_gsd-ud-train.conllu",
    "Arabic":   "data/ar_padt-ud-train.conllu",
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


def _has_case_child(token_id, index):
    """Return True if any token in the sentence has deprel='case' and head=token_id."""
    for tok in index.values():
        if tok["deprel"] == "case" and tok["head"] == token_id:
            return True
    return False
 
 
# UD deprel → phrase label mapping (module-level constant)
_DEPREL_TO_PHRASE = {
    "nsubj": "NP",  "obj":   "NP",  "iobj":  "NP",
    "nmod":  "NP",  "appos": "NP",  "det":   "DT",
    "amod":  "ADJP","advmod":"ADVP","aux":   "AUX",
    "cop":   "COP", "mark":  "SBAR","advcl": "SBAR",
    "acl":   "S",   "xcomp": "VP",  "ccomp": "S",
    "conj":  "CONJ","cc":    "CC",  "punct": "PUNCT",
    "case":  "CASE","obl":   "PP",
}

 
# Choose a phrase label for a token's subtree.
#  - Tokens with deprel obl/nmod that have a case child → PP
# - Otherwise use the deprel as the label (VP, NP, etc. are rare in UD; most will be things like 'nsubj', 'obj', 'obl', etc.)
# - Root tokens → S
def _phrase_label(token, index):
    deprel = token["deprel"] or "X"
    if deprel == "root":
        return "S"
    if deprel in ("obl", "nmod") and _has_case_child(token["id"], index):
        return "PP"
    return _DEPREL_TO_PHRASE.get(deprel, deprel.upper())
 
 #Recursively build a Tree rooted at token_id
def _build_subtree(token_id, index, children_map):
    token = index[token_id]
    word  = token["form"] or "_"
    upos  = token["upos"] or "X"
 
    # Preterminal leaf
    preterminal = Tree(upos, [Tree(word)])
 
    # Recursively build children subtrees, sorted by position
    dep_subtrees = [
        _build_subtree(child_id, index, children_map)
        for child_id in sorted(children_map.get(token_id, []))
    ]
 
    phrase_label = _phrase_label(token, index)
 
    if not dep_subtrees:
        # No dependents → just return the preterminal wrapped in its phrase
        return Tree(phrase_label, [preterminal])
    else:
        return Tree(phrase_label, [preterminal] + dep_subtrees)
 
#Convert a single parsed CoNLL-U sentence to a constituency Tree.
def ud_sentence_to_tree(sentence):
    # Build index and children map (skip multi-word tokens)
    index        = {tok["id"]: tok for tok in sentence
                    if isinstance(tok["id"], int)}
    children_map = defaultdict(list)
    root_id      = None
 
    for tok in index.values():
        head = tok["head"]
        if head == 0:
            root_id = tok["id"]
        else:
            children_map[head].append(tok["id"])
 
    if root_id is None:
        return None
 
    return _build_subtree(root_id, index, children_map)
 
 

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



def _int_defaultdict():
    return defaultdict(int)

#Rule extraction and MLE 
class PCFG:

    def __init__(self):
        self._rule_counts = defaultdict(_int_defaultdict)
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
 
        # Unary closure: compute total probability of reaching C from A
        # through any chain of NT->NT steps.  Uses a forward-index so each
        # iteration only touches reachable pairs
        fwd = defaultdict(list)
        for (b, c), p in raw_unary.items():
            fwd[b].append((c, p))

        closed = dict(raw_unary)   # (A, B) -> prob

        changed = True
        while changed:
            changed = False
            additions = {}
            for (a, b), p_ab in closed.items():
                for c, p_bc in fwd.get(b, []):
                    key     = (a, c)
                    new_val = p_ab * p_bc
                    cur     = max(closed.get(key, 0.0), additions.get(key, 0.0))
                    if new_val > cur + 1e-15:
                        additions[key] = new_val
            if additions:
                closed.update(additions)
                changed = True

        # Store closed unary rules as log-probs
        self.unary = {k: math.log(v) for k, v in closed.items() if v > 0}
 
        print(f"  Grammar compiled: "
              f"{len(self.lexical)} lexical rules, "
              f"{len(self.unary)} unary NT rules (after closure), "
              f"{len(self.binary)} binary rules, "
              f"{len(self.nonterminals)} non-terminals.")


def load_trees(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Treebank file not found: {path}")
    with open(path, encoding="utf-8") as f:
        sentences = conllu.parse(f.read())
    trees = [ud_sentence_to_tree(s) for s in sentences]
    return [t for t in trees if t is not None]


#training pipeline
#Load trees, apply transformations, extract PCFG
def train_grammar(tree_path, lang, markov_order=MARKOV_ORDER):
    trees = load_trees(tree_path)
    print(f"  Loaded {len(trees)} trees.")

    grammar = PCFG()

    for tree in trees:
        annotate_parents(tree)
        markovize(tree, order=markov_order)
        to_cnf(tree)
        grammar.count_rules(tree, is_root=True)

    grammar.compile()

    return grammar


def main():
    print("STEP 2: PCFG Training")

    grammars = {}

    for lang, path in TREE_FILES.items():
        print(f"\nTraining {lang} PCFG ")
        try:
            grammar = train_grammar(path, lang)
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