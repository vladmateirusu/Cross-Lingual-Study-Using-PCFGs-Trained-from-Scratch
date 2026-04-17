"""
Step 3: Probabilistic CKY Parser
----------------------------------
Implements the probabilistic CKY (Cocke-Kasami-Younger) algorithm.
Given a sentence and a compiled PCFG (from Step 2), returns the
highest-probability parse tree.

After parsing, extracts the PP-attachment decision from the parse:
  - VP-attachment : the PP node's parent is a VP-type node
  - NP-attachment : the PP node's parent is an NP-type node

Usage (standalone):
    python step3_cky_parser.py

Or import parse_sentence() and get_pp_attachment() in step4/step5.
"""

import math
import pickle
import os
from step2_pcfg_training import Tree


# ---------------------------------------------------------------------------
# CKY chart cell
# ---------------------------------------------------------------------------

NEG_INF = float("-inf")


class ChartCell:
    __slots__ = ("log_prob", "left_child", "right_child", "split")

    def __init__(self):
        self.log_prob    = NEG_INF
        self.left_child  = None   # label of left  child constituent
        self.right_child = None   # label of right child constituent
        self.split       = None   # split position k


# ---------------------------------------------------------------------------
# CKY algorithm
# ---------------------------------------------------------------------------

def cky_parse(tokens, grammar):
    """
    Run probabilistic CKY on a list of tokens using a compiled PCFG.

    Returns the highest-probability parse tree as a Tree object,
    or None if no parse was found.

    Args:
        tokens  : list of str  (words in the sentence)
        grammar : PCFG object  (compiled, CNF)
    """
    n          = len(tokens)
    nonterminals = list(grammar.nonterminals)

    # chart[i][j][A] = ChartCell for span (i, j) and non-terminal A
    chart = [[{} for _ in range(n)] for _ in range(n)]

    # ---- Initialisation: span length 1 (preterminals) ----
    for i, word in enumerate(tokens):
        for (A, w), log_p in grammar.unary.items():
            if w == word:
                if A not in chart[i][i] or chart[i][i][A].log_prob < log_p:
                    cell           = ChartCell()
                    cell.log_prob  = log_p
                    chart[i][i][A] = cell

        # Handle unknown words: allow any preterminal with very low prob
        if not chart[i][i]:
            for (A, _), _ in grammar.unary.items():
                if A not in chart[i][i]:
                    cell           = ChartCell()
                    cell.log_prob  = math.log(1e-10)
                    chart[i][i][A] = cell

    # ---- Fill: increasing span lengths ----
    for span in range(2, n + 1):          # span length
        for i in range(n - span + 1):     # start position
            j = i + span - 1              # end position

            for k in range(i, j):         # split position
                left_cell  = chart[i][k]
                right_cell = chart[k + 1][j]

                for (A, B, C), log_p in grammar.binary.items():
                    if B not in left_cell or C not in right_cell:
                        continue

                    total = (log_p
                             + left_cell[B].log_prob
                             + right_cell[C].log_prob)

                    if A not in chart[i][j] or chart[i][j][A].log_prob < total:
                        cell             = ChartCell()
                        cell.log_prob    = total
                        cell.left_child  = B
                        cell.right_child = C
                        cell.split       = k
                        chart[i][j][A]   = cell

    # ---- Check if start symbol spans entire sentence ----
    start = grammar.start
    if start not in chart[0][n - 1]:
        # Fallback: pick highest-scoring non-terminal at root
        root_cell = max(
            chart[0][n - 1].items(),
            key=lambda kv: kv[1].log_prob,
            default=(None, None)
        )
        if root_cell[0] is None:
            return None, NEG_INF
        start = root_cell[0]

    root_log_prob = chart[0][n - 1][start].log_prob

    # ---- Backtrack to build the parse tree ----
    tree = _backtrack(chart, tokens, 0, n - 1, start)
    return tree, root_log_prob


def _backtrack(chart, tokens, i, j, label):
    """Recursively reconstruct the parse tree from the chart."""
    cell = chart[i][j].get(label)
    if cell is None:
        return Tree(label, [Tree(tokens[i])])  # fallback leaf

    if i == j:
        # Preterminal -> terminal
        return Tree(label, [Tree(tokens[i])])

    k = cell.split
    left  = _backtrack(chart, tokens, i,     k, cell.left_child)
    right = _backtrack(chart, tokens, k + 1, j, cell.right_child)
    return Tree(label, [left, right])


# ---------------------------------------------------------------------------
# PP-attachment extraction
# ---------------------------------------------------------------------------

def get_pp_attachment(tree):
    """
    Walk the parse tree and find the first PP node.
    Return 'VP' if its parent is a verbal node, 'NP' if nominal, else None.

    In a CNF/markovized tree, PP nodes may be nested inside intermediate
    nodes marked with '|' or '+'. We search for the first node whose
    base label (before '^' parent annotation) is 'PP'.
    """
    result = _find_pp_parent(tree, parent_label=None)
    return result


def _base_label(label):
    """Strip parent-annotation suffix: 'NP^VP' -> 'NP'."""
    return label.split("^")[0].split("+")[0].split("|")[0]


def _find_pp_parent(tree, parent_label):
    """
    DFS to find the first PP node and return its parent's base category.
    Returns 'VP', 'NP', or None.
    """
    if tree is None:
        return None
    base = _base_label(tree.label)

    if base == "PP" and parent_label is not None:
        parent_base = _base_label(parent_label)
        if parent_base in {"VP", "VBZ", "VBD", "VBP", "VBN", "VBG", "MD",
                           "VERB", "AUX", "S", "SBAR"}:
            return "VP"
        elif parent_base in {"NP", "NML", "NOUN", "PROPN", "PRON"}:
            return "NP"
        else:
            return parent_base   # return raw for inspection

    for child in tree.children:
        result = _find_pp_parent(child, tree.label)
        if result is not None:
            return result

    return None


# ---------------------------------------------------------------------------
# Probability of a specific attachment reading
# ---------------------------------------------------------------------------

def score_attachment(tokens, grammar, attachment_type):
    """
    Parse the sentence and return the log-probability of the parse
    that has the given PP-attachment type ('VP' or 'NP').

    Because a vanilla PCFG returns one best parse, we:
      1. Get the best parse and record its attachment.
      2. If it matches, return its log-prob.
      3. If it doesn't, return NEG_INF (the grammar strongly disfavors
         the other reading — we don't enumerate all parses here).

    For a more rigorous analysis, n-best parsing could be used.
    """
    tree, log_prob = cky_parse(tokens, grammar)
    if tree is None:
        return NEG_INF
    att = get_pp_attachment(tree)
    if att == attachment_type:
        return log_prob
    return NEG_INF


# ---------------------------------------------------------------------------
# Demo / standalone test
# ---------------------------------------------------------------------------

def parse_sentence(sentence, grammar):
    """
    Public interface: parse a sentence string and return
    (tree, log_prob, attachment_decision).
    """
    tokens = sentence.strip().split()
    tree, log_prob = cky_parse(tokens, grammar)
    if tree is None:
        return None, NEG_INF, None
    attachment = get_pp_attachment(tree)
    return tree, log_prob, attachment


def main():
    print("=" * 60)
    print("STEP 3: Probabilistic CKY Parser — Demo")
    print("=" * 60)

    model_dir = "models"
    test_sentences = {
        "English":  "I ate the sushi with chopsticks",
        "Japanese": "私は 箸で 寿司を 食べた",   # placeholder
        "Arabic":   "أكلت السوشي بالعيدان",      # placeholder
    }

    for lang, sentence in test_sentences.items():
        model_path = os.path.join(model_dir, f"{lang.lower()}_grammar.pkl")
        if not os.path.exists(model_path):
            print(f"\n[{lang}] Model not found at {model_path}. "
                  f"Run step2_pcfg_training.py first.")
            continue

        print(f"\n[{lang}] Loading grammar ...")
        with open(model_path, "rb") as f:
            grammar = pickle.load(f)

        print(f"[{lang}] Parsing: '{sentence}'")
        tree, log_prob, attachment = parse_sentence(sentence, grammar)

        if tree is None:
            print(f"  No parse found.")
        else:
            print(f"  Parse       : {tree}")
            print(f"  Log-prob    : {log_prob:.4f}")
            print(f"  PP-attachment: {attachment}")


if __name__ == "__main__":
    main()
