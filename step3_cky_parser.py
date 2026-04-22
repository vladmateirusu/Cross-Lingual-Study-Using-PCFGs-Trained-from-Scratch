#Step 3: Probabilistic CKY Parser

import math
import pickle
import os
from collections import defaultdict
from step2_pcfg_training import Tree


# CKY chart cell
NEG_INF = float("-inf")


class ChartCell:
    __slots__ = ("log_prob", "left_child", "right_child", "split", "is_unary")

    def __init__(self):
        self.log_prob    = NEG_INF
        self.left_child  = None
        self.right_child = None
        self.split       = None
        self.is_unary    = False   # True when this cell was filled by a NT->NT rule


# Pre-index unary rules by RHS for efficient lookup
def _build_unary_index(grammar):
    index = defaultdict(list)
    for (A, B), log_p in grammar.unary.items():
        index[B].append((A, log_p))
    return index

# Unary closure helper
def _apply_unary(chart, i, j, unary_index):
    # Work from a queue of NTs currently in the span
    agenda = list(chart[i][j].keys())
    while agenda:
        B = agenda.pop()
        if B not in unary_index:
            continue
        b_prob = chart[i][j][B].log_prob
        for A, log_p in unary_index[B]:
            total = log_p + b_prob
            if A not in chart[i][j] or chart[i][j][A].log_prob < total:
                cell          = ChartCell()
                cell.log_prob = total
                cell.left_child  = B
                cell.is_unary    = True
                chart[i][j][A]   = cell
                agenda.append(A)   # A may trigger further unary rules


# CKY algorithm
def cky_parse(tokens, grammar):
    n           = len(tokens)
    chart       = [[{} for _ in range(n)] for _ in range(n)]
    unary_index = _build_unary_index(grammar)

    # initialise lexical rules
    for i, word in enumerate(tokens):
        for (A, w), log_p in grammar.lexical.items():
            if w == word:
                if A not in chart[i][i] or chart[i][i][A].log_prob < log_p:
                    cell           = ChartCell()
                    cell.log_prob  = log_p
                    chart[i][i][A] = cell

        if not chart[i][i]:
            if not hasattr(grammar, '_nt_logprob_cache'):
                # Build cache: NT -> log(1/num_NTs) scaled by lexical count
                from collections import Counter
                nt_freq = Counter(A for (A, _) in grammar.lexical)
                total = sum(nt_freq.values())
                grammar._nt_logprob_cache = {
                    A: math.log(cnt / total) for A, cnt in nt_freq.items()
                }
            for A, lp in grammar._nt_logprob_cache.items():
                if A not in chart[i][i]:
                    cell           = ChartCell()
                    cell.log_prob  = lp
                    chart[i][i][A] = cell

        # Unary closure over span-1 after lexical init
        _apply_unary(chart, i, i, unary_index)

    #increasing span lengths
    for span in range(2, n + 1):
        for i in range(n - span + 1):
            j = i + span - 1

            for k in range(i, j):
                left_cell  = chart[i][k]
                right_cell = chart[k + 1][j]

                for (A, B, C), log_p in grammar.binary.items():
                    if B not in left_cell or C not in right_cell:
                        continue
                    total = log_p + left_cell[B].log_prob + right_cell[C].log_prob
                    if A not in chart[i][j] or chart[i][j][A].log_prob < total:
                        cell             = ChartCell()
                        cell.log_prob    = total
                        cell.left_child  = B
                        cell.right_child = C
                        cell.split       = k
                        chart[i][j][A]   = cell

            # Unary closure after binary rules for this span
            _apply_unary(chart, i, j, unary_index)

    #root
    start = grammar.start
    if start not in chart[0][n - 1]:
        root_cell = max(
            chart[0][n - 1].items(),
            key=lambda kv: kv[1].log_prob,
            default=(None, None)
        )
        if root_cell[0] is None:
            return None, NEG_INF
        start = root_cell[0]

    root_log_prob = chart[0][n - 1][start].log_prob
    tree = _backtrack(chart, tokens, 0, n - 1, start)
    return tree, root_log_prob


# Backtracking
#reconstruct the parse tree from the chart
def _backtrack(chart, tokens, i, j, label):
    cell = chart[i][j].get(label)
    if cell is None:
        return Tree(label, [Tree(tokens[i])])  # fallback leaf

    # Terminal leaf
    if i == j and not cell.is_unary and cell.split is None:
        return Tree(label, [Tree(tokens[i])])

    # Unary NT rule — is_unary flag makes the intent explicit
    if cell.is_unary:
        child = _backtrack(chart, tokens, i, j, cell.left_child)
        return Tree(label, [child])

    # Binary rule
    k     = cell.split
    left  = _backtrack(chart, tokens, i,     k, cell.left_child)
    right = _backtrack(chart, tokens, k + 1, j, cell.right_child)
    return Tree(label, [left, right])

    
# PP-attachment extraction
def get_pp_attachment(tree):
    return _find_pp_parent(tree, parent_label=None)

#Strip annotation suffixes
def _base_label(label):
    return label.split("^")[0].split("+")[0].split("|")[0]


# Nominal base labels 
_NOMINAL_BASES = {"NP", "NML", "NOUN", "PROPN", "PRON", "NN", "NNS", "NNP", "NNPS"}


def _find_pp_parent(tree, parent_label):
    if tree is None:
        return None
    base = _base_label(tree.label)

    if base == "PP" and parent_label is not None:
        parent_base = _base_label(parent_label)
        if parent_base in _NOMINAL_BASES:
            return "NP"
        else:
            return "VP"

    for child in tree.children:
        result = _find_pp_parent(child, tree.label)
        if result is not None:
            return result

    return None


# Scoring and public interface
def score_attachment(tokens, grammar, attachment_type):
    tree, log_prob = cky_parse(tokens, grammar)
    if tree is None:
        return NEG_INF
    att = get_pp_attachment(tree)
    return log_prob if att == attachment_type else NEG_INF

#Strip Arabic diacritics so test sentences match unvowelized PADT training token
def _normalize_tokens(tokens):
    result = []
    for tok in tokens:
        stripped = ''.join(c for c in tok if not (0x064B <= ord(c) <= 0x065F))
        result.append(stripped)
    return result


def parse_sentence(sentence, grammar):
    tokens = _normalize_tokens(sentence.strip().split())
    tree, log_prob = cky_parse(tokens, grammar)
    if tree is None:
        return None, NEG_INF, None
    return tree, log_prob, get_pp_attachment(tree)


# Demo
def main():
    print(" Probabilistic CKY Parser — Demo")

    model_dir = "models"
    test_sentences = {
        "English":  "I ate the sushi with chopsticks",
        "Japanese": "私は 箸で 寿司を 食べた",
        "Arabic":   "أكلت السوشي بالعيدان",
    }

    for lang, sentence in test_sentences.items():
        model_path = os.path.join(model_dir, f"{lang.lower()}_grammar.pkl")
        if not os.path.exists(model_path):
            print(f"\n[{lang}] Model not found at {model_path}. "
                  f"Run step2_pcfg_training.py first.")
            continue

        print(f"\n[{lang}] Loading grammar")
        with open(model_path, "rb") as f:
            grammar = pickle.load(f)

        print(f"[{lang}] Parsing: '{sentence}'")
        tree, log_prob, attachment = parse_sentence(sentence, grammar)

        if tree is None:
            print("  No parse found.")
        else:
            print(f"  Parse        : {tree}")
            print(f"  Log-prob     : {log_prob:.4f}")
            print(f"  PP-attachment: {attachment}")


if __name__ == "__main__":
    main()