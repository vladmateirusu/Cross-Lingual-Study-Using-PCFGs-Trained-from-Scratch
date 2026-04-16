"""Probabilistic CKY (Viterbi) parser for a PCFG."""

import math
from collections import defaultdict
from nltk import PCFG, Nonterminal
from nltk.tree import Tree


def cky_parse(sentence: list[str], grammar: PCFG) -> Tree | None:
    """
    Run Viterbi CKY parsing on a tokenized sentence.
    Returns the most probable parse Tree, or None if no parse found.
    """
    n = len(sentence)
    # table[i][j][nt] = (log_prob, back_pointer)
    table = [[defaultdict(lambda: (-math.inf, None)) for _ in range(n + 1)] for _ in range(n)]

    # Index grammar rules
    unary = defaultdict(list)   # nt -> [(word, log_prob)]
    binary = defaultdict(list)  # nt -> [(B, C, log_prob)]

    for prod in grammar.productions():
        lhs = prod.lhs().symbol()
        lp = math.log(prod.prob()) if prod.prob() > 0 else -math.inf
        if len(prod.rhs()) == 1 and isinstance(prod.rhs()[0], str):
            unary[lhs].append((prod.rhs()[0], lp))
        elif len(prod.rhs()) == 2:
            b, c = prod.rhs()
            binary[lhs].append((b.symbol(), c.symbol(), lp))

    # Fill diagonal (lexical)
    for i, word in enumerate(sentence):
        for nt, entries in unary.items():
            for w, lp in entries:
                if w == word and lp > table[i][i + 1][nt][0]:
                    table[i][i + 1][nt] = (lp, word)

    # Fill spans length 2..n
    for span in range(2, n + 1):
        for i in range(n - span + 1):
            j = i + span
            for k in range(i + 1, j):
                for nt, entries in binary.items():
                    for b, c, lp in entries:
                        b_lp = table[i][k][b][0]
                        c_lp = table[k][j][c][0]
                        if b_lp == -math.inf or c_lp == -math.inf:
                            continue
                        total = lp + b_lp + c_lp
                        if total > table[i][j][nt][0]:
                            table[i][j][nt] = (total, (k, b, c))

    start = grammar.start().symbol()
    if table[0][n][start][0] == -math.inf:
        return None

    def build_tree(nt, i, j):
        _, back = table[i][j][nt]
        if isinstance(back, str):
            return Tree(nt, [back])
        k, b, c = back
        return Tree(nt, [build_tree(b, i, k), build_tree(c, k, j)])

    return build_tree(start, 0, n)
