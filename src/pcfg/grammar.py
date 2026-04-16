"""PCFG rule extraction and MLE probability estimation from treebanks."""

from collections import defaultdict
from nltk.tree import Tree
from nltk import PCFG, ProbabilisticProduction, Nonterminal


def extract_rules(tree: Tree) -> list[tuple]:
    """Recursively extract (lhs, rhs) production rules from an NLTK Tree."""
    rules = []
    if isinstance(tree, Tree):
        lhs = tree.label()
        rhs = tuple(
            child.label() if isinstance(child, Tree) else child
            for child in tree
        )
        rules.append((lhs, rhs))
        for child in tree:
            rules.extend(extract_rules(child))
    return rules


def estimate_mle(trees: list[Tree]) -> dict:
    """
    Compute MLE rule probabilities from a list of trees.
    Returns dict mapping (lhs, rhs) -> probability.
    """
    lhs_counts = defaultdict(int)
    rule_counts = defaultdict(int)

    for tree in trees:
        for lhs, rhs in extract_rules(tree):
            lhs_counts[lhs] += 1
            rule_counts[(lhs, rhs)] += 1

    probabilities = {}
    for (lhs, rhs), count in rule_counts.items():
        probabilities[(lhs, rhs)] = count / lhs_counts[lhs]
    return probabilities


def build_pcfg(prob_rules: dict) -> PCFG:
    """Build an NLTK PCFG object from a probability dictionary."""
    productions = []
    for (lhs, rhs), prob in prob_rules.items():
        nt_lhs = Nonterminal(lhs)
        nt_rhs = tuple(
            Nonterminal(s) if isinstance(s, str) and s.isupper() else s
            for s in rhs
        )
        productions.append(ProbabilisticProduction(nt_lhs, nt_rhs, prob=prob))
    return PCFG(Nonterminal("ROOT"), productions)
