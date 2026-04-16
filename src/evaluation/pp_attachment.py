"""
Classify PP-attachment decisions from parse trees and compute cross-lingual statistics.

For each ambiguous test sentence, we check whether the PP node's parent in the
best parse is a VP (verbal attachment) or NP (nominal attachment).
"""

from nltk.tree import Tree
import pandas as pd


def find_pp_attachment(parse: Tree) -> str | None:
    """
    Walk the parse tree to find a PP node and return its parent's label.
    Returns 'VP', 'NP', or None if no PP found.
    """
    def _walk(tree, parent_label=None):
        if not isinstance(tree, Tree):
            return None
        if tree.label().startswith("PP"):
            return parent_label
        for child in tree:
            result = _walk(child, tree.label())
            if result is not None:
                return result
        return None

    return _walk(parse)


def classify_attachment(parent_label: str | None) -> str:
    """Map a parent label to VP-attach, NP-attach, or unknown."""
    if parent_label is None:
        return "unknown"
    label = parent_label.split("^")[0]  # strip parent annotation
    if label in ("VP", "S", "SBAR", "IP"):
        return "VP-attach"
    if label in ("NP", "DP", "N"):
        return "NP-attach"
    return "unknown"


def summarize_results(records: list[dict]) -> pd.DataFrame:
    """
    records: list of dicts with keys: language, sentence_id, attachment
    Returns a summary DataFrame with VP-attach/NP-attach rates per language.
    """
    df = pd.DataFrame(records)
    summary = (
        df[df["attachment"] != "unknown"]
        .groupby(["language", "attachment"])
        .size()
        .unstack(fill_value=0)
    )
    for col in ("VP-attach", "NP-attach"):
        if col not in summary.columns:
            summary[col] = 0
    summary["total"] = summary["VP-attach"] + summary["NP-attach"]
    summary["VP-rate"] = summary["VP-attach"] / summary["total"]
    summary["NP-rate"] = summary["NP-attach"] / summary["total"]
    return summary
