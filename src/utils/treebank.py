"""Utilities for loading and converting Universal Dependencies treebanks."""

import conllu
from nltk.tree import Tree


def load_conllu(filepath: str) -> list:
    """Load a .conllu file and return a list of parsed sentence TokenLists."""
    with open(filepath, encoding="utf-8") as f:
        return conllu.parse(f.read())


def conllu_to_nltk_tree(token_list) -> Tree | None:
    """
    Convert a CoNLL-U TokenList to an NLTK Tree using dependency structure.
    Returns None if conversion fails.
    """
    # Build id -> token map
    tokens = {t["id"]: t for t in token_list if isinstance(t["id"], int)}
    if not tokens:
        return None

    def build_subtree(node_id):
        token = tokens[node_id]
        children = [build_subtree(t["id"]) for t in token_list
                    if isinstance(t["id"], int) and t["head"] == node_id]
        label = token["upos"] or "X"
        leaf = Tree(token["form"], [])
        if not children:
            return Tree(label, [leaf])
        return Tree(label, [leaf] + children)

    root_ids = [t["id"] for t in token_list
                if isinstance(t["id"], int) and t["head"] == 0]
    if not root_ids:
        return None
    return Tree("ROOT", [build_subtree(rid) for rid in root_ids])
