"""Parent annotation and horizontal Markovization (Johnson, 1998)."""

from nltk.tree import Tree


def parent_annotate(tree: Tree, parent_label: str = None) -> Tree:
    """
    Annotate each node with its parent label: NP -> NP^VP.
    Improves PCFG by conditioning rules on syntactic context.
    """
    if not isinstance(tree, Tree):
        return tree
    label = tree.label()
    new_label = f"{label}^{parent_label}" if parent_label else label
    new_children = [parent_annotate(child, label) for child in tree]
    return Tree(new_label, new_children)


def markovize(tree: Tree, h: int = 2, v: int = 1) -> Tree:
    """
    Horizontal Markovization: limit RHS context to h siblings.
    v controls vertical history (handled via parent annotation).
    h=2 means keep at most 2 siblings of context in binarized rules.
    """
    if not isinstance(tree, Tree):
        return tree
    children = [markovize(child, h, v) for child in tree]
    if len(children) <= 2:
        return Tree(tree.label(), children)

    # Binarize right-recursively with h-bounded history
    label = tree.label()
    result = children[-1]
    for i in range(len(children) - 2, 0, -1):
        history = children[max(0, i - h + 1): i + 1]
        intermediate_label = f"{label}|<{'+'.join(c.label() if isinstance(c, Tree) else str(c) for c in history[-h:])}>"
        result = Tree(intermediate_label, [children[i], result])
    return Tree(label, [children[0], result])
