"""
Reads Universal Dependencies (UD) treebanks for English, Japanese, and Arabic.
For every PP (prepositional/postpositional phrase head), records whether its
syntactic head is a VERB (VP-attachment) or NOUN (NP-attachment).
Outputs baseline attachment rates and PP frequency per sentence.

Expected treebank files (CoNLL-U format):
  data/en_ewt-ud-train.conllu
  data/ja_gsd-ud-train.conllu
  data/ar_padt-ud-train.conllu

Download from: https://universaldependencies.org/
"""

import os
import conllu
from collections import defaultdict


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TREEBANKS = {
    "English": "data/en_ewt-ud-train.conllu",
    "Japanese": "data/ja_gsd-ud-train.conllu",
    "Arabic":   "data/ar_padt-ud-train.conllu",
}

VERBAL_UPOS  = {"VERB", "AUX"}
NOMINAL_UPOS = {"NOUN", "PROPN", "PRON"}

#We find PPs by looking for tokens labeled obl or nmod that have a case child (the preposition/postposition)
#These tokens act as the main noun of the PP, and their head determines whether it attaches to a verb or noun
PP_DEPRELS = {"obl", "nmod"}   

#helpers

#Load the file and return a list of parsed sentences
def load_treebank(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"file not found: {path}\n"
        )
    with open(path, encoding="utf-8") as f:
        data = f.read()
    return conllu.parse(data)

#dict {token_id -> token}
def build_index(sentence):
    return {token["id"]: token for token in sentence
            if isinstance(token["id"], int)}

#Return True if any token in the sentence has deprel='case' and head=token_id
def has_case_dependent(token_id, index):
    for tok in index.values():
        if tok["deprel"] == "case" and tok["head"] == token_id:
            return True
    return False

"""
    Analyze a UD treebank and return:
      - vp_count   : number of PPs attaching to a verbal head
      - np_count   : number of PPs attaching to a nominal head
      - other_count: PPs attaching to other POS
      - total_sents: number of sentences
      - total_pps  : total PPs found
    """
def analyze_treebank(path):
    sentences = load_treebank(path)

    vp_count    = 0
    np_count    = 0
    other_count = 0
    total_sents = len(sentences)
    total_pps   = 0

    for sent in sentences:
        index = build_index(sent)

        for token in sent:
            # Skip multi-word tokens
            if not isinstance(token["id"], int):
                continue

            # A PP in UD is typically a nominal/oblique dependent
            # that has a 'case' adposition as its own dependent
            if token["deprel"] not in PP_DEPRELS:
                continue
            if not has_case_dependent(token["id"], index):
                continue

            total_pps += 1

            # Find the head of this PP node
            head_id = token["head"]
            if head_id == 0:
                other_count += 1
                continue

            head_tok = index.get(head_id)
            if head_tok is None:
                other_count += 1
                continue

            head_upos = head_tok["upos"]
            if head_upos in VERBAL_UPOS:
                vp_count += 1
            elif head_upos in NOMINAL_UPOS:
                np_count += 1
            else:
                other_count += 1

    return {
        "vp_count":    vp_count,
        "np_count":    np_count,
        "other_count": other_count,
        "total_sents": total_sents,
        "total_pps":   total_pps,
    }

#main 
def main():

    print("Treebank PP-Attachment Baseline Analysis")

    results = {}

    for lang, path in TREEBANKS.items():
        print(f"\nAnalyzing {lang} ({path})")
        try:
            stats = analyze_treebank(path)
            results[lang] = stats

            vp    = stats["vp_count"]
            np    = stats["np_count"]
            other = stats["other_count"]
            total = vp + np + other
            sents = stats["total_sents"]
            pps   = stats["total_pps"]

            vp_rate = vp / total * 100 if total > 0 else 0
            np_rate = np / total * 100 if total > 0 else 0
            pps_per_sent = pps / sents if sents > 0 else 0

            print(f"  Sentences        : {sents}")
            print(f"  Total PPs found  : {pps}")
            print(f"  PPs/sentence     : {pps_per_sent:.2f}")
            print(f"  VP-attachment    : {vp}  ({vp_rate:.1f}%)")
            print(f"  NP-attachment    : {np}  ({np_rate:.1f}%)")
            print(f"  Other/root       : {other}")

        except FileNotFoundError as e:
            print(f"  [SKIPPED] {e}")

    # --- Summary table ---
    print("\n" + "=" * 60)
    print("SUMMARY TABLE (for paper)")
    print("=" * 60)
    print(f"{'Language':<12} {'VP%':>6} {'NP%':>6} {'PPs/sent':>10}")
    print("-" * 38)
    for lang, stats in results.items():
        vp    = stats["vp_count"]
        np    = stats["np_count"]
        other = stats["other_count"]
        total = vp + np + other
        sents = stats["total_sents"]
        pps   = stats["total_pps"]
        vp_rate = vp / total * 100 if total > 0 else 0
        np_rate = np / total * 100 if total > 0 else 0
        pps_per_sent = pps / sents if sents > 0 else 0
        print(f"{lang:<12} {vp_rate:>5.1f}% {np_rate:>5.1f}% {pps_per_sent:>10.2f}")

    return results


if __name__ == "__main__":
    main()
