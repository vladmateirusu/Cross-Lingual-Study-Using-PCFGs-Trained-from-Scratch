"""
Step 4 & 5: Ambiguous Test Set + Evaluation
"""

import os
import json
import csv
import math
import pickle

from step3_cky_parser import cky_parse, get_pp_attachment, NEG_INF


# Configuration
MODEL_DIR   = "models"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# Step 4: Ambiguous test sentences
TEST_SENTENCES = {

    # ENGLISH — 30 sentences
    # PP ambiguity: "with X" / "in X" / "on X" / "at X" / "for X"
    
    "English": [
        # (sentence, vp_reading_gloss, np_reading_gloss)
        ("I ate the sushi with chopsticks",
         "I used chopsticks to eat",          "The sushi came with chopsticks"),
        ("She photographed the actor with a telephoto lens",
         "She used a lens to photograph",     "The actor had a lens"),
        ("He called the manager with the complaint",
         "He used the complaint to call",     "The manager had the complaint"),
        ("They attacked the soldier with a knife",
         "They used a knife to attack",       "The soldier had a knife"),
        ("I saw the professor with the students",
         "I saw the professor alongside students", "The professor was with students"),
        ("She read the letter with glasses",
         "She used glasses to read",          "The letter came with glasses"),
        ("He painted the wall with a brush",
         "He used a brush to paint",          "The wall had a brush on it"),
        ("They found the suspect with the evidence",
         "They used evidence to find",        "The suspect had the evidence"),
        ("I met the director with the script",
         "I met the director using the script","The director had the script"),
        ("She greeted the guest with a smile",
         "She smiled while greeting",         "The guest had a smile"),
        ("He arrested the criminal with a warrant",
         "He used a warrant to arrest",       "The criminal had a warrant"),
        ("They cleaned the table with a cloth",
         "They used a cloth to clean",        "The table had a cloth on it"),
        ("I bought the car with the sunroof",
         "I used the sunroof deal to buy",    "The car had a sunroof"),
        ("She wrote the report with a pen",
         "She used a pen to write",           "The report had a pen attached"),
        ("He carried the box with the handle",
         "He used the handle to carry",       "The box had a handle"),
        ("They identified the package with a scanner",
         "They used a scanner to identify",   "The package had a scanner"),
        ("I examined the patient with a stethoscope",
         "I used a stethoscope to examine",   "The patient had a stethoscope"),
        ("She served the dish with chopsticks",
         "She used chopsticks to serve",      "The dish came with chopsticks"),
        ("He signed the contract with a pen",
         "He used a pen to sign",             "The contract came with a pen"),
        ("They caught the thief with a camera",
         "They used a camera to catch",       "The thief had a camera"),
        ("I fixed the machine with a wrench",
         "I used a wrench to fix",            "The machine had a wrench"),
        ("She measured the room with a tape",
         "She used a tape to measure",        "The room had tape marks"),
        ("He unlocked the door with the key",
         "He used the key to unlock",         "The door had a key"),
        ("They decorated the tree with lights",
         "They used lights to decorate",      "The tree had lights"),
        ("I helped the student with the assignment",
         "I helped using the assignment",     "The student had the assignment"),
        ("She tested the software with a script",
         "She used a script to test",         "The software had a script"),
        ("He repaired the bike with the tool",
         "He used the tool to repair",        "The bike had the tool"),
        ("They trained the model with the data",
         "They used data to train",           "The model had the data"),
        ("I contacted the professor with the form",
         "I used the form to contact",        "The professor had the form"),
        ("She cleaned the window with a sponge",
         "She used a sponge to clean",        "The window had a sponge"),
    ],

    # JAPANESE — 30 sentences  (romanized placeholders)
    # Replace with native-script tokenized sentences.
    # Structure: Subject + Object + PP (postposition) + Verb (verb-final)
    # Japanese postpositions: で (de, instrument), に (ni, location/goal)
    "Japanese": [
        ("watashi wa sushi wo hashi de tabeta",
         "ate using chopsticks",  "sushi with chopsticks"),
        ("kare wa sensei wo tegami de yonda",
         "called via letter",     "teacher with letter"),
        ("kanojo wa gakusei wo kamera de satsuei shita",
         "photographed using camera", "student with camera"),
        ("watashi wa hako wo te de hakonda",
         "carried by hand",       "box with handle"),
        ("kare wa kabe wo brush de nutta",
         "painted using brush",   "wall with brush"),
        ("kanojo wa repoto wo pen de kaita",
         "wrote using pen",       "report with pen"),
        ("watashi wa isha wo stetho de shinsatsu shita",
         "examined using stetho", "doctor with stetho"),
        ("kare wa keiyaku wo pen de shomei shita",
         "signed using pen",      "contract with pen"),
        ("kanojo wa mado wo sponge de migaita",
         "cleaned using sponge",  "window with sponge"),
        ("watashi wa kuruma wo kagi de aketa",
         "opened using key",      "car with key"),
        ("kare wa hanzaisha wo camera de tsukamaeta",
         "caught using camera",   "criminal with camera"),
        ("kanojo wa sofuto wo script de test shita",
         "tested using script",   "software with script"),
        ("watashi wa kyaku wo egao de mukaeta",
         "greeted with smile",    "guest with smile"),
        ("kare wa tsukue wo nuno de fuita",
         "wiped using cloth",     "desk with cloth"),
        ("kanojo wa nimotsu wo scanner de kakunin shita",
         "checked using scanner", "package with scanner"),
        ("watashi wa jitensha wo tool de naoishita",
         "repaired using tool",   "bike with tool"),
        ("kare wa model wo data de kunren shita",
         "trained using data",    "model with data"),
        ("kanojo wa heya wo tape de hakatta",
         "measured using tape",   "room with tape"),
        ("watashi wa ki wo raito de kazarita",
         "decorated using lights","tree with lights"),
        ("kare wa tobira wo kagi de aketa",
         "unlocked using key",    "door with key"),
        ("kanojo wa mono wo wrench de naoshita",
         "fixed using wrench",    "thing with wrench"),
        ("watashi wa heya wo soji shita mopp de",
         "cleaned using mop",     "room with mop"),
        ("kare wa tegami wo megane de yonda",
         "read using glasses",    "letter with glasses"),
        ("kanojo wa kabe wo burashi de nurimashita",
         "painted using brush",   "wall with brush"),
        ("watashi wa shokuhin wo fork de tabeta",
         "ate using fork",        "food with fork"),
        ("kare wa shashin wo camera de totta",
         "took using camera",     "photo with camera"),
        ("kanojo wa shigoto wo konpyuta de shita",
         "worked using computer", "work with computer"),
        ("watashi wa hako wo tape de tojita",
         "sealed using tape",     "box with tape"),
        ("kare wa kawa wo naifu de kitta",
         "cut using knife",       "leather with knife"),
        ("kanojo wa bideo wo projector de mita",
         "watched using projector","video with projector"),
    ],

    # ARABIC — 30 sentences  (transliterated placeholders)
    # Replace with native-script tokenized sentences.
    # Arabic prepositions: بـ (bi, with/instrument), في (fi, in), على (ala, on)
    "Arabic": [
        ("akaltu alsushi bialeidani",
         "ate using chopsticks",  "sushi with chopsticks"),
        ("sawwartu altalibu bialkamira",
         "photographed using camera", "student with camera"),
        ("katabtu alttaqrir bialqalam",
         "wrote using pen",       "report with pen"),
        ("qara'tu alrisalata bialnnadhdharatayn",
         "read using glasses",    "letter with glasses"),
        ("hasaltu ealaa almujrimi biddalil",
         "caught using evidence", "criminal with evidence"),
        ("nazaftu alnaafidhata bialisifinja",
         "cleaned using sponge",  "window with sponge"),
        ("fatahtu albaaba bialmuftah",
         "opened using key",      "door with key"),
        ("qistu alghurfata bilshirit",
         "measured using tape",   "room with tape"),
        ("sallahttu aldraajata biladaati",
         "repaired using tool",   "bike with tool"),
        ("rashahtu alssafha bialmibasha",
         "painted using brush",   "page with brush"),
        ("waqaetu almukhriim bialmistanda",
         "caught using scanner",  "criminal with scanner"),
        ("darbtu almismaar bialmitarqa",
         "hit using hammer",      "nail with hammer"),
        ("nazaftu almaa'ida bilqumash",
         "cleaned using cloth",   "table with cloth"),
        ("rasama alhaita bilfirsha",
         "drew using brush",      "wall with brush"),
        ("taawwaltu almareeda bialssamaaeat",
         "examined using stethoscope", "patient with stethoscope"),
        ("waqaeat almujrim bialkamiira",
         "caught using camera",   "criminal with camera"),
        ("waqaeat alsariq bialddalil",
         "caught using evidence", "thief with evidence"),
        ("qara'tu almaqala bialnnadhdharatayn",
         "read using glasses",    "article with glasses"),
        ("kataba almuqawila biallaqalam",
         "wrote using pen",       "contract with pen"),
        ("dhabahtu alasuud biassikin",
         "cut using knife",       "meat with knife"),
        ("tanahtu alkhashb bialminshar",
         "sawed using saw",       "wood with saw"),
        ("hafara alard bialmijraf",
         "dug using shovel",      "ground with shovel"),
        ("ramaa alkulata biayyadihi",
         "threw using hands",     "ball with hands"),
        ("fatahtu alnnaafidha bialmiqdah",
         "opened using handle",   "window with handle"),
        ("lamasa almareeda biyadihi",
         "touched using hand",    "patient with hand"),
        ("nazaftu alsajjaada bilmiknasa",
         "cleaned using broom",   "carpet with broom"),
        ("rakabtu aldraajata biyaday",
         "rode using hands",      "bike with hands"),
        ("shabaka alssabak bishabakatihi",
         "fished using net",      "fish with net"),
        ("qataa alwraq bilmiqass",
         "cut using scissors",    "paper with scissors"),
        ("aedda altaaam bialssakkin",
         "prepared using knife",  "food with knife"),
    ],
}


# Step 5: Evaluation
# Run the CKY parser on each sentence, record attachment decisio and probability margin
# Returns a list of dicts with per-sentence results
def evaluate_language(lang, sentences, grammar):
    records = []

    for idx, entry in enumerate(sentences):
        sentence, vp_gloss, np_gloss = entry
        tokens = sentence.strip().split()

        # Parse once to get the best parse
        tree, log_prob = cky_parse(tokens, grammar)

        if tree is None:
            records.append({
                "id":          idx + 1,
                "sentence":    sentence,
                "attachment":  "FAIL",
                "log_prob":    None,
                "margin":      None,
                "vp_gloss":    vp_gloss,
                "np_gloss":    np_gloss,
            })
            continue

        attachment = get_pp_attachment(tree)

        # Compute margin
        if attachment == "VP":
            margin = log_prob         # positive
        elif attachment == "NP":
            margin = -abs(log_prob)   # negative to signal NP preference
        else:
            margin = 0.0

        records.append({
            "id":         idx + 1,
            "sentence":   sentence,
            "attachment": attachment if attachment else "OTHER",
            "log_prob":   log_prob,
            "margin":     margin,
            "vp_gloss":   vp_gloss,
            "np_gloss":   np_gloss,
        })

    return records


def summarize(lang, records):
    total   = len(records)
    vp      = sum(1 for r in records if r["attachment"] == "VP")
    np      = sum(1 for r in records if r["attachment"] == "NP")
    fails   = sum(1 for r in records if r["attachment"] == "FAIL")
    other   = total - vp - np - fails

    margins = [r["margin"] for r in records if r["margin"] is not None]
    avg_margin = sum(margins) / len(margins) if margins else 0.0

    vp_rate = vp / total * 100 if total > 0 else 0
    np_rate = np / total * 100 if total > 0 else 0

    print(f"\n  {lang} Results ({total} sentences):")
    print(f"    VP-attachment : {vp:3d}  ({vp_rate:.1f}%)")
    print(f"    NP-attachment : {np:3d}  ({np_rate:.1f}%)")
    print(f"    Other/unknown : {other:3d}")
    print(f"    Parse failures: {fails:3d}")
    print(f"    Avg margin    : {avg_margin:.4f}")
    print(f"    (positive margin => VP preference, negative => NP preference)")

    return {
        "language":    lang,
        "total":       total,
        "vp_count":    vp,
        "np_count":    np,
        "vp_rate":     vp_rate,
        "np_rate":     np_rate,
        "avg_margin":  avg_margin,
        "parse_fails": fails,
    }


def save_results(all_records, summaries):

    # Detailed JSON
    json_path = os.path.join(RESULTS_DIR, "results_detailed.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)
    print(f"\n  Detailed results saved to {json_path}")

    # Summary CSV
    csv_path = os.path.join(RESULTS_DIR, "results_summary.csv")
    fieldnames = ["language", "total", "vp_count", "np_count",
                  "vp_rate", "np_rate", "avg_margin", "parse_fails"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    print(f"  Summary CSV    saved to {csv_path}")

    # Per-sentence CSV 
    per_sent_path = os.path.join(RESULTS_DIR, "results_per_sentence.csv")
    flat = []
    for lang, records in all_records.items():
        for r in records:
            r2 = dict(r); r2["language"] = lang
            flat.append(r2)
    fields2 = ["language", "id", "sentence", "attachment",
                "log_prob", "margin", "vp_gloss", "np_gloss"]
    with open(per_sent_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields2)
        writer.writeheader()
        writer.writerows(flat)
    print(f"  Per-sentence CSV saved to {per_sent_path}")


# Main
def main():
    print("=" * 60)
    print("STEP 4 & 5: Test Set Evaluation")
    print("=" * 60)

    all_records = {}
    summaries   = []

    for lang, sentences in TEST_SENTENCES.items():
        model_path = os.path.join(MODEL_DIR, f"{lang.lower()}_grammar.pkl")

        if not os.path.exists(model_path):
            print(f"\n[{lang}] Grammar not found at {model_path}. "
                  f"Run step2_pcfg_training.py first.")
            continue

        print(f"\n[{lang}] Loading grammar")
        with open(model_path, "rb") as f:
            grammar = pickle.load(f)

        print(f"[{lang}] Evaluating {len(sentences)} sentences ")
        records = evaluate_language(lang, sentences, grammar)
        summary = summarize(lang, records)

        all_records[lang] = records
        summaries.append(summary)

    #Final paper table
    print("result table")
    print(f"{'Language':<12} {'VP%':>6} {'NP%':>6} {'Avg margin':>12}")
    print("-" * 40)
    for s in summaries:
        print(f"{s['language']:<12} {s['vp_rate']:>5.1f}% "
              f"{s['np_rate']:>5.1f}% {s['avg_margin']:>12.4f}")

    if summaries:
        save_results(all_records, summaries)

    return all_records, summaries


if __name__ == "__main__":
    main()
