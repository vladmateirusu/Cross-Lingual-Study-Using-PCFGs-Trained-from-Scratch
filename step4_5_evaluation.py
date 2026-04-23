
import os
import json
import csv
import math
import pickle

from step3_cky_parser import cky_parse, get_pp_attachment, score_attachment, NEG_INF


# Configuration
MODEL_DIR   = "models"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# Step 4: Ambiguous test sentences
# Tuple format: (sentence, vp_gloss, np_gloss, verified)
TEST_SENTENCES = {

    # ENGLISH — 30 sentences
    # All hand-crafted; structure: Subject Verb Det-Object PP
    "English": [
        ("I ate the sushi with chopsticks",
         "I used chopsticks to eat",            "The sushi came with chopsticks",        True),
        ("She photographed the actor with a telephoto lens",
         "She used a lens to photograph",       "The actor had a lens",                  True),
        ("He called the manager with the complaint",
         "He used the complaint to call",       "The manager had the complaint",          True),
        ("They attacked the soldier with a knife",
         "They used a knife to attack",         "The soldier had a knife",               True),
        ("I saw the professor with the students",
         "I saw alongside the students",        "The professor was with students",        True),
        ("She read the letter with glasses",
         "She used glasses to read",            "The letter came with glasses",           True),
        ("He painted the wall with a brush",
         "He used a brush to paint",            "The wall had a brush on it",             True),
        ("They found the suspect with the evidence",
         "They used evidence to find",          "The suspect had the evidence",           True),
        ("I met the director with the script",
         "I met using the script",              "The director had the script",            True),
        ("She greeted the guest with a smile",
         "She smiled while greeting",           "The guest had a smile",                  True),
        ("He arrested the criminal with a warrant",
         "He used a warrant to arrest",         "The criminal had a warrant",             True),
        ("They cleaned the table with a cloth",
         "They used a cloth to clean",          "The table had a cloth on it",            True),
        ("I bought the car with the sunroof",
         "I used the sunroof deal to buy",      "The car had a sunroof",                  True),
        ("She wrote the report with a pen",
         "She used a pen to write",             "The report had a pen attached",          True),
        ("He carried the box with the handle",
         "He used the handle to carry",         "The box had a handle",                   True),
        ("They identified the package with a scanner",
         "They used a scanner to identify",     "The package had a scanner",              True),
        ("I examined the patient with a stethoscope",
         "I used a stethoscope to examine",     "The patient had a stethoscope",          True),
        ("She served the dish with chopsticks",
         "She used chopsticks to serve",        "The dish came with chopsticks",          True),
        ("He signed the contract with a pen",
         "He used a pen to sign",               "The contract came with a pen",           True),
        ("They caught the thief with a camera",
         "They used a camera to catch",         "The thief had a camera",                 True),
        ("I fixed the machine with a wrench",
         "I used a wrench to fix",              "The machine had a wrench",               True),
        ("She measured the room with a tape",
         "She used a tape to measure",          "The room had tape marks",                True),
        ("He unlocked the door with the key",
         "He used the key to unlock",           "The door had a key",                     True),
        ("They decorated the tree with lights",
         "They used lights to decorate",        "The tree had lights",                    True),
        ("I helped the student with the assignment",
         "I helped using the assignment",       "The student had the assignment",         True),
        ("She tested the software with a script",
         "She used a script to test",           "The software had a script",              True),
        ("He repaired the bike with the tool",
         "He used the tool to repair",          "The bike had the tool",                  True),
        ("They trained the model with the data",
         "They used data to train",             "The model had the data",                 True),
        ("I contacted the professor with the form",
         "I used the form to contact",          "The professor had the form",             True),
        ("She cleaned the window with a sponge",
         "She used a sponge to clean",          "The window had a sponge",                True),
    ],

    # JAPANESE — 30 sentences
    # Structure: Topic-は Instrument-で Object-を Verb
    "Japanese": [
        ("私 は 箸 で 寿司 を 食べ た",
         "箸で食べた",        "箸付きの寿司",       True),
        ("彼 は 手紙 で 先生 を 呼ん だ",
         "手紙で呼んだ",      "手紙を持つ先生",     False),
        ("彼女 は カメラ で 学生 を 撮影 し た",
         "カメラで撮影",      "カメラを持つ学生",   False),
        ("私 は 手 で 箱 を 運ん だ",
         "手で運んだ",        "手付きの箱",         False),
        ("彼 は ブラシ で 壁 を 塗っ た",
         "ブラシで塗った",    "ブラシ付きの壁",     False),
        ("彼女 は ペン で レポート を 書い た",
         "ペンで書いた",      "ペン付きのレポート", False),
        ("私 は 聴診器 で 患者 を 診察 し た",
         "聴診器で診察",      "聴診器を持つ患者",   False),
        ("彼 は ペン で 契約 を 署名 し た",
         "ペンで署名",        "ペン付きの契約",     False),
        ("彼女 は スポンジ で 窓 を 拭い た",
         "スポンジで拭いた",  "スポンジ付きの窓",   False),
        ("私 は 鍵 で 車 を 開け た",
         "鍵で開けた",        "鍵付きの車",         False),
        ("彼 は カメラ で 犯罪者 を 捕まえ た",
         "カメラで捕まえた",  "カメラを持つ犯罪者", False),
        ("彼女 は スクリプト で ソフト を テスト し た",
         "スクリプトでテスト","スクリプト付きのソフト", False),
        ("私 は 笑顔 で 客 を 迎え た",
         "笑顔で迎えた",      "笑顔の客",           False),
        ("彼 は 布 で 机 を 拭い た",
         "布で拭いた",        "布付きの机",         False),
        ("彼女 は スキャナ で 荷物 を 確認 し た",
         "スキャナで確認",    "スキャナ付きの荷物", False),
        ("私 は 道具 で 自転車 を 修理 し た",
         "道具で修理",        "道具付きの自転車",   False),
        ("彼 は データ で モデル を 訓練 し た",
         "データで訓練",      "データ付きのモデル", False),
        ("彼女 は テープ で 部屋 を 測っ た",
         "テープで測った",    "テープ付きの部屋",   False),
        ("私 は ライト で 木 を 飾っ た",
         "ライトで飾った",    "ライト付きの木",     False),
        ("彼 は 鍵 で ドア を 開け た",
         "鍵で開けた",        "鍵付きのドア",       False),
        ("彼女 は レンチ で 機械 を 直し た",
         "レンチで直した",    "レンチ付きの機械",   False),
        ("私 は モップ で 部屋 を 掃除 し た",
         "モップで掃除",      "モップ付きの部屋",   False),
        ("彼 は 眼鏡 で 手紙 を 読ん だ",
         "眼鏡で読んだ",      "眼鏡付きの手紙",     False),
        ("彼女 は ブラシ で 壁 を 塗り まし た",
         "ブラシで塗った",    "ブラシ付きの壁",     False),
        ("私 は フォーク で 食べ物 を 食べ た",
         "フォークで食べた",  "フォーク付きの食べ物", False),
        ("彼 は カメラ で 写真 を 撮っ た",
         "カメラで撮った",    "カメラ付きの写真",   False),
        ("彼女 は コンピュータ で 仕事 を し た",
         "コンピュータで仕事","コンピュータ付きの仕事", False),
        ("私 は テープ で 箱 を 閉じ た",
         "テープで閉じた",    "テープ付きの箱",     False),
        ("彼 は ナイフ で 革 を 切っ た",
         "ナイフで切った",    "ナイフ付きの革",     False),
        ("彼女 は プロジェクタ で 動画 を 見 た",
         "プロジェクタで見た","プロジェクタ付きの動画", False),
    ],

    # ARABIC — 30 sentences
    # Structure: Verb Object ب+Instrument  (VSO + PP)
    "Arabic": [
        ("أكلت السوشي بالعيدان",
         "أكل بالعيدان",       "سوشي بالعيدان",       False),
        ("صورت الطالب بالكاميرا",
         "صور بالكاميرا",      "طالب بالكاميرا",      False),
        ("كتبت التقرير بالقلم",
         "كتب بالقلم",         "تقرير بالقلم",        False),
        ("قرأت الرسالة بالنظارتين",
         "قرأ بالنظارة",       "رسالة بالنظارة",      False),
        ("أمسكت المجرم بالدليل",
         "أمسك بالدليل",       "مجرم بالدليل",        False),
        ("نظفت النافذة بالإسفنجة",
         "نظف بالإسفنجة",      "نافذة بالإسفنجة",     False),
        ("فتحت الباب بالمفتاح",
         "فتح بالمفتاح",       "باب بالمفتاح",        False),
        ("قست الغرفة بالشريط",
         "قاس بالشريط",        "غرفة بالشريط",        False),
        ("أصلحت الدراجة بالأداة",
         "أصلح بالأداة",       "دراجة بالأداة",       False),
        ("رسمت الجدار بالفرشاة",
         "رسم بالفرشاة",       "جدار بالفرشاة",       False),
        ("أمسكت المخرب بالماسحة",
         "أمسك بالماسحة",      "مخرب بالماسحة",       False),
        ("ضربت المسمار بالمطرقة",
         "ضرب بالمطرقة",       "مسمار بالمطرقة",      False),
        ("نظفت المائدة بالقماش",
         "نظف بالقماش",        "مائدة بالقماش",       False),
        ("رسمت الحائط بالفرشاة",
         "رسم بالفرشاة",       "حائط بالفرشاة",       False),
        ("فحصت المريض بالسماعة",
         "فحص بالسماعة",       "مريض بالسماعة",       False),
        ("أمسكت السارق بالدليل",
         "أمسك بالدليل",       "سارق بالدليل",        False),
        ("قرأت المقالة بالنظارتين",
         "قرأ بالنظارة",       "مقالة بالنظارة",      False),
        ("كتب المقاول العقد بالقلم",
         "كتب بالقلم",         "عقد بالقلم",          False),
        ("ذبحت اللحم بالسكين",
         "ذبح بالسكين",        "لحم بالسكين",         False),
        ("نشرت الخشب بالمنشار",
         "نشر بالمنشار",       "خشب بالمنشار",        False),
        ("حفرت الأرض بالمجرفة",
         "حفر بالمجرفة",       "أرض بالمجرفة",        False),
        ("رميت الكرة بيدي",
         "رمى بيديه",          "كرة بيديه",           False),
        ("فتحت النافذة بالمقبض",
         "فتح بالمقبض",        "نافذة بالمقبض",       False),
        ("لمست المريض بيده",
         "لمس بيده",           "مريض بيده",           False),
        ("نظفت السجادة بالمكنسة",
         "نظف بالمكنسة",       "سجادة بالمكنسة",      False),
        ("ركبت الدراجة بيدي",
         "ركب بيديه",          "دراجة بيديه",         False),
        ("اصطاد السمك بشبكته",
         "اصطاد بشبكته",       "سمك بشبكته",          False),
        ("قطعت الورق بالمقص",
         "قطع بالمقص",         "ورق بالمقص",          False),
        ("أعددت الطعام بالسكين",
         "أعد بالسكين",        "طعام بالسكين",        False),
        ("حركت الصندوق بالرافعة",
         "حرك بالرافعة",       "صندوق بالرافعة",      False),
    ],
}


# Step 5: Evaluation
#Strip Arabic diacritics so tokens match unvowelized PADT training vocab
def _normalize(tokens):
    return [''.join(c for c in t if not (0x064B <= ord(c) <= 0x065F))
            for t in tokens]

def _collect_labels(tree):
    if tree is None: return []
    return [tree.label] + [l for c in tree.children for l in _collect_labels(c)]

#Parse each sentence and record attachment decision + margin
def evaluate_language(lang, sentences, grammar):
    MAX_UNILATERAL = 999.0
    records = []

    for idx, entry in enumerate(sentences):
        sentence, vp_gloss, np_gloss, verified = entry
        tokens = _normalize(sentence.strip().split())

        # Score each attachment type independently
        vp_score = score_attachment(tokens, grammar, "VP")
        np_score = score_attachment(tokens, grammar, "NP")

        # Best overall parse (for the attachment label and log_prob columns)
        tree, log_prob = cky_parse(tokens, grammar)

        if tree is None:
            records.append({
                "id":         idx + 1,
                "sentence":   sentence,
                "attachment": "FAIL",
                "log_prob":   None,
                "vp_score":   None,
                "np_score":   None,
                "margin":     None,
                "verified":   verified,
                "vp_gloss":   vp_gloss,
                "np_gloss":   np_gloss,
            })
            continue

        attachment = get_pp_attachment(tree) if tree else "OTHER"
        if attachment is None:
            attachment = "OTHER"

        if attachment == "OTHER":
            labels = set(_collect_labels(tree))
            print(f"  DEBUG [{lang}] sent {idx+1}: {labels}")
        # Compute margin: vp_score − np_score
        both_valid = vp_score > NEG_INF and np_score > NEG_INF
        if both_valid:
            margin = vp_score - np_score
        elif vp_score > NEG_INF:
            margin =  MAX_UNILATERAL   # only VP parse found
        elif np_score > NEG_INF:
            margin = -MAX_UNILATERAL   # only NP parse found
        else:
            margin = 0.0

        records.append({
            "id":         idx + 1,
            "sentence":   sentence,
            "attachment": attachment,
            "log_prob":   log_prob,
            "vp_score":   vp_score   if vp_score > NEG_INF else None,
            "np_score":   np_score   if np_score > NEG_INF else None,
            "margin":     margin,
            "verified":   verified,
            "vp_gloss":   vp_gloss,
            "np_gloss":   np_gloss,
        })

    return records


def summarize(lang, records):
    total  = len(records)
    vp     = sum(1 for r in records if r["attachment"] == "VP")
    np     = sum(1 for r in records if r["attachment"] == "NP")
    fails  = sum(1 for r in records if r["attachment"] == "FAIL")
    other  = total - vp - np - fails

    verified_total = sum(1 for r in records if r["verified"])
    verified_vp    = sum(1 for r in records if r["verified"] and r["attachment"] == "VP")
    verified_np    = sum(1 for r in records if r["verified"] and r["attachment"] == "NP")

    margins    = [r["margin"] for r in records if r["margin"] is not None]
    avg_margin = sum(margins) / len(margins) if margins else 0.0

    vp_rate = vp / total * 100 if total > 0 else 0
    np_rate = np / total * 100 if total > 0 else 0

    print(f"\n  {lang} Results ({total} sentences):")
    print(f"    VP-attachment   : {vp:3d}  ({vp_rate:.1f}%)")
    print(f"    NP-attachment   : {np:3d}  ({np_rate:.1f}%)")
    print(f"    Other/unknown   : {other:3d}")
    print(f"    Parse failures  : {fails:3d}")
    print(f"    Avg margin      : {avg_margin:.4f}  "
          f"(vp_score − np_score; + = VP preferred, − = NP preferred)")
    print(f"    Verified sents  : {verified_total}/{total}  "
          f"(VP={verified_vp}, NP={verified_np})")

    return {
        "language":       lang,
        "total":          total,
        "vp_count":       vp,
        "np_count":       np,
        "vp_rate":        vp_rate,
        "np_rate":        np_rate,
        "avg_margin":     avg_margin,
        "parse_fails":    fails,
        "verified_total": verified_total,
    }


def save_results(all_records, summaries, baseline_results=None):

    # Detailed JSON
    json_path = os.path.join(RESULTS_DIR, "results_detailed.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)
    print(f"\n  Detailed results  → {json_path}")

    # Summary CSV
    csv_path = os.path.join(RESULTS_DIR, "results_summary.csv")
    fieldnames = ["language", "total", "vp_count", "np_count",
                  "vp_rate", "np_rate", "avg_margin", "parse_fails",
                  "verified_total"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    print(f"  Summary CSV       → {csv_path}")

    # Per-sentence CSV
    per_sent_path = os.path.join(RESULTS_DIR, "results_per_sentence.csv")
    flat = []
    for lang, records in all_records.items():
        for r in records:
            r2 = dict(r)
            r2["language"] = lang
            flat.append(r2)
    fields2 = ["language", "id", "sentence", "attachment",
                "log_prob", "vp_score", "np_score", "margin",
                "verified", "vp_gloss", "np_gloss"]
    with open(per_sent_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields2)
        writer.writeheader()
        writer.writerows(flat)
    print(f"  Per-sentence CSV  → {per_sent_path}")

    # Table 2: Baseline vs. Parser comparison (the paper's main results table)
    if baseline_results:
        table2_path = os.path.join(RESULTS_DIR, "table2_comparison.csv")
        with open(table2_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Language",
                "Baseline_VP%", "Baseline_NP%",
                "Parser_VP%",   "Parser_NP%",
                "Avg_Margin (VP_score - NP_score)",
                "Parse_Fails",  "Verified_Sents"
            ])
            for s in summaries:
                lang  = s["language"]
                base  = baseline_results.get(lang, {})
                b_vp  = base.get("vp_count",    0)
                b_np  = base.get("np_count",     0)
                b_oth = base.get("other_count",  0)
                b_tot = b_vp + b_np + b_oth
                b_vp_pct = b_vp / b_tot * 100 if b_tot > 0 else 0.0
                b_np_pct = b_np / b_tot * 100 if b_tot > 0 else 0.0
                writer.writerow([
                    lang,
                    f"{b_vp_pct:.1f}", f"{b_np_pct:.1f}",
                    f"{s['vp_rate']:.1f}", f"{s['np_rate']:.1f}",
                    f"{s['avg_margin']:.4f}",
                    s["parse_fails"],
                    s["verified_total"],
                ])
        print(f"  Table 2 CSV       → {table2_path}")


# Main
def main(baseline_results=None):
    print("STEP 4 & 5: Test Set Evaluation")

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

        print(f"[{lang}] Evaluating {len(sentences)} sentences")
        records = evaluate_language(lang, sentences, grammar)
        summary = summarize(lang, records)

        all_records[lang] = records
        summaries.append(summary)

    # Print Table 2 to stdout
    print("TABLE 2 — Treebank Baseline vs. PCFG Parser (30-sentence test set)")
    print("Margin = mean(vp_score − np_score) per language")

    if baseline_results:
        header = (f"{'Language':<12} {'Base VP%':>9} {'Base NP%':>9} "
                  f"{'Parser VP%':>11} {'Parser NP%':>11} {'Avg Margin':>11}")
        print(header)
        print("-" * 72)
        for s in summaries:
            lang  = s["language"]
            base  = baseline_results.get(lang, {})
            b_vp  = base.get("vp_count",   0)
            b_np  = base.get("np_count",    0)
            b_oth = base.get("other_count", 0)
            b_tot = b_vp + b_np + b_oth
            b_vp_pct = b_vp / b_tot * 100 if b_tot > 0 else 0.0
            b_np_pct = b_np / b_tot * 100 if b_tot > 0 else 0.0
            print(f"{lang:<12} {b_vp_pct:>8.1f}% {b_np_pct:>8.1f}% "
                  f"{s['vp_rate']:>10.1f}% {s['np_rate']:>10.1f}% "
                  f"{s['avg_margin']:>11.4f}")
    else:
        # No baseline passed — print parser-only table
        print(f"{'Language':<12} {'Parser VP%':>11} {'Parser NP%':>11} {'Avg Margin':>11}")
        for s in summaries:
            print(f"{s['language']:<12} {s['vp_rate']:>10.1f}% "
                  f"{s['np_rate']:>10.1f}% {s['avg_margin']:>11.4f}")

    print("Note: no gold labels exist for the test set by design.")
    print("Margin > 0 → grammar assigns higher probability to VP-attached parse.")
    print("Margin < 0 → grammar assigns higher probability to NP-attached parse.")

    if summaries:
        save_results(all_records, summaries, baseline_results)

    return all_records, summaries


if __name__ == "__main__":
    main()