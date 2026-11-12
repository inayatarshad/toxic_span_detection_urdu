"""
prepare_data.py — leakage-safe preparation of URTOX for the PeerJ revision.

Steps (in this order, all BEFORE any split is made):
  1. Parse tokens / BIO tags; repair or drop malformed rows (logged).
  2. Normalise tag names to {O, B-TOXIC, I-TOXIC}; repair any I- that follows O.
  3. Exact de-duplication on a normalised text key (NFC, diacritics and
     punctuation removed, whitespace collapsed). One copy is kept.
  4. Near-duplicate grouping with MinHash-LSH over character 5-gram shingles
     (Jaccard >= 0.8, verified exactly). Near-duplicates are NOT removed but are
     placed in the same group, and a whole group always goes to one split.
  5. Group-level stratified 80/10/10 split (strata = post label x has-span),
     fixed seed. No data augmentation is used anywhere, so no augmented
     derivative can cross a split boundary.
  6. Token-level text normalisation that keeps a strict 1:1 token/tag
     alignment (no token is ever added, removed or merged):
        NFC -> remove Arabic-script diacritics and tatweel -> map Arabic
        yeh/kaf code points to their Urdu forms -> URL -> <url>,
        @mention -> <user>.  No transliteration is performed: the text stays in
        its original script.
  7. Write data/urtox_clean.jsonl and data/dataset_statistics.json/.csv.

Usage:  python revision/prepare_data.py --csv URTOX_v2.csv --out outputs/data
"""
import argparse, ast, json, os, re, random, unicodedata, hashlib
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

TAG_MAP = {"O": "O", "B-TOXIC": "B-TOXIC", "I-TOXIC": "I-TOXIC"}
DIACRITICS = re.compile(r"[ً-ٰٟۖ-ۭـ]")  # harakat, superscript alef, Quranic marks, tatweel
URL_RE = re.compile(r"^(https?://|www\.)\S+$", re.I)
MENTION_RE = re.compile(r"^@\w+$")
PUNCT_RE = re.compile(r"[\W_]+", re.UNICODE)


# ----------------------------------------------------------------- parsing
def parse_list(s):
    try:
        v = ast.literal_eval(s)
        return v if isinstance(v, list) else None
    except Exception:
        return None


def parse_tags(s):
    v = parse_list(s)
    if v is not None:
        return v, False
    # malformed literal (e.g. a missing quote): recover the tag tokens
    return re.findall(r"[BI]-\w+|\bO\b", str(s)), True


def norm_tag(t):
    t = str(t).strip().upper()
    return TAG_MAP.get(t, "O" if t == "O" else t)


def repair_bio(tags):
    out, prev = [], "O"
    for t in tags:
        if t == "I-TOXIC" and prev == "O":
            t = "B-TOXIC"
        out.append(t)
        prev = t
    return out


# ----------------------------------------------------------- normalisation
def normalise_token(tok):
    t = unicodedata.normalize("NFC", str(tok))
    if URL_RE.match(t):
        return "<url>"
    if MENTION_RE.match(t):
        return "<user>"
    t2 = DIACRITICS.sub("", t)
    t2 = t2.replace("ي", "ی").replace("ك", "ک")  # Arabic yeh/kaf -> Urdu
    return t2 if t2.strip() else t  # never produce an empty token


def text_key(text):
    t = unicodedata.normalize("NFC", str(text))
    t = DIACRITICS.sub("", t).lower()
    t = PUNCT_RE.sub(" ", t)
    return " ".join(t.split())


# ------------------------------------------------------------ near-dup LSH
def shingles(key, k=5):
    s = key.replace(" ", "")
    if len(s) <= k:
        return {s}
    return {s[i:i + k] for i in range(len(s) - k + 1)}


def minhash_groups(keys, threshold=0.8, num_perm=64, bands=16, seed=13):
    rows = num_perm // bands
    rng = np.random.RandomState(seed)
    P = (1 << 61) - 1
    a = rng.randint(1, 1 << 31, size=num_perm).astype(np.uint64)
    b = rng.randint(0, 1 << 31, size=num_perm).astype(np.uint64)
    sh = [shingles(k) for k in keys]
    sigs = np.empty((len(keys), num_perm), dtype=np.uint64)
    for i, S in enumerate(sh):
        hv = np.array([int(hashlib.md5(x.encode("utf-8")).hexdigest()[:15], 16) for x in S], dtype=np.uint64)
        sigs[i] = ((np.outer(hv, a) + b) % np.uint64(P)).min(axis=0)
    parent = list(range(len(keys)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    checked, n_pairs = set(), 0
    for bnd in range(bands):
        buckets = defaultdict(list)
        for i in range(len(keys)):
            buckets[sigs[i, bnd * rows:(bnd + 1) * rows].tobytes()].append(i)
        for idx in buckets.values():
            if len(idx) < 2:
                continue
            for x in range(len(idx)):
                for y in range(x + 1, len(idx)):
                    i, j = idx[x], idx[y]
                    if (i, j) in checked:
                        continue
                    checked.add((i, j))
                    A, B = sh[i], sh[j]
                    if len(A & B) / max(1, len(A | B)) >= threshold:
                        n_pairs += 1
                        ri, rj = find(i), find(j)
                        if ri != rj:
                            parent[rj] = ri
    return [find(i) for i in range(len(keys))], n_pairs


# ------------------------------------------------------------------ splits
def group_stratified_split(groups, strata, fracs=(0.8, 0.1, 0.1), seed=2026):
    """Assign whole groups to splits, balancing each stratum separately."""
    g2idx = defaultdict(list)
    for i, g in enumerate(groups):
        g2idx[g].append(i)
    # a group's stratum = majority stratum of its members
    g_stratum = {g: Counter(strata[i] for i in idx).most_common(1)[0][0] for g, idx in g2idx.items()}
    rnd = random.Random(seed)
    split = [None] * len(groups)
    names = ["train", "validation", "test"]
    for s in sorted(set(g_stratum.values())):
        gs = sorted([g for g in g2idx if g_stratum[g] == s])
        rnd.shuffle(gs)
        total = sum(len(g2idx[g]) for g in gs)
        targets = [f * total for f in fracs]
        filled = [0, 0, 0]
        for g in gs:
            # put the group in the split furthest below its target
            k = max(range(3), key=lambda j: (targets[j] - filled[j]) / max(targets[j], 1e-9))
            for i in g2idx[g]:
                split[i] = names[k]
            filled[k] += len(g2idx[g])
    return split


# --------------------------------------------------------------- statistics
def spans_of(tags):
    spans, start = [], None
    for i, t in enumerate(tags + ["O"]):
        if t == "B-TOXIC" or (t == "I-TOXIC" and start is None):
            if start is not None:
                spans.append((start, i))
            start = i
        elif t == "O":
            if start is not None:
                spans.append((start, i))
            start = None
    return spans


def describe(records):
    L = np.array([len(r["tokens"]) for r in records])
    sp = [spans_of(r["tags"]) for r in records]
    span_len = np.array([e - s for S in sp for s, e in S]) if any(sp) else np.array([0])
    tagged = [i for i, S in enumerate(sp) if S]
    tox_ratio = np.array([sum(t != "O" for t in records[i]["tags"]) / len(records[i]["tags"]) for i in tagged])
    tagc = Counter(t for r in records for t in r["tags"])
    tot = sum(tagc.values())
    return {
        "posts": len(records),
        "posts_label_toxic": sum(r["label"] == "toxic" for r in records),
        "posts_label_non_toxic": sum(r["label"] != "toxic" for r in records),
        "posts_with_toxic_span": len(tagged),
        "posts_multi_span": int(sum(len(S) > 1 for S in sp)),
        "share_multi_span_among_span_posts": round(sum(len(S) > 1 for S in sp) / max(1, len(tagged)), 4),
        "tokens_total": int(tot),
        "tag_counts": dict(tagc),
        "tag_share": {k: round(v / tot, 4) for k, v in tagc.items()},
        "post_len_tokens": {"mean": round(float(L.mean()), 2), "median": float(np.median(L)), "sd": round(float(L.std()), 2),
                             "min": int(L.min()), "p95": float(np.percentile(L, 95)), "max": int(L.max())},
        "span_len_tokens": {"n_spans": int(len(span_len)), "mean": round(float(span_len.mean()), 2),
                             "median": float(np.median(span_len)), "min": int(span_len.min()), "max": int(span_len.max())},
        "toxic_token_ratio_in_span_posts": {"mean": round(float(tox_ratio.mean()), 4) if len(tox_ratio) else 0,
                                             "median": round(float(np.median(tox_ratio)), 4) if len(tox_ratio) else 0},
        "sub_label_counts": dict(Counter(r["sub_label"] for r in records)),
        "sub_label_x_has_span": {k: dict(v) for k, v in _xtab(records, sp).items()},
        "label_x_has_span": {k: dict(v) for k, v in _xtab(records, sp, key="label").items()},
    }


def _xtab(records, sp, key="sub_label"):
    d = defaultdict(Counter)
    for r, S in zip(records, sp):
        d[r[key]]["with_span" if S else "no_span"] += 1
    return d


def script_profile(text):
    lat = len(re.findall(r"[A-Za-z]", text))
    ara = len(re.findall(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]", text))
    if lat > ara:
        return "roman"
    if lat >= 3 and ara >= 3:
        return "mixed"
    return "nastaliq"


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="URTOX_v2.csv")
    ap.add_argument("--out", default="outputs/data")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--near_dup_threshold", type=float, default=0.8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    log = Counter()

    df = pd.read_csv(args.csv)
    log["rows_in_csv"] = len(df)
    recs = []
    for _, row in df.iterrows():
        toks = parse_list(row["tokens"])
        tags, repaired = parse_tags(row["BIO_tags"])
        if toks is None:
            log["dropped_unparseable_tokens"] += 1
            continue
        if repaired:
            log["bio_literal_repaired"] += 1
        tags = [norm_tag(t) for t in tags]
        if len(toks) != len(tags) or len(toks) == 0:
            log["dropped_token_tag_length_mismatch"] += 1
            continue
        if any(t not in TAG_MAP for t in tags):
            log["dropped_unknown_tag"] += 1
            continue
        fixed = repair_bio(tags)
        if fixed != tags:
            log["bio_I_after_O_repaired"] += 1
        sub = str(row["sub_label"]).strip().lower()
        recs.append(dict(id=int(row["id"]), text=str(row["text"]), label=str(row["label"]).strip().lower(),
                         sub_label=sub, tokens=[str(t) for t in toks], tags=fixed))
    log["rows_parsed"] = len(recs)
    stats_raw = describe(recs)

    # exact de-duplication on normalised text
    seen, kept, conflicts = {}, [], 0
    for r in recs:
        k = text_key(r["text"])
        if k in seen:
            if seen[k]["tags"] != r["tags"] or seen[k]["label"] != r["label"]:
                conflicts += 1
            log["dropped_exact_duplicate"] += 1
            continue
        seen[k] = r
        kept.append(r)
    log["exact_duplicates_with_conflicting_labels"] = conflicts
    recs = kept

    keys = [text_key(r["text"]) for r in recs]
    groups, n_pairs = minhash_groups(keys, threshold=args.near_dup_threshold)
    gsize = Counter(groups)
    log["near_duplicate_pairs_jaccard>=%.2f" % args.near_dup_threshold] = n_pairs
    log["posts_in_near_duplicate_groups"] = sum(v for v in gsize.values() if v > 1)
    log["near_duplicate_groups"] = sum(1 for v in gsize.values() if v > 1)

    strata = [f"{r['label']}|{int(any(t != 'O' for t in r['tags']))}" for r in recs]
    split = group_stratified_split(groups, strata, seed=args.seed)

    for r, g, s in zip(recs, groups, split):
        r["group"] = int(g)
        r["split"] = s
        r["norm_tokens"] = [normalise_token(t) for t in r["tokens"]]
        r["script"] = script_profile(r["text"])
    log["tokens_changed_by_normalisation"] = sum(a != b for r in recs for a, b in zip(r["tokens"], r["norm_tokens"]))

    # leakage audit: no group, exact key or near-dup crosses splits
    g2s = defaultdict(set)
    for r in recs:
        g2s[r["group"]].add(r["split"])
    assert all(len(v) == 1 for v in g2s.values()), "group crosses splits"
    k2s = defaultdict(set)
    for r, k in zip(recs, keys):
        k2s[k].add(r["split"])
    assert all(len(v) == 1 for v in k2s.values())

    with open(os.path.join(args.out, "urtox_clean.jsonl"), "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    pd.DataFrame([{"id": r["id"], "split": r["split"], "group": r["group"]} for r in recs]).to_csv(
        os.path.join(args.out, "split_assignment.csv"), index=False)

    stats = {"preparation_log": dict(log), "raw_parsed_before_dedup": stats_raw, "clean_all": describe(recs),
             "script_profile": dict(Counter(r["script"] for r in recs))}
    for s in ["train", "validation", "test"]:
        stats[f"clean_{s}"] = describe([r for r in recs if r["split"] == s])
    with open(os.path.join(args.out, "dataset_statistics.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps(stats["preparation_log"], indent=2))
    for s in ["train", "validation", "test"]:
        d = stats[f"clean_{s}"]
        print(s, d["posts"], "span-posts", d["posts_with_toxic_span"], "tags", d["tag_share"])


if __name__ == "__main__":
    main()
