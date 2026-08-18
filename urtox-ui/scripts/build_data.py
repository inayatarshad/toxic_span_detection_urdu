"""Precompute URTOX statistics and a stratified explorer sample.

Reads the released URTOX_v2.csv and writes two JSON files consumed by the site:

  src/data/stats.json    aggregate statistics over all 14,337 records
  public/data/sample.json  stratified sample used by the dataset explorer

Every number the site displays for the dataset comes from this script, so the
figures on the page can always be regenerated from the released CSV.

Usage:  py scripts/build_data.py
"""

import ast
import collections
import csv
import json
import os
import random
import re
import sys

csv.field_size_limit(10 ** 9)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
CSV_PATH = os.path.join(REPO, "URTOX_v2.csv")
AUDIO_CSV_PATH = os.path.join(REPO, "urdu_toxic_audio_dataset.csv")
STATS_OUT = os.path.join(HERE, os.pardir, "src", "data", "stats.json")
SAMPLE_OUT = os.path.join(HERE, os.pardir, "public", "data", "sample.json")

SAMPLE_SIZE = 2000
SEED = 42

# sub_label values arrive with stray trailing whitespace on a handful of rows
SUB_LABEL_ORDER = ["normal", "offensive", "hate", "insult", "slur", "threat"]


def parse_list(raw):
    """Parse a stringified Python list, falling back to quoted-item extraction.

    A small number of BIO_tags cells have an unterminated quote, so literal_eval
    raises. Recovering the quoted items keeps those rows in the statistics
    instead of silently dropping them.
    """
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    try:
        value = ast.literal_eval(raw)
        return value if isinstance(value, list) else None
    except (ValueError, SyntaxError):
        items = re.findall(r"'([^']*)'", raw)
        return items or None


def spans_from_bio(bio):
    """Return [start, end_exclusive] token index pairs for each B/I-Toxic run."""
    spans = []
    start = None
    for i, tag in enumerate(bio):
        if tag == "B-Toxic":
            if start is not None:
                spans.append([start, i])
            start = i
        elif tag == "I-Toxic":
            if start is None:
                start = i
        else:
            if start is not None:
                spans.append([start, i])
                start = None
    if start is not None:
        spans.append([start, len(bio)])
    return spans


def bucket(n, edges, labels):
    for edge, label in zip(edges, labels):
        if n <= edge:
            return label
    return labels[-1]


def main():
    with open(CSV_PATH, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        columns = reader.fieldnames
        rows = list(reader)

    total = len(rows)

    label_counts = collections.Counter()
    sub_label_counts = collections.Counter()
    bio_counts = collections.Counter()
    span_length_counts = collections.Counter()
    spans_per_record = collections.Counter()
    token_buckets = collections.Counter()
    toxic_phrases = collections.Counter()

    token_lengths = []
    char_lengths = []
    total_spans = 0
    total_span_tokens = 0

    length_mismatch = 0
    toxic_without_span = 0
    non_toxic_with_span = 0
    toxic_list_populated = 0
    latin_run = 0
    with_digits = 0
    with_emoji = 0

    latin_re = re.compile(r"[A-Za-z]{3,}")
    digit_re = re.compile(r"[0-9]")
    emoji_re = re.compile(r"[\U0001F300-\U0001FAFF☀-➿]")

    records = []

    for row in rows:
        label = row["label"].strip()
        sub_label = row["sub_label"].strip()
        text = row["text"]

        label_counts[label] += 1
        sub_label_counts[sub_label] += 1

        tokens = parse_list(row["tokens"]) or []
        bio = parse_list(row["BIO_tags"]) or []
        # keep only the three valid tags; a couple of rows carry parse debris
        bio = [t for t in bio if t in ("O", "B-Toxic", "I-Toxic")]

        if tokens and bio and len(tokens) != len(bio):
            length_mismatch += 1

        bio_counts.update(bio)
        spans = spans_from_bio(bio)
        total_spans += len(spans)
        spans_per_record[len(spans)] += 1
        for start, end in spans:
            span_length_counts[end - start] += 1
            total_span_tokens += end - start

        if label == "toxic" and not spans:
            toxic_without_span += 1
        if label == "non_toxic" and spans:
            non_toxic_with_span += 1

        token_lengths.append(len(tokens))
        char_lengths.append(len(text))
        token_buckets[
            bucket(len(tokens), [10, 20, 30, 50, 100], ["1-10", "11-20", "21-30", "31-50", "51-100", "100+"])
        ] += 1

        phrases = parse_list(row["toxic_list"])
        flat = []
        if phrases:
            for item in phrases:
                if isinstance(item, list):
                    flat.extend(item)
                else:
                    flat.append(item)
        flat = [p.strip() for p in flat if isinstance(p, str) and p.strip() not in ("", "[]")]
        if flat:
            toxic_list_populated += 1
            toxic_phrases.update(flat)

        if latin_re.search(text):
            latin_run += 1
        if digit_re.search(text):
            with_digits += 1
        if emoji_re.search(text):
            with_emoji += 1

        # Only rows whose tokens and tags line up can be rendered span-accurately
        if tokens and len(tokens) == len(bio):
            records.append(
                {
                    "id": int(row["id"]),
                    "text": text,
                    "label": label,
                    "sub": sub_label if sub_label in SUB_LABEL_ORDER else "normal",
                    "toks": " ".join(tokens),
                    "spans": spans,
                }
            )

    # Stratified explorer sample: proportional across sub_label, deterministic.
    random.seed(SEED)
    by_sub = collections.defaultdict(list)
    for rec in records:
        by_sub[rec["sub"]].append(rec)

    sample = []
    for sub, bucket_rows in by_sub.items():
        share = len(bucket_rows) / len(records)
        take = max(12, round(SAMPLE_SIZE * share))
        take = min(take, len(bucket_rows))
        sample.extend(random.sample(bucket_rows, take))
    sample.sort(key=lambda r: r["id"])

    span_length_hist = [
        {"length": length, "count": span_length_counts[length]}
        for length in sorted(span_length_counts)
        if length <= 8
    ]
    span_length_hist.append(
        {"length": 9, "count": sum(v for k, v in span_length_counts.items() if k > 8), "plus": True}
    )

    audio_rows = 0
    audio_columns = []
    if os.path.exists(AUDIO_CSV_PATH):
        with open(AUDIO_CSV_PATH, encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            audio_columns = reader.fieldnames
            audio_rows = sum(1 for _ in reader)

    token_lengths.sort()
    char_lengths.sort()

    stats = {
        "generatedFrom": "URTOX_v2.csv",
        "records": total,
        "columns": columns,
        "labels": [
            {"key": "toxic", "count": label_counts["toxic"]},
            {"key": "non_toxic", "count": label_counts["non_toxic"]},
        ],
        "subLabels": [
            {
                "key": key,
                "count": sum(v for k, v in sub_label_counts.items() if k.strip() == key),
            }
            for key in SUB_LABEL_ORDER
        ],
        "bio": [
            {"key": "O", "count": bio_counts["O"]},
            {"key": "B-Toxic", "count": bio_counts["B-Toxic"]},
            {"key": "I-Toxic", "count": bio_counts["I-Toxic"]},
        ],
        "totalTokens": sum(bio_counts.values()),
        "totalSpans": total_spans,
        "meanSpanTokens": round(total_span_tokens / total_spans, 2),
        "spanLengthHist": span_length_hist,
        "spansPerRecord": [
            {"spans": k, "count": v} for k, v in sorted(spans_per_record.items()) if k <= 5
        ]
        + [{"spans": 6, "count": sum(v for k, v in spans_per_record.items() if k > 5), "plus": True}],
        "tokenLength": {
            "min": token_lengths[0],
            "max": token_lengths[-1],
            "mean": round(sum(token_lengths) / total, 2),
            "median": token_lengths[total // 2],
        },
        "charLength": {
            "min": char_lengths[0],
            "max": char_lengths[-1],
            "mean": round(sum(char_lengths) / total, 2),
            "median": char_lengths[total // 2],
        },
        "tokenBuckets": [
            {"key": key, "count": token_buckets[key]}
            for key in ["1-10", "11-20", "21-30", "31-50", "51-100", "100+"]
        ],
        "distinctToxicPhrases": len(toxic_phrases),
        "topToxicPhrases": [
            {"phrase": p, "count": c} for p, c in toxic_phrases.most_common(24)
        ],
        "scriptFeatures": {
            "latinRun": latin_run,
            "digits": with_digits,
            "emoji": with_emoji,
        },
        "quality": {
            "toxicWithoutSpan": toxic_without_span,
            "nonToxicWithSpan": non_toxic_with_span,
            "tokenTagLengthMismatch": length_mismatch,
            "toxicListPopulated": toxic_list_populated,
            "renderableRecords": len(records),
        },
        "audio": {
            "rows": audio_rows,
            "columns": audio_columns,
        },
        "sample": {"size": len(sample), "seed": SEED, "strategy": "stratified by sub_label"},
    }

    os.makedirs(os.path.dirname(STATS_OUT), exist_ok=True)
    os.makedirs(os.path.dirname(SAMPLE_OUT), exist_ok=True)

    with open(STATS_OUT, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)
    with open(SAMPLE_OUT, "w", encoding="utf-8") as fh:
        json.dump(sample, fh, ensure_ascii=False, separators=(",", ":"))

    print("records          ", total)
    print("renderable       ", len(records))
    print("explorer sample  ", len(sample))
    print("stats.json       ", os.path.getsize(STATS_OUT) // 1024, "KB")
    print("sample.json      ", os.path.getsize(SAMPLE_OUT) // 1024, "KB")


if __name__ == "__main__":
    main()
