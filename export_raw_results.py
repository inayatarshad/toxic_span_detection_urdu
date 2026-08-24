"""
export_raw_results.py — write PeerJ raw-data files from the MUTEX evaluation loop.

Drop this into the repo root and call the two functions from
`URTOX_XLM+CRF_with_improv(2).ipynb` after each evaluation.

Produces:
  raw_data/raw_data_test_predictions.csv   token-level gold vs predicted BIO labels
  raw_data/raw_data_table4.csv             per-seed P/R/F1 for every model
  raw_data/raw_data_statistical_tests.csv  paired t-tests recomputed from the per-seed file
"""

import os
import itertools
import pandas as pd
from scipy import stats

OUT = "raw_data"
os.makedirs(OUT, exist_ok=True)


def log_predictions(post_ids, tokens, gold_tags, pred_tags, model_name, seed):
    """Append token-level gold/predicted labels for one evaluation run.

    post_ids   : list[int]              — URTOX id per test post
    tokens     : list[list[str]]        — tokens per post
    gold_tags  : list[list[str]]        — gold BIO tags per post
    pred_tags  : list[list[str]]        — predicted BIO tags per post
    """
    rows = []
    for pid, toks, gold, pred in zip(post_ids, tokens, gold_tags, pred_tags):
        for idx, (tok, g, p) in enumerate(zip(toks, gold, pred)):
            rows.append(
                dict(model=model_name, seed=seed, post_id=pid, token_index=idx,
                     token=tok, gold_bio=g, predicted_bio=p, correct=int(g == p))
            )
    path = os.path.join(OUT, "raw_data_test_predictions.csv")
    pd.DataFrame(rows).to_csv(
        path, mode="a", index=False, header=not os.path.exists(path), encoding="utf-8"
    )


def per_post_f1(gold, pred):
    g = {i for i, t in enumerate(gold) if t != "O"}
    p = {i for i, t in enumerate(pred) if t != "O"}
    if not g and not p:
        return 1.0
    if not g or not p:
        return 0.0
    tp = len(g & p)
    if tp == 0:
        return 0.0
    prec, rec = tp / len(p), tp / len(g)
    return 2 * prec * rec / (prec + rec)


def build_tables():
    """Recompute per-seed scores and paired t-tests from the logged predictions."""
    df = pd.read_csv(os.path.join(OUT, "raw_data_test_predictions.csv"))

    runs = []
    for (model, seed), grp in df.groupby(["model", "seed"]):
        tp = ((grp.gold_bio != "O") & (grp.predicted_bio != "O")).sum()
        pred_pos = (grp.predicted_bio != "O").sum()
        gold_pos = (grp.gold_bio != "O").sum()
        prec = tp / pred_pos if pred_pos else 0.0
        rec = tp / gold_pos if gold_pos else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        post_f1 = grp.groupby("post_id").apply(
            lambda x: per_post_f1(list(x.gold_bio), list(x.predicted_bio))
        ).mean()
        runs.append(dict(model=model, seed=seed, precision=100 * prec,
                         recall=100 * rec, token_f1=100 * f1,
                         macro_post_f1=100 * post_f1))

    runs = pd.DataFrame(runs).sort_values(["model", "seed"])
    runs.to_csv(os.path.join(OUT, "raw_data_table4.csv"), index=False)

    proposed = "XLM-RoBERTa+CRF"
    tests = []
    for model in runs.model.unique():
        if model == proposed:
            continue
        a = runs[runs.model == proposed].sort_values("seed").token_f1.values
        b = runs[runs.model == model].sort_values("seed").token_f1.values
        t, p = stats.ttest_rel(a, b)
        tests.append(dict(comparison=f"{proposed} vs {model}",
                          test="paired two-sided t-test",
                          n=len(a), df=len(a) - 1, statistic_t=round(t, 3),
                          p_value=round(p, 5),
                          delta_f1_pct=round(a.mean() - b.mean(), 3),
                          alpha=0.05, significant="yes" if p < 0.05 else "no"))
    pd.DataFrame(tests).to_csv(
        os.path.join(OUT, "raw_data_statistical_tests.csv"), index=False)
    print(runs.groupby("model")[["precision", "recall", "token_f1"]].agg(["mean", "std"]))


if __name__ == "__main__":
    build_tables()
