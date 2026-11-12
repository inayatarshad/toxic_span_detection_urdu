"""
analyze.py — builds every results table of the revised manuscript from the saved
per-post test predictions (no GPU needed).

Metrics (all on the held-out test split, at the level of the original URTOX
word tokens):
  token_F1      micro precision/recall/F1 over toxic tokens (B-TOXIC or I-TOXIC
                counts as toxic).                                   PRIMARY
  post_F1_all   SemEval-2021-style mean of per-post F1 over toxic-token sets,
                with the convention F1=1 when gold and prediction are both empty.
  post_F1_gold  the same mean restricted to posts whose gold set is non-empty
                (removes the effect of the empty/empty convention).
  span_F1       exact-match span F1 (a predicted span counts only if both
                boundaries match a gold span).
  char_F1       SemEval-style per-post character-offset F1, after mapping each
                token to its character offsets in the raw text (coverage reported).
  invalid_BIO   share of predicted spans that begin with I-TOXIC.

Statistics:
  * mean, SD and 95% t-interval across the five seeds;
  * 95% paired bootstrap CI (10,000 resamples of test posts) of the difference in
    seed-averaged token-F1 between two systems;
  * two-sided paired approximate-randomisation (permutation) test at post level
    (10,000 permutations), Holm-Bonferroni corrected within each family.

Outputs: <out>/results/*.csv, results_summary.md
"""
import argparse, glob, json, math, os, re
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from scipy import stats

PRETTY = {"xlmr_crf": "XLM-R + CRF (MUTEX)", "xlmr": "XLM-R (softmax)", "mbert": "mBERT (softmax)",
          "bilstm_crf": "BiLSTM-CRF", "xlmr_large_crf": "XLM-R-large + CRF"}


# ------------------------------------------------------------------ helpers
def spans(tags):
    out, start = [], None
    for i, t in enumerate(list(tags) + ["O"]):
        if t == "B-TOXIC" or (t == "I-TOXIC" and start is None):
            if start is not None:
                out.append((start, i))
            start = i
        elif t == "O" and start is not None:
            out.append((start, i)); start = None
    return out


def tox_set(tags):
    return {i for i, t in enumerate(tags) if t != "O"}


def f1_sets(g, p):
    if not g and not p:
        return 1.0
    if not g or not p:
        return 0.0
    tp = len(g & p)
    if tp == 0:
        return 0.0
    P, R = tp / len(p), tp / len(g)
    return 2 * P * R / (P + R)


def prf(tp, fp, fn):
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    return P, R, (2 * P * R / (P + R) if P + R else 0.0)


def char_offsets(text, tokens):
    """Greedy left-to-right search of each token in the raw text."""
    offs, cur, found = [], 0, 0
    for tok in tokens:
        cands = [tok, tok.replace('""', '"')]
        pos = -1
        for c in cands:
            pos = text.find(c, cur)
            if pos != -1 and pos - cur <= 40:
                tok = c
                break
            pos = -1
        if pos == -1:
            offs.append(None)
        else:
            offs.append((pos, pos + len(tok))); cur = pos + len(tok); found += 1
    return offs, found / max(1, len(tokens))


def per_post_counts(gold, pred):
    g, p = tox_set(gold), tox_set(pred)
    return len(g & p), len(p - g), len(g - p)


# ------------------------------------------------------------------ loading
def load_runs(out):
    runs = defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(out, "runs", "*", "seed*", "test_predictions.jsonl"))):
        if not os.path.exists(os.path.join(os.path.dirname(f), "metrics.json")):
            continue
        run = f.split(os.sep)[-3]
        seed = int(f.split(os.sep)[-2].replace("seed", ""))
        runs[run][seed] = {json.loads(l)["id"]: json.loads(l) for l in open(f, encoding="utf-8")}
    return runs


def run_metrics(preds, meta):
    tp = fp = fn = 0
    post_all, post_gold, sp_tp, sp_p, sp_g, inv, nspan = [], [], 0, 0, 0, 0, 0
    char_f1, cov = [], []
    for pid, x in preds.items():
        g, p = x["gold"], x["pred"]
        a, b, c = per_post_counts(g, p)
        tp += a; fp += b; fn += c
        f = f1_sets(tox_set(g), tox_set(p))
        post_all.append(f)
        if tox_set(g):
            post_gold.append(f)
        gs, ps = set(spans(g)), spans(p)
        sp_tp += len(gs & set(ps)); sp_p += len(ps); sp_g += len(gs)
        nspan += len(ps)
        inv += sum(1 for s, e in ps if p[s] == "I-TOXIC")
        offs, cv = meta[pid]["offsets"]
        cov.append(cv)
        gc = {k for i in tox_set(g) if offs[i] for k in range(*offs[i])}
        pc = {k for i in tox_set(p) if offs[i] for k in range(*offs[i])}
        char_f1.append(f1_sets(gc, pc))
    P, R, F = prf(tp, fp, fn)
    sP, sR, sF = prf(sp_tp, sp_p - sp_tp, sp_g - sp_tp)
    return {"token_P": P, "token_R": R, "token_F1": F, "post_F1_all": float(np.mean(post_all)),
            "post_F1_gold": float(np.mean(post_gold)), "span_P": sP, "span_R": sR, "span_F1": sF,
            "char_F1": float(np.mean(char_f1)), "char_alignment_coverage": float(np.mean(cov)),
            "invalid_BIO_span_share": inv / max(1, nspan)}


def mean_sd_ci(v):
    v = np.asarray(v, float)
    m, sd = v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)
    h = stats.t.ppf(0.975, len(v) - 1) * sd / math.sqrt(len(v)) if len(v) > 1 else 0.0
    return m, sd, m - h, m + h


# --------------------------------------------------------------- statistics
def span_counts(gold, pred):
    g, p = set(spans(gold)), set(spans(pred))
    return len(g & p), len(p - g), len(g - p)


def count_matrix(runs_by_seed, ids, fn=per_post_counts):
    seeds = sorted(runs_by_seed)
    M = np.zeros((len(seeds), len(ids), 3))
    for si, s in enumerate(seeds):
        for ni, pid in enumerate(ids):
            x = runs_by_seed[s][pid]
            M[si, ni] = fn(x["gold"], x["pred"])
    return M  # (seeds, posts, [tp, fp, fn])


def f1_from(tp, fp, fn):
    d = 2 * tp + fp + fn
    return np.where(d > 0, 2 * tp / np.maximum(d, 1e-12), 0.0)


def compare(A, B, R=10000, seed=0):
    """A, B: (seeds, N, 3). Statistic = mean over seeds of micro token-F1."""
    rng = np.random.default_rng(seed)
    S = min(A.shape[0], B.shape[0])
    A, B = A[:S], B[:S]
    N = A.shape[1]
    obs = f1_from(*A.sum(1).T).mean() - f1_from(*B.sum(1).T).mean()
    # paired bootstrap over posts
    idx = rng.integers(0, N, size=(R, N))
    W = np.zeros((R, N)); np.add.at(W, (np.repeat(np.arange(R), N), idx.ravel()), 1)
    tA, tB = W @ A.transpose(1, 0, 2).reshape(N, -1), W @ B.transpose(1, 0, 2).reshape(N, -1)
    tA, tB = tA.reshape(R, S, 3), tB.reshape(R, S, 3)
    boot = f1_from(tA[..., 0], tA[..., 1], tA[..., 2]).mean(1) - f1_from(tB[..., 0], tB[..., 1], tB[..., 2]).mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # approximate randomisation: swap system outputs per post (all seeds jointly)
    Sw = rng.random((R, N)) < 0.5
    D = (B - A).transpose(1, 0, 2).reshape(N, -1)
    totA = A.sum(1).reshape(-1) + Sw @ D
    totB = B.sum(1).reshape(-1) - Sw @ D
    totA, totB = totA.reshape(R, S, 3), totB.reshape(R, S, 3)
    perm = f1_from(totA[..., 0], totA[..., 1], totA[..., 2]).mean(1) - f1_from(totB[..., 0], totB[..., 1], totB[..., 2]).mean(1)
    p = (np.sum(np.abs(perm) >= abs(obs) - 1e-12) + 1) / (R + 1)
    return obs, lo, hi, p


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps)); run = 0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


# ------------------------------------------------------------ error analysis
def error_buckets(preds):
    c = Counter()
    for x in preds.values():
        g, p = spans(x["gold"]), spans(x["pred"])
        gs, ps = set(g), set(p)
        for s in g:
            if s in ps:
                c["gold_span_exact"] += 1
            else:
                ov = [q for q in p if q[0] < s[1] and s[0] < q[1]]
                if not ov:
                    c["gold_span_missed"] += 1
                else:
                    q = ov[0]
                    if q[0] >= s[0] and q[1] <= s[1]:
                        c["boundary_pred_too_short"] += 1
                    elif q[0] <= s[0] and q[1] >= s[1]:
                        c["boundary_pred_too_long"] += 1
                    else:
                        c["boundary_shifted"] += 1
                    if len(ov) > 1:
                        c["gold_span_split_into_several"] += 1
        for q in p:
            if not any(q[0] < s[1] and s[0] < q[1] for s in g):
                c["spurious_span_in_post_without_gold" if not g else "spurious_span_in_post_with_gold"] += 1
    return c


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/data/urtox_clean.jsonl")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--perm", type=int, default=10000)
    args = ap.parse_args()
    res = os.path.join(args.out, "results"); os.makedirs(res, exist_ok=True)

    recs = {json.loads(l)["id"]: json.loads(l) for l in open(args.data, encoding="utf-8")}
    for r in recs.values():
        r["offsets"] = char_offsets(r["text"], r["tokens"])
    runs = load_runs(args.out)
    if not runs:
        print("no finished runs"); return
    test_ids = sorted(next(iter(next(iter(runs.values())).values())).keys())

    # ---- per-run metrics + aggregate
    rows = []
    for run, by_seed in runs.items():
        for s, preds in sorted(by_seed.items()):
            m = run_metrics(preds, recs); m.update(run=run, seed=s); rows.append(m)
    per_run = pd.DataFrame(rows); per_run.to_csv(os.path.join(res, "per_seed_metrics.csv"), index=False)
    agg = []
    metrics = ["token_P", "token_R", "token_F1", "post_F1_all", "post_F1_gold", "span_F1", "char_F1", "invalid_BIO_span_share"]
    for run, g in per_run.groupby("run"):
        row = {"run": run, "n_seeds": len(g)}
        for k in metrics:
            m, sd, lo, hi = mean_sd_ci(g[k])
            row[f"{k}_mean"], row[f"{k}_sd"], row[f"{k}_ci_lo"], row[f"{k}_ci_hi"] = m, sd, lo, hi
        row["char_alignment_coverage"] = g["char_alignment_coverage"].mean()
        agg.append(row)
    agg = pd.DataFrame(agg); agg.to_csv(os.path.join(res, "main_results.csv"), index=False)

    # ---- comparisons
    mats = {"token": {run: count_matrix(by_seed, test_ids) for run, by_seed in runs.items()},
            "span": {run: count_matrix(by_seed, test_ids, span_counts) for run, by_seed in runs.items()}}
    ref = "xlmr_crf__main"
    have = mats["token"]
    fams = {"model_comparison": [(ref, r) for r in ["xlmr__main", "mbert__main", "bilstm_crf__main", "xlmr_large_crf__main"] if r in have],
            "ablation_preprocessing": [(ref, r) for r in ["xlmr_crf__no_preprocessing"] if r in have],
            "learning_curve": [(ref, r) for r in sorted(have) if r.startswith("xlmr_crf__frac")]}
    comp = []
    for metric, col in [("token", "token_F1"), ("span", "span_F1")]:
        for fam, pairs in fams.items():
            out = []
            for a, b in pairs:
                if a not in have:
                    continue
                d, lo, hi, p = compare(mats[metric][a], mats[metric][b], R=args.perm)
                sa = per_run[per_run.run == a].set_index("seed")[col]
                sb = per_run[per_run.run == b].set_index("seed")[col]
                common = sorted(set(sa.index) & set(sb.index))
                dm, dsd, dlo, dhi = mean_sd_ci((sa[common] - sb[common]).values)
                out.append({"metric": metric, "family": fam, "A": a, "B": b, "delta_F1_pts": 100 * d,
                            "boot_ci95_lo_pts": 100 * lo, "boot_ci95_hi_pts": 100 * hi, "perm_p": p,
                            "seed_level_delta_mean_pts": 100 * dm, "seed_level_delta_ci95_lo_pts": 100 * dlo,
                            "seed_level_delta_ci95_hi_pts": 100 * dhi,
                            "seeds_A_better": int(sum(sa[common] > sb[common])), "n_seeds": len(common)})
            if out:
                adj = holm(np.array([o["perm_p"] for o in out]))
                for o, q in zip(out, adj):
                    o["perm_p_holm"] = q
                comp += out
    pd.DataFrame(comp).to_csv(os.path.join(res, "significance_tests.csv"), index=False)

    # ---- MUTEX breakdowns (pooled over seeds)
    if ref in runs:
        cat_rows, script_rows = [], []
        for key, outrows in [("sub_label", cat_rows), ("script", script_rows)]:
            groups = defaultdict(list)
            for pid in test_ids:
                groups[recs[pid][key]].append(pid)
            for grp, ids in sorted(groups.items()):
                f1s, pf = [], []
                for s, preds in runs[ref].items():
                    t = np.sum([per_post_counts(preds[i]["gold"], preds[i]["pred"]) for i in ids], 0)
                    f1s.append(prf(*t)[2])
                    pf.append(np.mean([f1_sets(tox_set(preds[i]["gold"]), tox_set(preds[i]["pred"])) for i in ids]))
                gold_tok = sum(len(tox_set(runs[ref][next(iter(runs[ref]))][i]["gold"])) for i in ids)
                m, sd, _, _ = mean_sd_ci(f1s)
                outrows.append({key: grp, "test_posts": len(ids),
                                "posts_with_gold_span": sum(bool(tox_set(recs[i]["tags"])) for i in ids),
                                "gold_toxic_tokens": gold_tok, "token_F1_mean": m if gold_tok else float("nan"),
                                "token_F1_sd": sd if gold_tok else float("nan"), "post_F1_all_mean": float(np.mean(pf))})
        pd.DataFrame(cat_rows).to_csv(os.path.join(res, "mutex_by_category.csv"), index=False)
        pd.DataFrame(script_rows).to_csv(os.path.join(res, "mutex_by_script.csv"), index=False)
        # length buckets
        lb = []
        for lo_, hi_ in [(1, 10), (11, 20), (21, 40), (41, 80), (81, 10000)]:
            ids = [i for i in test_ids if lo_ <= len(recs[i]["tokens"]) <= hi_]
            f1s = [prf(*np.sum([per_post_counts(p[i]["gold"], p[i]["pred"]) for i in ids], 0))[2] for p in runs[ref].values()]
            m, sd, _, _ = mean_sd_ci(f1s)
            lb.append({"post_length_tokens": f"{lo_}-{hi_ if hi_ < 10000 else 'max'}", "test_posts": len(ids), "token_F1_mean": m, "token_F1_sd": sd})
        pd.DataFrame(lb).to_csv(os.path.join(res, "mutex_by_length.csv"), index=False)
        # error analysis, pooled over seeds
        eb = Counter()
        for preds in runs[ref].values():
            eb += error_buckets(preds)
        n = len(runs[ref])
        ea = pd.DataFrame([{"category": k, "count_mean_per_seed": v / n} for k, v in sorted(eb.items())])
        ea.to_csv(os.path.join(res, "mutex_error_analysis.csv"), index=False)
        # truncation audit
        trunc = sum(x["truncated_words"] for p in runs[ref].values() for x in p.values()) / n
        json.dump({"mean_truncated_words_per_seed_on_test": trunc}, open(os.path.join(res, "truncation_audit.json"), "w"))
        # learning curve (variants frac*)
        lc = []
        for run in runs:
            mm = re.match(r"xlmr_crf__frac([0-9.]+)$", run)
            if mm:
                g = per_run[per_run.run == run]
                m, sd, lo, hi = mean_sd_ci(g.token_F1)
                lc.append({"train_fraction": float(mm.group(1)), "n_seeds": len(g), "token_F1_mean": m, "token_F1_sd": sd, "ci_lo": lo, "ci_hi": hi})
        g = per_run[per_run.run == ref]
        m, sd, lo, hi = mean_sd_ci(g.token_F1)
        lc.append({"train_fraction": 1.0, "n_seeds": len(g), "token_F1_mean": m, "token_F1_sd": sd, "ci_lo": lo, "ci_hi": hi})
        pd.DataFrame(lc).sort_values("train_fraction").to_csv(os.path.join(res, "learning_curve.csv"), index=False)

    # ---- summary
    with open(os.path.join(args.out, "results_summary.md"), "w") as f:
        f.write("# Results summary (auto-generated by revision/analyze.py)\n\n## Test-set metrics (mean ± SD over seeds)\n\n")
        f.write("| run | seeds | token P | token R | token F1 [95% CI] | post F1 (all) | post F1 (gold≠∅) | span F1 | char F1 | invalid BIO |\n|---|---|---|---|---|---|---|---|---|---|\n")
        for _, r in agg.sort_values("token_F1_mean", ascending=False).iterrows():
            f.write(f"| {r.run} | {r.n_seeds} | {100*r.token_P_mean:.1f}±{100*r.token_P_sd:.1f} | {100*r.token_R_mean:.1f}±{100*r.token_R_sd:.1f} | "
                    f"{100*r.token_F1_mean:.1f}±{100*r.token_F1_sd:.1f} [{100*r.token_F1_ci_lo:.1f}, {100*r.token_F1_ci_hi:.1f}] | "
                    f"{100*r.post_F1_all_mean:.1f} | {100*r.post_F1_gold_mean:.1f} | {100*r.span_F1_mean:.1f} | {100*r.char_F1_mean:.1f} | {100*r.invalid_BIO_span_share_mean:.2f}% |\n")
        f.write("\n## Paired comparisons (F1 points; token-level and exact-span-level)\n\n")
        for o in comp:
            f.write(f"- [{o['metric']}] {o['A']} vs {o['B']}: Δ={o['delta_F1_pts']:+.2f} [{o['boot_ci95_lo_pts']:+.2f}, {o['boot_ci95_hi_pts']:+.2f}], "
                    f"p={o['perm_p']:.4f}, Holm p={o['perm_p_holm']:.4f}, A better in {o['seeds_A_better']}/{o['n_seeds']} seeds\n")
    print(open(os.path.join(args.out, "results_summary.md")).read())


if __name__ == "__main__":
    main()
