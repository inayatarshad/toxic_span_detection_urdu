"""
explain.py — Integrated-Gradients explanations for MUTEX and their QUANTITATIVE
evaluation (requires the checkpoint saved by train.py --save_model).

Explained quantity: F(x) = sum over words of p(toxic | word), where p(toxic) is
the emission softmax mass on {B-TOXIC, I-TOXIC}. IG is computed in the
input-embedding space against a baseline in which every non-special sub-word
embedding is replaced by the <pad> embedding (m Riemann steps, trapezoid rule).
A word's attribution is the sum of its sub-words' attributions. We report the
completeness gap |sum(attr) - (F(x) - F(x'))| / |F(x) - F(x')|.

Compared attribution methods: IG, last-layer attention (mean over heads and
queries), and random scores.

Plausibility (agreement with human-annotated toxic tokens, test posts that have a
gold span): average precision (AUPRC) of word scores; token-F1 of the top-k words
(k = number of gold toxic words).

Faithfulness (ERASER, DeYoung et al. 2020) on test posts that MUTEX flags as
toxic (max word p(toxic) > 0.5), with post score s(x) = max_w p(toxic | w):
  comprehensiveness@q = s(x) - s(x without top-q% words)
  sufficiency@q       = s(x) - s(only the top-q% words)
for q in {10, 20, 50}% (at least one word); AOPC = mean over q.
Higher comprehensiveness and lower sufficiency = more faithful.

Also writes qualitative examples (true positive, false positive, false
negative, boundary error) with per-word scores for the manuscript.
"""
import argparse, json, os, random, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import TransformerTagger, TransformerCollate, ENCODERS, load  # noqa: E402


def word_index(enc, b, n):
    m = [[] for _ in range(n)]
    for pos, w in enumerate(enc.word_ids(b)):
        if w is not None:
            m[w].append(pos)
    return m


class Explainer:
    def __init__(self, model, tok, dev, max_len=256):
        self.m, self.tok, self.dev, self.max_len = model, tok, dev, max_len
        self.col = TransformerCollate(tok, max_len)
        self.emb = model.enc.get_input_embeddings()

    def encode(self, words):
        enc = self.tok([[self.col.safe(w) for w in words]], is_split_into_words=True, truncation=True,
                       max_length=self.max_len, return_tensors="pt")
        wi = word_index(enc, 0, len(words))
        n = sum(1 for x in wi if x)
        first = torch.tensor([[x[0] for x in wi[:n]]])
        return enc, wi[:n], first

    def word_probs(self, words):
        if not words:
            return np.zeros(0)
        enc, wi, first = self.encode(words)
        with torch.no_grad():
            em = self.m.emissions(enc["input_ids"].to(self.dev), enc["attention_mask"].to(self.dev), first.to(self.dev), None)
        return (1 - em.softmax(-1)[0, :, 0]).cpu().numpy()

    def ig(self, words, steps=32):
        enc, wi, first = self.encode(words)
        ids = enc["input_ids"].to(self.dev); am = enc["attention_mask"].to(self.dev); first = first.to(self.dev)
        x = self.emb(ids).detach()
        special = torch.tensor([[i in self.tok.all_special_ids for i in ids[0].tolist()]], device=self.dev)
        base = torch.where(special.unsqueeze(-1), x, self.emb(torch.full_like(ids, self.tok.pad_token_id)).detach())
        alphas = torch.linspace(0, 1, steps + 1, device=self.dev)
        grads = []
        for chunk in alphas.split(16):
            xi = (base + chunk.view(-1, 1, 1) * (x - base)).requires_grad_(True)
            em = self.m.emissions(None, am.expand(len(chunk), -1), first.expand(len(chunk), -1), None, inputs_embeds=xi)
            F = (1 - em.softmax(-1)[..., 0]).sum()
            g, = torch.autograd.grad(F, xi)
            grads.append(g.detach())
        g = torch.cat(grads)
        avg = (g[:-1] + g[1:]).mean(0) / 2 if steps > 0 else g[0]
        attr_sub = ((x - base)[0] * avg).sum(-1).cpu().numpy()
        with torch.no_grad():
            f_x = (1 - self.m.emissions(None, am, first, None, inputs_embeds=x).softmax(-1)[..., 0]).sum().item()
            f_b = (1 - self.m.emissions(None, am, first, None, inputs_embeds=base).softmax(-1)[..., 0]).sum().item()
        wattr = np.array([attr_sub[p].sum() for p in wi])
        gap = abs(wattr.sum() - (f_x - f_b)) / max(abs(f_x - f_b), 1e-6)
        return wattr, gap

    def attention(self, words):
        enc, wi, first = self.encode(words)
        with torch.no_grad():
            out = self.m.enc(input_ids=enc["input_ids"].to(self.dev), attention_mask=enc["attention_mask"].to(self.dev),
                             output_attentions=True)
        a = out.attentions[-1][0].mean(0).mean(0).cpu().numpy()  # mean over heads and queries -> (L,)
        return np.array([a[p].sum() for p in wi])


def avg_precision(scores, gold):
    order = np.argsort(-scores)
    hits, ap = 0, 0.0
    for r, i in enumerate(order, 1):
        if gold[i]:
            hits += 1; ap += hits / r
    return ap / max(1, gold.sum())


def topk_f1(scores, gold):
    k = int(gold.sum())
    top = set(np.argsort(-scores)[:k])
    g = set(np.where(gold)[0])
    tp = len(top & g)
    return 0.0 if tp == 0 else tp / k


def spans_(tags):
    out, start = [], None
    for i, t in enumerate(list(tags) + ["O"]):
        if t == "B-TOXIC" or (t == "I-TOXIC" and start is None):
            if start is not None:
                out.append((start, i))
            start = i
        elif t == "O" and start is not None:
            out.append((start, i)); start = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/data/urtox_clean.jsonl")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--encoder_override", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = os.path.join(args.out, "runs", "xlmr_crf__main", f"seed{args.seed}")
    name = args.encoder_override or ENCODERS["xlmr_crf"][0]
    tok = AutoTokenizer.from_pretrained(name)
    model = TransformerTagger(name, True)
    model.enc.config._attn_implementation = "eager"   # attention weights for the baseline
    try:
        model.enc.set_attn_implementation("eager")
    except Exception:
        pass
    model.load_state_dict(torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu"))
    model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    ex = Explainer(model, tok, dev)

    recs = {r["id"]: r for r in load(args.data, "norm_tokens") if r["split"] == "test"}
    preds = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(run_dir, "test_predictions.jsonl"), encoding="utf-8")}
    ids = sorted(set(recs) & set(preds))
    if args.limit:
        ids = ids[:args.limit]
    rng = np.random.default_rng(0)
    plaus, faith, gaps, per_post = [], [], [], []
    for n_done, pid in enumerate(ids):
        r = recs[pid]
        words = r["words"]
        enc, wi, _ = ex.encode(words)
        n = len(wi)
        words = words[:n]
        gold = np.array([t != "O" for t in r["tags"][:n]])
        probs = ex.word_probs(words)
        flagged = probs.max() > 0.5 if n else False
        if not gold.any() and not flagged:
            continue
        ig, gap = ex.ig(words, args.steps); gaps.append(gap)
        att = ex.attention(words)
        methods = {"integrated_gradients": ig, "attention": att}
        for k in range(3):
            methods[f"random_{k}"] = rng.random(n)
        if gold.any():
            row = {"id": pid}
            for mname, sc in methods.items():
                row[f"{mname}_AP"] = avg_precision(sc, gold)
                row[f"{mname}_topkF1"] = topk_f1(sc, gold)
            plaus.append(row)
        if flagged:
            s0 = probs.max()
            row = {"id": pid}
            for mname, sc in methods.items():
                comp, suff = [], []
                order = np.argsort(-sc)
                for q in (0.1, 0.2, 0.5):
                    k = max(1, int(round(q * n)))
                    top = set(order[:k].tolist())
                    rest = [w for i, w in enumerate(words) if i not in top]
                    only = [w for i, w in enumerate(words) if i in top]
                    comp.append(s0 - (ex.word_probs(rest).max() if rest else 0.0))
                    suff.append(s0 - ex.word_probs(only).max())
                row[f"{mname}_comp_AOPC"] = float(np.mean(comp)); row[f"{mname}_suff_AOPC"] = float(np.mean(suff))
            faith.append(row)
        per_post.append({"id": pid, "words": words, "gold": r["tags"][:n], "pred": preds[pid]["pred"][:n],
                         "p_toxic": [round(float(v), 3) for v in probs], "ig": [round(float(v), 4) for v in ig]})
        if n_done % 100 == 0:
            print(f"  explained {n_done}/{len(ids)}", flush=True)

    res = os.path.join(args.out, "results"); os.makedirs(res, exist_ok=True)
    import pandas as pd
    P, Fa = pd.DataFrame(plaus), pd.DataFrame(faith)
    P.to_csv(os.path.join(res, "xai_plausibility_per_post.csv"), index=False)
    Fa.to_csv(os.path.join(res, "xai_faithfulness_per_post.csv"), index=False)
    summ = []
    for mname in ["integrated_gradients", "attention", "random"]:
        cols = [c for c in P.columns if c.startswith(mname)] if mname != "random" else None

        def col(df, metric):
            try:
                if mname == "random":
                    return df[[f"random_{k}_{metric}" for k in range(3)]].mean(1)
                return df[f"{mname}_{metric}"]
            except KeyError:
                return pd.Series(dtype=float)
        summ.append({"method": mname, "n_posts_plausibility": len(P), "AUPRC_mean": col(P, "AP").mean(),
                     "AUPRC_sd": col(P, "AP").std(), "topk_token_F1_mean": col(P, "topkF1").mean(),
                     "n_posts_faithfulness": len(Fa), "comprehensiveness_AOPC_mean": col(Fa, "comp_AOPC").mean(),
                     "sufficiency_AOPC_mean": col(Fa, "suff_AOPC").mean()})
    S = pd.DataFrame(summ); S.to_csv(os.path.join(res, "xai_summary.csv"), index=False)
    # paired Wilcoxon IG vs attention / random on per-post values
    from scipy.stats import wilcoxon
    tests = []
    for metric, df in [("AP", P), ("topkF1", P), ("comp_AOPC", Fa), ("suff_AOPC", Fa)]:
        for other in ["attention", "random"]:
            if f"integrated_gradients_{metric}" not in df:
                continue
            a = df[f"integrated_gradients_{metric}"]
            b = df[[f"random_{k}_{metric}" for k in range(3)]].mean(1) if other == "random" else df[f"{other}_{metric}"]
            try:
                st, p = wilcoxon(a, b)
            except ValueError:
                st, p = float("nan"), float("nan")
            tests.append({"metric": metric, "IG_vs": other, "n": len(a), "IG_mean": a.mean(), "other_mean": b.mean(), "wilcoxon_p": p})
    pd.DataFrame(tests).to_csv(os.path.join(res, "xai_tests.csv"), index=False)
    json.dump({"ig_completeness_gap_median": float(np.median(gaps)) if gaps else None, "ig_steps": args.steps,
               "posts_explained": len(per_post)}, open(os.path.join(res, "xai_completeness.json"), "w"), indent=1)
    with open(os.path.join(res, "xai_examples.jsonl"), "w", encoding="utf-8") as f:
        for x in per_post:
            g, p = spans_(x["gold"]), spans_(x["pred"])
            if g and set(g) == set(p):
                x["case"] = "true_positive_exact"
            elif not g and p:
                x["case"] = "false_positive"
            elif g and not p:
                x["case"] = "false_negative"
            else:
                x["case"] = "boundary_or_partial"
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(S.to_string())


if __name__ == "__main__":
    main()
