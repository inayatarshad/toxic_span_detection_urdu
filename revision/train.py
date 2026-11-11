"""
train.py — training/evaluation for the PeerJ revision of MUTEX.

Models (all predict ONE label per original URTOX word token, so gold tags
never have to be re-aligned):
  xlmr_crf        XLM-RoBERTa-base encoder + linear emissions + CRF  (MUTEX)
  xlmr            XLM-RoBERTa-base encoder + softmax (no CRF)
  mbert           bert-base-multilingual-cased + softmax
  bilstm_crf      word embeddings + 2-layer BiLSTM + CRF (trained from scratch)
  xlmr_large_crf  XLM-RoBERTa-large + CRF (optional)

Sub-word handling: each word is split by the model tokenizer; the emission for a
word is read from its FIRST sub-word. Words removed by truncation are predicted
"O" and are still counted at evaluation time.

Every (model, variant, seed) run writes
  <out>/runs/<run_name>/seed<k>/{metrics.json, test_predictions.jsonl,
                                 validation_predictions.jsonl, history.json}
and is skipped if metrics.json already exists (safe to resume).

Usage:
  python revision/train.py --data outputs/data/urtox_clean.jsonl --out outputs \
         --models xlmr_crf xlmr mbert bilstm_crf --seeds 42 123 456 789 1011
"""
import argparse, json, math, os, random, time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

LABELS = ["O", "B-TOXIC", "I-TOXIC"]
L2I = {l: i for i, l in enumerate(LABELS)}
ENCODERS = {"xlmr_crf": ("xlm-roberta-base", True), "xlmr": ("xlm-roberta-base", False),
            "mbert": ("bert-base-multilingual-cased", False), "xlmr_large_crf": ("xlm-roberta-large", True)}


# ====================================================================== CRF
class CRF(nn.Module):
    """Linear-chain CRF with hard BIO constraints (O->I and START->I forbidden)."""

    def __init__(self, n=3, constrain=True):
        super().__init__()
        self.n = n
        self.trans = nn.Parameter(torch.zeros(n, n))
        self.start = nn.Parameter(torch.zeros(n))
        self.end = nn.Parameter(torch.zeros(n))
        mask = torch.zeros(n, n)
        smask = torch.zeros(n)
        if constrain:
            mask[L2I["O"], L2I["I-TOXIC"]] = -1e4
            smask[L2I["I-TOXIC"]] = -1e4
        self.register_buffer("tmask", mask)
        self.register_buffer("smask", smask)

    def _T(self):
        return self.trans + self.tmask, self.start + self.smask

    def nll(self, em, tags, mask):  # em (B,T,n) float32, tags (B,T), mask (B,T) bool, mask[:,0] True
        T_, S_ = self._T()
        B, T, _ = em.shape
        tags = tags.clamp(min=0)
        score = S_[tags[:, 0]] + em[:, 0].gather(1, tags[:, :1]).squeeze(1)
        for t in range(1, T):
            s = T_[tags[:, t - 1], tags[:, t]] + em[:, t].gather(1, tags[:, t:t + 1]).squeeze(1)
            score = score + s * mask[:, t]
        last = tags.gather(1, (mask.sum(1) - 1).unsqueeze(1)).squeeze(1)
        score = score + self.end[last]
        alpha = S_.unsqueeze(0) + em[:, 0]
        for t in range(1, T):
            nxt = torch.logsumexp(alpha.unsqueeze(2) + T_.unsqueeze(0) + em[:, t].unsqueeze(1), dim=1)
            alpha = torch.where(mask[:, t].unsqueeze(1), nxt, alpha)
        logZ = torch.logsumexp(alpha + self.end.unsqueeze(0), dim=1)
        return (logZ - score).mean()

    @torch.no_grad()
    def decode(self, em, mask):
        T_, S_ = self._T()
        B, T, n = em.shape
        score = S_.unsqueeze(0) + em[:, 0]
        back = []
        for t in range(1, T):
            s = score.unsqueeze(2) + T_.unsqueeze(0)
            best, idx = s.max(dim=1)
            nxt = best + em[:, t]
            score = torch.where(mask[:, t].unsqueeze(1), nxt, score)
            back.append(torch.where(mask[:, t].unsqueeze(1), idx, torch.arange(n, device=em.device).expand(B, n)))
        score = score + self.end.unsqueeze(0)
        best_last = score.argmax(1)
        out = [best_last]
        for idx in reversed(back):
            best_last = idx.gather(1, best_last.unsqueeze(1)).squeeze(1)
            out.append(best_last)
        out.reverse()
        return torch.stack(out, 1)


# =================================================================== models
class TransformerTagger(nn.Module):
    def __init__(self, name, use_crf, dropout=0.1):
        super().__init__()
        from transformers import AutoModel
        self.enc = AutoModel.from_pretrained(name)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(self.enc.config.hidden_size, len(LABELS))
        self.crf = CRF() if use_crf else None

    def emissions(self, input_ids, attention_mask, first_idx, word_mask, inputs_embeds=None):
        if inputs_embeds is not None:
            h = self.enc(inputs_embeds=inputs_embeds, attention_mask=attention_mask).last_hidden_state
        else:
            h = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        em = self.head(self.drop(h)).float()
        idx = first_idx.unsqueeze(-1).expand(-1, -1, em.size(-1))
        return em.gather(1, idx)  # (B, W, 3) word-level emissions

    def loss(self, batch):
        em = self.emissions(batch["input_ids"], batch["attention_mask"], batch["first_idx"], batch["word_mask"])
        return _loss(self.crf, em, batch["labels"], batch["word_mask"])

    def predict(self, batch):
        em = self.emissions(batch["input_ids"], batch["attention_mask"], batch["first_idx"], batch["word_mask"])
        return _predict(self.crf, em, batch["word_mask"]), em.softmax(-1)


class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size, emb=300, hidden=256, dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hidden, num_layers=2, bidirectional=True, batch_first=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(2 * hidden, len(LABELS))
        self.crf = CRF()

    def emissions(self, word_ids, word_mask):
        x = self.drop(self.emb(word_ids))
        lengths = word_mask.sum(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        h, _ = self.lstm(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=word_ids.size(1))
        return self.head(self.drop(h)).float()

    def loss(self, batch):
        return _loss(self.crf, self.emissions(batch["word_ids"], batch["word_mask"]), batch["labels"], batch["word_mask"])

    def predict(self, batch):
        em = self.emissions(batch["word_ids"], batch["word_mask"])
        return _predict(self.crf, em, batch["word_mask"]), em.softmax(-1)


def _loss(crf, em, labels, mask):
    if crf is not None:
        return crf.nll(em, labels, mask)
    return nn.functional.cross_entropy(em[mask], labels[mask])


def _predict(crf, em, mask):
    if crf is not None:
        return crf.decode(em, mask)
    return em.argmax(-1)


# ===================================================================== data
def load(path, field):
    recs = [json.loads(l) for l in open(path, encoding="utf-8")]
    for r in recs:
        r["words"] = r[field]
        r["y"] = [L2I[t] for t in r["tags"]]
    return recs


class TransformerCollate:
    def __init__(self, tok, max_len):
        self.tok, self.max_len = tok, max_len
        self.cache = {}

    def safe(self, w):
        # a word that the tokenizer maps to zero sub-words would lose its label; use <unk> instead
        if w not in self.cache:
            pieces = self.tok.tokenize(w)
            self.cache[w] = w if any(p not in ("▁", "") for p in pieces) else self.tok.unk_token
        return self.cache[w]

    def __call__(self, recs):
        enc = self.tok([[self.safe(w) for w in r["words"]] for r in recs], is_split_into_words=True, truncation=True,
                       max_length=self.max_len, padding=True, return_tensors="pt")
        W = max(len(r["words"]) for r in recs)
        first = torch.zeros(len(recs), W, dtype=torch.long)
        wmask = torch.zeros(len(recs), W, dtype=torch.bool)
        labels = torch.zeros(len(recs), W, dtype=torch.long)
        kept = []
        for b, r in enumerate(recs):
            seen = set()
            for pos, wid in enumerate(enc.word_ids(b)):
                if wid is not None and wid not in seen:
                    seen.add(wid)
                    first[b, wid] = pos
            n = len(seen)  # words surviving truncation are a prefix
            wmask[b, :n] = True
            labels[b, :len(r["y"])] = torch.tensor(r["y"])
            kept.append(n)
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "first_idx": first,
                "word_mask": wmask, "labels": labels, "kept": kept, "recs": recs}


class WordCollate:
    def __init__(self, vocab, max_words=512):
        self.v, self.max_words = vocab, max_words

    def __call__(self, recs):
        W = min(self.max_words, max(len(r["words"]) for r in recs))
        ids = torch.zeros(len(recs), W, dtype=torch.long)
        wmask = torch.zeros(len(recs), W, dtype=torch.bool)
        labels = torch.zeros(len(recs), W, dtype=torch.long)
        kept = []
        for b, r in enumerate(recs):
            n = min(W, len(r["words"]))
            ids[b, :n] = torch.tensor([self.v.get(w, 1) for w in r["words"][:n]])
            wmask[b, :n] = True
            labels[b, :n] = torch.tensor(r["y"][:n])
            kept.append(n)
        return {"word_ids": ids, "word_mask": wmask, "labels": labels, "kept": kept, "recs": recs}


def batches(recs, bs, collate, shuffle, rng):
    idx = list(range(len(recs)))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        yield collate([recs[j] for j in idx[i:i + bs]])


def to_dev(batch, dev):
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}


# =================================================================== metrics
def token_prf(golds, preds):
    tp = fp = fn = 0
    for g, p in zip(golds, preds):
        for a, b in zip(g, p):
            ga, pb = a != "O", b != "O"
            tp += ga and pb
            fp += (not ga) and pb
            fn += ga and (not pb)
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    return P, R, (2 * P * R / (P + R) if P + R else 0.0)


# ================================================================ run logic
def evaluate(model, recs, collate, bs, dev, amp):
    model.eval()
    out = []
    with torch.no_grad():
        for batch in batches(recs, bs, collate, False, None):
            b = to_dev(batch, dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                pred, prob = model.predict(b)
            pred, prob = pred.cpu().numpy(), prob.float().cpu().numpy()
            for i, r in enumerate(batch["recs"]):
                n, k = len(r["words"]), batch["kept"][i]
                tags = [LABELS[t] for t in pred[i, :k]] + ["O"] * (n - k)
                ptox = [round(float(1 - prob[i, j, 0]), 4) for j in range(k)] + [0.0] * (n - k)
                out.append({"id": r["id"], "gold": r["tags"], "pred": tags, "p_toxic": ptox, "truncated_words": n - k})
    return out


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def run_one(args, model_key, seed, data, run_dir):
    set_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = dev.type == "cuda" and model_key != "bilstm_crf"
    train, val, test = data
    rng = random.Random(seed)
    if model_key == "bilstm_crf":
        cnt = Counter(w for r in train for w in r["words"])
        vocab = {"<pad>": 0, "<unk>": 1}
        for w, c in cnt.items():
            if c >= 1:
                vocab[w] = len(vocab)
        model = BiLSTMCRF(len(vocab)).to(dev)
        collate = WordCollate(vocab)
        opt = torch.optim.Adam(model.parameters(), lr=args.lstm_lr)
        epochs, patience, bs = args.lstm_epochs, args.lstm_patience, 32
        sched = None
    else:
        from transformers import AutoTokenizer, get_linear_schedule_with_warmup
        name, use_crf = ENCODERS[model_key]
        name = args.encoder_override or name
        tok = AutoTokenizer.from_pretrained(name)
        model = TransformerTagger(name, use_crf).to(dev)
        collate = TransformerCollate(tok, args.max_len)
        enc_params = list(model.enc.parameters())
        other = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
        lr = args.lr_large if "large" in model_key else args.lr
        opt = torch.optim.AdamW([{"params": enc_params, "lr": lr}, {"params": other, "lr": args.head_lr}],
                                weight_decay=0.01)
        epochs, patience, bs = args.epochs, args.patience, args.batch_size
        steps = epochs * math.ceil(len(train) / bs)
        sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    best_f1, best_state, bad, hist = -1.0, None, 0, []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for batch in batches(train, bs, collate, True, rng):
            b = to_dev(batch, dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss = model.loss(b)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            if sched:
                sched.step()
            tot += loss.item(); nb += 1
        vp = evaluate(model, val, collate, 64, dev, amp)
        P, R, F = token_prf([x["gold"] for x in vp], [x["pred"] for x in vp])
        hist.append({"epoch": ep, "train_loss": tot / max(nb, 1), "val_token_P": P, "val_token_R": R, "val_token_F1": F,
                     "elapsed_s": round(time.time() - t0, 1)})
        print(f"  [{model_key} s{seed}] ep{ep} loss={tot / max(nb, 1):.4f} valF1={F:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if F > best_f1 + 1e-4:
            best_f1, bad = F, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    vp = evaluate(model, val, collate, 64, dev, amp)
    tp = evaluate(model, test, collate, 64, dev, amp)
    os.makedirs(run_dir, exist_ok=True)
    for name, rows in [("validation", vp), ("test", tp)]:
        with open(os.path.join(run_dir, f"{name}_predictions.jsonl"), "w", encoding="utf-8") as f:
            for x in rows:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
    P, R, F = token_prf([x["gold"] for x in tp], [x["pred"] for x in tp])
    metrics = {"model": model_key, "seed": seed, "best_val_token_F1": best_f1, "test_token_P": P, "test_token_R": R,
               "test_token_F1": F, "epochs_run": len(hist), "train_time_s": round(time.time() - t0, 1),
               "n_train": len(train), "n_val": len(val), "n_test": len(test),
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
               "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0,
               "config": {k: v for k, v in vars(args).items() if k not in ("models", "seeds")}}
    json.dump(hist, open(os.path.join(run_dir, "history.json"), "w"), indent=1)
    if args.save_model and seed == args.save_model_seed and model_key == "xlmr_crf" and args.variant == "main":
        torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))
    json.dump(metrics, open(os.path.join(run_dir, "metrics.json"), "w"), indent=1)  # written last = run complete
    print(f"  DONE {model_key} s{seed}: test token F1={F:.4f}", flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/data/urtox_clean.jsonl")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--models", nargs="+", default=["xlmr_crf", "xlmr", "mbert", "bilstm_crf"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 1011])
    ap.add_argument("--variant", default="main", help="main | no_preprocessing | frac<0-1>")
    ap.add_argument("--train_fraction", type=float, default=1.0)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--lr_large", type=float, default=1.5e-5)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--lstm_lr", type=float, default=1e-3)
    ap.add_argument("--lstm_epochs", type=int, default=30)
    ap.add_argument("--lstm_patience", type=int, default=5)
    ap.add_argument("--encoder_override", default=None, help="for smoke tests only")
    ap.add_argument("--save_model", action="store_true")
    ap.add_argument("--save_model_seed", type=int, default=42)
    ap.add_argument("--deadline_hours", type=float, default=None, help="do not START new runs after this")
    ap.add_argument("--limit", type=int, default=None, help="smoke tests only")
    args = ap.parse_args()
    T0 = float(os.environ.get("PIPELINE_T0", time.time()))

    field = "tokens" if args.variant == "no_preprocessing" else "norm_tokens"
    recs = load(args.data, field)
    train = [r for r in recs if r["split"] == "train"]
    val = [r for r in recs if r["split"] == "validation"]
    test = [r for r in recs if r["split"] == "test"]
    if args.train_fraction < 1.0:
        rnd = random.Random(7)
        rnd.shuffle(train)
        train = sorted(train[:int(len(train) * args.train_fraction)], key=lambda r: r["id"])
    if args.limit:
        train, val, test = train[:args.limit], val[:args.limit // 4], test[:args.limit // 4]
    print(f"data: train={len(train)} val={len(val)} test={len(test)} variant={args.variant}", flush=True)

    for mk in args.models:
        for s in args.seeds:
            run_dir = os.path.join(args.out, "runs", f"{mk}__{args.variant}", f"seed{s}")
            if os.path.exists(os.path.join(run_dir, "metrics.json")):
                print(f"skip {run_dir} (done)")
                continue
            if args.deadline_hours and (time.time() - T0) / 3600 > args.deadline_hours:
                print(f"deadline reached, not starting {run_dir}")
                return
            print(f"=== start {mk} {args.variant} seed {s}", flush=True)
            run_one(args, mk, s, (train, val, test), run_dir)


if __name__ == "__main__":
    main()
