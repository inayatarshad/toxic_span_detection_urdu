# Revision pipeline (PeerJ CS #145907, second round)

Everything reported in the revised manuscript is produced by the four scripts in
this folder, from `URTOX_v2.csv` in the repository root.

```bash
bash revision/run_all.sh outputs        # GPU recommended; ~6–9 h on one Kaggle T4/P100
```

| Step | Script | Output |
|---|---|---|
| Leakage-safe preparation (dedup → near-dup grouping → group-stratified 80/10/10 split → alignment-preserving normalisation) | `prepare_data.py` | `outputs/data/urtox_clean.jsonl`, `split_assignment.csv`, `dataset_statistics.json` |
| Training/evaluation of XLM-R+CRF (MUTEX), XLM-R, mBERT, BiLSTM-CRF; 5 seeds; ablations | `train.py` | `outputs/runs/<model>__<variant>/seed<k>/` (per-post test predictions, metrics, training history) |
| Integrated-Gradients explanations and their quantitative evaluation (plausibility vs. gold spans, ERASER faithfulness, attention and random baselines) | `explain.py` | `outputs/results/xai_*.csv` |
| All tables, bootstrap CIs, permutation tests with Holm correction, error analysis | `analyze.py` | `outputs/results/*.csv`, `outputs/results_summary.md` |

Design decisions (fixed a priori, not tuned on the test split):

* Labels are predicted for the original URTOX word tokens; a word's label is read
  from its first sub-word, so gold BIO tags are never re-aligned.
* Text stays in its original script — no transliteration.
* Hyper-parameters: AdamW, lr 3e-5 (encoder) / 1e-3 (head, CRF), batch 16,
  ≤10 epochs, linear warm-up 10 %, early stopping on validation token-F1
  (patience 3), max 256 sub-words; BiLSTM-CRF: Adam 1e-3, ≤30 epochs, patience 5.
* Seeds 42, 123, 456, 789, 1011. The test split is used once per run, after
  model selection on the validation split.
* No data augmentation is used.

`run_all.sh` is resumable: rerunning skips finished runs.
