# MUTEX — An Explainable Hybrid Deep Learning Framework for Fine-Grained Toxic Span Detection in Low-Resource Languages

**Manuscript:** *An explainable hybrid deep learning framework for fine-grained toxic span detection in low-resource languages* (PeerJ Computer Science, submission ID 145907, revised version)
**Authors:** Inayat Arshad, Fajar Saleem, Sarah Iqbal, Norah Dhafer Alqahtani, Ijaz Hussain
**Corresponding author:** Ijaz Hussain — ijazhussain@pieas.edu.pk

---

## 1. Description

This repository contains the dataset (**URTOX**), the code and the evaluation outputs behind every result in the revised manuscript.

**MUTEX** is a sequence-labelling pipeline for **word-level toxic span detection in Urdu**. It applies established components to a new task and resource:

1. a multilingual transformer encoder (**XLM-RoBERTa**),
2. a **conditional random field (CRF)** decoder with BIO constraints, so that predicted spans are always well formed,
3. **Integrated Gradients** token attributions, evaluated quantitatively for plausibility and faithfulness.

**Main results** (held-out test split, mean ± SD over five seeds):

| Model | Token-level F1 | Exact-span F1 | Ill-formed spans |
|---|---|---|---|
| BiLSTM-CRF | 62.8 ± 0.7 | 51.2 ± 0.5 | 0% |
| Multilingual BERT | 66.0 ± 0.3 | 52.0 ± 0.3 | 4.6% |
| XLM-RoBERTa (softmax) | 70.4 ± 0.5 | 57.4 ± 1.3 | 4.6% |
| **MUTEX (XLM-RoBERTa + CRF)** | **70.2 ± 0.6** | **59.9 ± 1.3** | **0%** |

The CRF does not change token-level F1; it improves exact-span F1 by 2.5 points and removes ill-formed BIO sequences. Text normalisation and a larger encoder give no measurable benefit. Full tables are in `revision/outputs/results_summary.md`.

> **Scope.** All experiments are on Urdu in the Perso-Arabic script. Roman Urdu, cross-domain behaviour and other languages were not evaluated.
>
> This repository also hosts a separate *multimodal* extension (audio + text). The files `train_audio_wav2vec.ipynb`, `final_fused_results.ipynb` and `urdu_toxic_audio_dataset.csv` belong to that work and are **not** used for any result in this manuscript.

---

## 2. Resource locations

| Resource | Location |
|---|---|
| Code repository | https://github.com/inayatarshad/toxic_span_detection_urdu |
| URTOX dataset (Hugging Face) | https://huggingface.co/datasets/inayatarshad/URTOX |
| Dataset documentation portal | https://toxic-span-detection-urdu.vercel.app |

---

## 3. Dataset

### 3.1 URTOX at a glance

Statistics are computed by `revision/prepare_data.py` (see `revision/outputs/data/dataset_statistics.json`).

| Property | Value |
|---|---|
| Released file | `URTOX_v2.csv`, 14,337 posts |
| After validation and exact de-duplication | 14,160 posts, 412,203 word tokens |
| Annotation | Word-level **BIO** tags (`B-TOXIC`, `I-TOXIC`, `O`) plus one post-level label and category |
| Post labels | 7,641 toxic (54.0%) / 6,519 non-toxic (46.0%) |
| Posts with at least one toxic span | 6,221 (43.9%) |
| Tag distribution | O 95.82% · B-TOXIC 2.37% · I-TOXIC 1.82% |
| Words per post | mean 29.1, median 20, max 408 |
| Toxic spans | 9,751; mean 1.77 words, median 1 |
| Categories | offensive 5,271 · hate 1,232 · insult 1,041 · slur 83 · threat 14 · non-toxic 6,519 |
| Script | 13,774 Perso-Arabic · 380 mixed · 6 predominantly Latin |
| Inter-annotator agreement | Cohen's κ = 0.82; Krippendorff's α = 0.81 (independent labels, all word tokens) |
| Split | 11,326 train / 1,417 validation / 1,417 test (`revision/outputs/data/split_assignment.csv`) |
| Sources | X, Instagram, Reddit; Daily Jang, UrduPoint, BOL News Urdu, Independent Urdu; YouTube. Collected October 2025 – January 2026. The source of each individual post is **not** recorded in the released file. |

### 3.2 File: `URTOX_v2.csv`

| Column | Description |
|---|---|
| `id` | Record identifier |
| `text` | Raw post text |
| `label` | Post-level label: `toxic` / `non_toxic` |
| `sub_label` | Post-level category: `normal`, `offensive`, `hate`, `insult`, `slur`, `threat` |
| `toxic_spans` | Character offsets of toxic characters |
| `tokens` | Word-level tokenisation, aligned one-to-one with `BIO_tags` |
| `toxic_list` | Surface forms of the annotated toxic words |
| `BIO_tags` | Gold label for each word |

Known issues, handled by `revision/prepare_data.py`: two rows with mismatched token/tag lengths (dropped), one malformed tag list (repaired), and 175 exact duplicates (removed before splitting).

### 3.3 Annotation protocol (summary)

Two of the authors annotated every post independently at word level; a third, senior author adjudicated the cases that remained unresolved after discussion (about 15% of posts). All three are native speakers of Urdu. Guidelines and limitations are described in the manuscript (*Annotation Protocol*, *Limitations*).

### 3.4 Ethical statement

All records were collected from publicly accessible sources. The corpus contains offensive language by construction and is released solely for research on content moderation and online safety.

---

## 4. Code

**Everything reported in the revised manuscript is produced by the scripts in `revision/`** (see `revision/README.md`):

| File | Purpose |
|---|---|
| `revision/prepare_data.py` | Validation, de-duplication, near-duplicate grouping, group-level stratified 80/10/10 split, alignment-preserving normalisation, dataset statistics |
| `revision/train.py` | Training and evaluation of XLM-R+CRF (MUTEX), XLM-R, mBERT and BiLSTM-CRF; resumable, one output folder per run |
| `revision/explain.py` | Integrated Gradients and its quantitative evaluation (plausibility, comprehensiveness, sufficiency; attention and random baselines) |
| `revision/analyze.py` | All metrics, bootstrap confidence intervals, randomisation tests with Holm correction, breakdowns, error analysis |
| `revision/run_all.sh` | Runs the whole study with one command |
| `revision/outputs/` | Outputs of the run reported in the manuscript (Section 7) |

Other content:

| File | Note |
|---|---|
| `URTOX_v2.csv` | The dataset |
| `urdu_toxic_span_detection.ipynb`, `roberta_model.ipynb`, `URTOX_XLM+CRF_with_improv(2).ipynb` | Earlier exploratory notebooks, kept for history. **They are not used for any reported result.** |
| `hf-space-api/`, `urtox-ui/`, `OPEN_HOUSE_DEPLOYMENT.md` | Demo API and web interface. The demo serves an earlier XLM-RoBERTa checkpoint without the CRF and is not the evaluated system. |

---

## 5. Requirements

### 5.1 Computing infrastructure used for the reported results

| Component | Specification |
|---|---|
| Platform | Kaggle Notebooks |
| GPU | 2 × NVIDIA Tesla T4 (16 GB each), one process per GPU |
| Python / PyTorch / Transformers | 3.13 / 2.11 / 5.20 |
| Time per MUTEX run | about 21 minutes (peak GPU memory 6.4 GB) |
| Whole study | 37 training runs, 9.8 GPU-hours, 5.9 hours wall-clock |

Any single GPU with at least 8 GB of memory is sufficient for the base models (16 GB for `xlm-roberta-large`).

### 5.2 Software dependencies

```
python>=3.10
torch>=2.1
transformers>=4.38
numpy
pandas
scipy
```

The CRF, the metrics, the statistical tests and Integrated Gradients are implemented in the released code; no further packages are needed.

---

## 6. Usage

```bash
git clone https://github.com/inayatarshad/toxic_span_detection_urdu.git
cd toxic_span_detection_urdu
pip install torch transformers numpy pandas scipy
bash revision/run_all.sh outputs
```

`run_all.sh` prepares the data, trains every model with seeds 42, 123, 456, 789 and 1011, evaluates the explanations and writes all tables to `outputs/results/`. Finished runs are skipped if the script is started again. To recompute the tables from the released predictions without a GPU:

```bash
mkdir out && tar -xzf revision/outputs/urtox_revision_outputs_full.tar.gz -C out
python revision/analyze.py --data out/data/urtox_clean.jsonl --out out
```

Hyperparameters (fixed in advance, no search): AdamW, learning rate 3e-5 (encoder) and 1e-3 (emission layer and CRF), batch size 16, at most 10 epochs, 10% linear warm-up, early stopping on validation token-level F1 (patience 3), at most 256 sub-words. No class weighting, focal loss or data augmentation.

---

## 7. Evaluation outputs (`revision/outputs/`)

These are the raw data referenced in the manuscript.

| File | Contents |
|---|---|
| `results_summary.md` | All main results and paired comparisons |
| `results/per_seed_metrics.csv` | Metrics of every individual run |
| `results/main_results.csv` | Mean, SD and 95% CI for each configuration |
| `results/significance_tests.csv` | Paired bootstrap CIs and randomisation tests (token-level and exact-span) |
| `results/mutex_by_category.csv`, `mutex_by_length.csv`, `mutex_by_script.csv` | Breakdowns of the MUTEX results |
| `results/mutex_error_analysis.csv` | Outcome of every gold span |
| `results/learning_curve.csv` | Performance against training-set size |
| `results/xai_*.csv`, `xai_examples.jsonl` | Explainability evaluation and per-post attributions |
| `data/split_assignment.csv`, `data/dataset_statistics.json` | The fixed split and the dataset statistics |
| `urtox_revision_outputs_full.tar.gz` | Everything above plus per-post gold and predicted labels for all 37 runs |
| `pipeline_log.txt` | Complete training log |

---

## 8. Citation

```bibtex
@article{arshad2026mutex,
  title   = {An Explainable Hybrid Deep Learning Framework for Fine-Grained
             Toxic Span Detection in Low-Resource Languages},
  author  = {Arshad, Inayat and Saleem, Fajar and Iqbal, Sarah and
             Alqahtani, Norah Dhafer and Hussain, Ijaz},
  journal = {PeerJ Computer Science},
  year    = {2026},
  note    = {Under review}
}

@misc{arshad2026urtox,
  title        = {URTOX: A Token-Level Annotated Urdu Toxic Span Dataset},
  author       = {Arshad, Inayat and Saleem, Fajar and Iqbal, Sarah and
                  Alqahtani, Norah Dhafer and Hussain, Ijaz},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/datasets/inayatarshad/URTOX}}
}
```

An earlier version is available as a preprint at https://arxiv.org/abs/2603.05057; the results in this repository supersede it.

Third-party resources: XLM-RoBERTa (Conneau et al., 2020), BERT (Devlin et al., 2019), Integrated Gradients (Sundararajan et al., 2017), CRF (Lafferty et al., 2001), HuggingFace Transformers, PyTorch.

---

## 9. License and contact

**License:** MIT (code and dataset).

**Contact:** Ijaz Hussain — ijazhussain@pieas.edu.pk · Inayat Arshad — https://github.com/inayatarshad

---

## 10. Disclaimer

This dataset contains language that many readers will find offensive, including hate speech and slurs in Urdu. It is distributed exclusively to support research on the detection and moderation of harmful content. The views expressed in the collected posts are those of their original authors and are not endorsed by the dataset authors or their institutions.
