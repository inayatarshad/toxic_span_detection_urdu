# MUTEX — An Explainable Hybrid Deep Learning Framework for Fine-Grained Toxic Span Detection in Low-Resource Languages

**Manuscript:** *An explainable hybrid deep learning framework for fine-grained toxic span detection in low-resource languages* (PeerJ Computer Science, submission ID 145907)
**Authors:** Inayat Arshad, Fajar Saleem, Sarah Iqbal, Norah Dhafer Alqahtani, Ijaz Hussain
**Corresponding author:** Ijaz Hussain — ijazhussain@pieas.edu.pk

---

## 1. Description

This repository contains the complete dataset (**URTOX**), source code, trained-model configuration and evaluation outputs used to produce every result reported in the manuscript.

**MUTEX** is a hybrid sequence-labelling framework for **fine-grained (token-level) toxic span detection in Urdu**, a low-resource, morphologically rich, cursive-script language. It combines:

1. a multilingual transformer encoder (**XLM-RoBERTa**) for contextual token embeddings,
2. a **Conditional Random Field (CRF)** decoding layer that enforces valid BIO transitions at span boundaries,
3. an Urdu-specific **preprocessing pipeline** (Unicode NFC normalisation, diacritic handling, Roman-Urdu conversion, noise removal, word segmentation),
4. a **multi-domain training strategy** over social-media, news and YouTube text, and
5. an **explainability (XAI) module** based on gradient-based token attribution with Integrated Gradients.

MUTEX attains a token-level F1 of **0.600** on the held-out URTOX test set, establishing the first supervised benchmark for Urdu toxic span detection.

> **Scope note.** This repository also hosts an ongoing *multimodal* extension (audio + text late fusion, "MUTEX-M"). The files used for **this manuscript** are the **text-only** ones listed in Section 4. The audio and fusion notebooks (`train_audio_wav2vec.ipynb`, `final_fused_results.ipynb`, `urdu_toxic_audio_dataset.csv`) belong to a separate submission and are **not** used for any result reported here.

---

## 2. Repository / resource locations

| Resource | Location |
|---|---|
| Code repository | https://github.com/inayatarshad/toxic_span_detection_urdu |
| URTOX dataset (Hugging Face) | https://huggingface.co/datasets/inayatarshad/URTOX |
| Dataset documentation & statistics portal | https://toxic-span-detection-urdu.vercel.app |
| Archived release (DOI) | *(add Zenodo DOI here after archiving — see Section 9)* |

---

## 3. Dataset Information

### 3.1 URTOX at a glance

| Property | Value |
|---|---|
| Name | **URTOX** (Urdu TOXic span corpus) |
| Records | 14,342 manually annotated posts/comments |
| Language | Urdu (Nastaliq script) + Roman Urdu (~18%) + Urdu–English code-switching |
| Annotation | Token-level **BIO** tagging (`B-TOXIC`, `I-TOXIC`, `O`) |
| Class balance | 54% toxic / 46% non-toxic |
| Sources | Social media (X, Instagram, Reddit): 5,254 · Urdu newspapers (Daily Jang, UrduPoint, BOL News Urdu, Independent Urdu): 4,300 · YouTube (comments, captions, descriptions): 4,788 |
| Toxicity categories | Offensive 36.9% · Hate 8.8% · Insult 8.2% · Neutral 46.1% |
| Inter-annotator agreement | Cohen's κ = 0.82; Krippendorff's α = 0.81 |
| Splits | 80% train (11,474) / 10% validation (1,434) / 10% test (1,434), stratified by toxic ratio and source domain |
| Format | CSV (UTF-8), auto-converted to Parquet on Hugging Face |
| License | MIT |

### 3.2 File: `URTOX_v2.csv`

| Column | Type | Description |
|---|---|---|
| `id` | int64 | Unique record identifier |
| `text` | string | Raw Urdu / Roman-Urdu post text (2–1,870 characters) |
| `label` | string | Post-level label: `toxic` / `non_toxic` |
| `sub_label` | string | Fine-grained category (hate speech, personal insult, offensive language, profanity, neutral, …) |
| `toxic_spans` | list[int] / string | Character offsets of annotated toxic characters |
| `tokens` | list[string] | Whitespace/SpaCy word-level tokenisation aligned to `BIO_tags` |
| `toxic_list` | list[string] | Surface forms of the annotated toxic tokens |
| `BIO_tags` | list[string] | Per-token gold labels in `{B-TOXIC, I-TOXIC, O}` |

`tokens[i]` corresponds one-to-one with `BIO_tags[i]`. Character offsets in `toxic_spans` index into the **raw** `text` field (before preprocessing).

### 3.3 Annotation protocol (summary)

Three trained native Urdu speakers annotated each post at word level using the BIO scheme. Agreement was measured with Cohen's κ and Krippendorff's α; an adjudication round resolved the 15% of samples with disagreement (predominantly sarcasm and culturally specific insults). Full protocol and guidelines are in the manuscript (Methodology → *Annotation Protocol*) and on the documentation portal.

### 3.4 Ethical statement

All records were collected from **publicly accessible** posts. Usernames, handles, URLs, e-mail addresses and other direct identifiers were removed during preprocessing. The corpus contains offensive language by construction and is released **solely for research on content moderation and online safety**.

---

## 4. Code Information

Files used to produce the results in this manuscript:

| File | Purpose |
|---|---|
| `urdu_toxic_span_detection.ipynb` | End-to-end pipeline: loading URTOX, preprocessing, BIO alignment, dataset splits, baseline training (BiLSTM-CRF, mBERT), evaluation |
| `URTOX_XLM+CRF_with_improv(2).ipynb` | **Proposed MUTEX model**: XLM-RoBERTa + CRF, multi-domain training, class weighting / focal loss, ablations, cross-domain transfer, Integrated-Gradients explainability |
| `roberta_model.ipynb` | RoBERTa-Large baseline for comparison |
| `requirements-training.txt` | Pinned Python dependencies for training/evaluation |
| `URTOX_v2.csv` | The URTOX dataset (see Section 3) |
| `hf-space-api/` | Inference API serving the trained MUTEX model (demo deployment) |
| `urtox-ui/` | Web interface for the dataset/demo portal |
| `OPEN_HOUSE_DEPLOYMENT.md` | Deployment notes for the demo |
| `raw_data/` | Raw evaluation outputs underlying every table in the manuscript (see Section 7) |

*Not used in this manuscript (multimodal extension):* `train_audio_wav2vec.ipynb`, `final_fused_results.ipynb`, `urdu_toxic_audio_dataset.csv`.

---

## 5. Requirements

### 5.1 Computing infrastructure used by the authors

| Component | Specification |
|---|---|
| Operating system | Ubuntu 22.04.4 LTS (64-bit) |
| CPU | Intel Xeon (16 physical cores) |
| RAM | 128 GB |
| GPU | 1 × NVIDIA RTX A6000, 48 GB GDDR6 VRAM |
| CUDA / cuDNN | CUDA 12.1 / cuDNN 8.9 |
| Python | 3.10 |
| Deep-learning stack | PyTorch 2.1.0, HuggingFace Transformers 4.38, `pytorch-crf` 0.7.2 |
| Peak VRAM (training) | ~48 GB |
| Inference throughput | ~3,200 tokens/second |
| Wall-clock training time | ~4.5 h per run (XLM-R + CRF, 5 seeds) |

A GPU with ≥16 GB VRAM is sufficient to reproduce results with `batch_size=16` and gradient accumulation; the full configuration above reproduces the reported numbers exactly.

### 5.2 Software dependencies

```
python>=3.10
torch>=2.1.0
transformers>=4.38.0
pytorch-crf>=0.7.2
datasets>=2.16.0
tokenizers>=0.15.0
scikit-learn>=1.3.0
seqeval>=1.2.2
captum>=0.7.0          # Integrated Gradients (XAI module)
spacy>=3.7.0           # word-level tokenisation
numpy>=1.24
pandas>=2.0
scipy>=1.11            # paired t-tests
matplotlib>=3.7
seaborn>=0.13
jupyter
```

Install with:

```bash
pip install -r requirements-training.txt
```

---

## 6. Usage Instructions

### 6.1 Clone and set up

```bash
git clone https://github.com/inayatarshad/toxic_span_detection_urdu.git
cd toxic_span_detection_urdu

python -m venv .venv && source .venv/bin/activate
pip install -r requirements-training.txt
python -m spacy download xx_ent_wiki_sm
```

### 6.2 Load the dataset

From the repository:

```python
import pandas as pd, ast
df = pd.read_csv("URTOX_v2.csv")
df["tokens"]    = df["tokens"].apply(ast.literal_eval)
df["BIO_tags"]  = df["BIO_tags"].apply(ast.literal_eval)
assert all(len(t) == len(b) for t, b in zip(df["tokens"], df["BIO_tags"]))
```

From Hugging Face:

```python
from datasets import load_dataset
urtox = load_dataset("inayatarshad/URTOX")
print(urtox["train"][0])
```

### 6.3 Reproduce the reported results

Run the notebooks in this order (Jupyter or `jupyter nbconvert --to notebook --execute`):

1. `urdu_toxic_span_detection.ipynb` — preprocessing, splits, baselines (BiLSTM-CRF, mBERT, XLM-R without CRF) → Table 4 baseline rows.
2. `URTOX_XLM+CRF_with_improv(2).ipynb` — proposed MUTEX model, ablations, cross-domain transfer, learning curve, XAI → Tables 4–12, Figures 4–5.

Random seeds **42, 123, 456, 789, 1011** are set at the top of each notebook; all reported numbers are means ± SD over these five runs. Every table is written to `raw_data/` as CSV when the notebooks are executed.

### 6.4 Key hyperparameters

| Hyperparameter | Search space | Selected |
|---|---|---|
| Encoder | — | `xlm-roberta-base` |
| Learning rate | {1e-5, 3e-5, 5e-5} | 3e-5 |
| Batch size | {16, 32} | 16 |
| Dropout | {0.1, 0.3} | 0.1 |
| Max sequence length | — | 256 word-pieces |
| Optimiser | — | AdamW (linear warmup 10%) |
| Early stopping | — | validation token-level F1, patience 5 epochs |
| Class weights | inverse frequency | w_O = 0.46, w_B = 2.21, w_I = 2.56 |

---

## 7. Methodology (processing / modelling steps)

1. **Collection** — public posts scraped from X, Instagram, Reddit, four Urdu news outlets and YouTube; deduplication via fuzzy matching (Levenshtein distance < 0.8); stratified 20% sampling per domain.
2. **Annotation** — word-level BIO tagging by trained native speakers; κ = 0.82, α = 0.81; adjudication of disagreements.
3. **Preprocessing** — Unicode NFC normalisation → diacritic handling → Roman-Urdu → Urdu transliteration → URL/e-mail/emoji/punctuation removal → whitespace normalisation → word segmentation → SpaCy tokenisation aligned to gold BIO tags.
4. **Modelling** — XLM-RoBERTa encoder produces contextual embeddings; a linear emission layer feeds a CRF that scores label transitions and decodes the globally optimal BIO sequence (Viterbi).
5. **Training** — multi-domain, domain-balanced mini-batch sampling with domain-weighted loss (Algorithm 1); class-weighted cross-entropy (focal loss compared as an alternative); early stopping on validation F1.
6. **Evaluation** — token-level precision/recall/F1 on the held-out test set (n = 1,434), five seeds; paired *t*-tests against baselines; ablations by 5-fold cross-validation; cross-domain transfer matrix; learning-curve analysis.
7. **Explainability** — Integrated Gradients token attribution over the predicted toxicity score; attributions rendered as span highlights in the demo interface.

### 7.1 Raw evaluation outputs (`raw_data/`)

| File | Contents |
|---|---|
| `raw_data_table4_model_comparison.csv` | Per-seed token-level P/R/F1 for BiLSTM-CRF, mBERT, XLM-R, XLM-R+CRF |
| `raw_data_table5_category_performance.csv` | Per-toxicity-category F1 and mean span length |
| `raw_data_table6_7_8_domain_results.csv` | Domain-specific, multi- vs single-domain, and cross-domain transfer F1 |
| `raw_data_table9_domain_bias.csv` | Script / code-switching / formality bias analysis |
| `raw_data_table10_learning_curve.csv` | F1 vs training-set size with *p*-values |
| `raw_data_table11_preprocessing_ablation.csv` | 5-fold ablation of preprocessing components with 95% CIs |
| `raw_data_table12_supervision_comparison.csv` | Supervised vs weakly-supervised (ARE, attention) comparison |
| `raw_data_test_predictions.csv` | Token-level gold vs predicted BIO labels for all 1,434 test posts |
| `raw_data_statistical_tests.csv` | Paired *t*-test statistics (*t*, df, *p*, Δ) for every comparison reported |

These are the files referenced as **raw data** in the manuscript's *Data Availability* statement.

---

## 8. Citation

If you use URTOX or MUTEX, please cite:

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

A preprint of this work is available at https://arxiv.org/abs/2603.05057.

### Third-party resources used

- XLM-RoBERTa — Conneau et al. (2020), ACL.
- mBERT / BERT — Devlin et al. (2019), NAACL.
- Integrated Gradients — Sundararajan, Taly & Yan (2017), ICML.
- CRF — Lafferty, McCallum & Pereira (2001), ICML.
- SpaCy, HuggingFace Transformers, `pytorch-crf`, Captum, seqeval.

---

## 9. License & Contribution Guidelines

**License:** MIT (code and dataset). See `LICENSE`.

**Archiving:** a versioned release of this repository is archived on Zenodo; cite the DOI listed in Section 2 for the exact version used in the manuscript.

**Contributions:** issues and pull requests are welcome. Please (a) open an issue describing the change, (b) keep notebooks cleared of outputs before committing, and (c) do not add personally identifying information to the dataset.

**Contact:** Ijaz Hussain — ijazhussain@pieas.edu.pk · Inayat Arshad — https://github.com/inayatarshad

---

## 10. Disclaimer

This dataset contains language that many readers will find offensive, including hate speech, slurs and profanity in Urdu and Roman Urdu. It is distributed exclusively to support research on automatic detection and moderation of harmful content. The views expressed in the collected posts are those of their original authors and are not endorsed by the dataset authors or their institutions.
