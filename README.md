🔍 Multimodal Urdu Toxic Span Detection
A multimodal framework for detecting toxic content in Urdu text and audio, combining XLM-RoBERTa for text-based toxic span detection with Wav2Vec2 for audio-based toxicity classification via late fusion.

📌 Overview
This project extends MUTEX (Multilingual Transformer + CRF for Urdu Toxic Span Detection) by adding an audio modality, creating the first multimodal pipeline for Urdu toxic content detection.
| Modality | Model | F1 Score |
|---|---|---|
| Text only | XLM-RoBERTa | 67% |
| Audio only | Wav2Vec2 | 70% |
| **Multimodal Fusion** | **XLM-RoBERTa + Wav2Vec2** | **79.34%** |

🗂️ Project Structure
📁 Project
│
├── 📓 Notebook 1 — Text Model (MUTEX)
│   └── XLM-RoBERTa fine-tuned on URTOX for toxic span detection
│
├── 📓 Notebook 2 — Audio Model
│   ├── TTS audio generation from URTOX using Edge TTS
│   ├── Wav2Vec2 feature extraction
│   └── Audio toxic classifier training
│
├── 📓 Notebook 3 — Fusion
│   ├── Late fusion (0.6 text + 0.4 audio)
│   ├── Final multimodal evaluation
│   └── Real-world WhatsApp audio testing pipeline
│
└── 📁 Dataset
    ├── URTOX_v2.csv                    ← Original text dataset (14,342 samples)
    ├── urdu_toxic_audio_dataset.csv    ← Dataset with audio paths
    └── urdu_toxic_audio_og/            ← MP3 audio files folder

📊 Dataset — URTOX
URTOX is a manually annotated Urdu toxic span dataset containing 14,342 samples collected from:

| Source | Samples | Toxic % | Non-Toxic % |
|---|---|---|---|
| Social Media (X, Instagram, Reddit) | 5,254 | 57% | 43% |
| Urdu Newspapers | 4,300 | 52% | 48% |
| YouTube | 4,788 | 56% | 44% |
| **Total** | **14,342** | **54%** | **46%** |

**Dataset Columns
id            → unique identifier
text          → raw Urdu text
tokens        → tokenized words (list)
BIO_tags      → token-level BIO annotation (B-Toxic, I-Toxic, O)
toxic_spans   → toxic word spans
toxic_list    → list of toxic words
label         → sentence-level label (toxic / non_toxic)
sub_label     → toxicity category (hate, insult, offensive, neutral)
audio_path    → path to corresponding MP3 file


┌─────────────────────────────────────────────────────┐
│                   INPUT: Audio File                  │
└──────────────────────┬──────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
  ┌───────────────┐        ┌─────────────────┐
  │   Wav2Vec2    │        │  XLM-RoBERTa    │
  │  (Audio       │        │  (Text Model)   │
  │   Features)   │        │                 │
  └───────┬───────┘        └────────┬────────┘
          │                         │
          ▼                         ▼
  ┌───────────────┐        ┌─────────────────┐
  │ Audio Toxic   │        │  Toxic Span     │
  │ Classifier    │        │  Detection      │
  │ P(toxic)=70% │        │  P(toxic)=67%  │
  └───────┬───────┘        └────────┬────────┘
          │                         │
          └────────────┬────────────┘
                       ▼
          ┌────────────────────────┐
          │    Late Fusion         │
          │  0.4 × audio +         │
          │  0.6 × text            │
          └────────────┬───────────┘
                       ▼
          ┌────────────────────────┐
          │     FINAL OUTPUT       │
          │  ✅ Toxic / Non-Toxic  │
          │  ⚠️  Toxic Spans       │
          │  📊 Confidence Score   │
          └────────────────────────┘
          
## 📝 About

This project focuses on toxic span detection in Urdu text using a 
transformer-based token classification approach. The goal is not only 
to identify whether a sentence contains toxic content, but also to 
**locate and highlight the specific toxic words or spans** within the text.

**No prior work exists on Urdu toxic span detection** — this is the 
first proposed framework for fine-grained toxicity localization in Urdu.

> 📄 A journal paper has been submitted to **Elsevier Applied Soft Computing** 
> and will be published soon.

### 🎯 Task Summary

| Component | Details |
|---|---|
| Task | Toxic span detection at word/token level in Urdu |
| Model | XLM-RoBERTa fine-tuned for token classification |
| Dataset | Newly created and manually annotated Urdu toxic span dataset |
| Output | Highlighted toxic words/spans within each input sentence |
| Use Case | Moderation, abusive language detection, explainable toxicity analysis |
| Framework | HuggingFace Transformers + PyTorch |
