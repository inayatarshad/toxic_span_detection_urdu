🔍 Multimodal Urdu Toxic Span Detection

A multimodal framework for detecting toxic content in Urdu text and audio, combining XLM-RoBERTa for text-based toxic span detection with Wav2Vec2 for audio-based toxicity classification via late fusion.

📌 Overview
This project extends MUTEX (Multilingual Transformer + CRF for Urdu Toxic Span Detection) by adding an audio modality, creating the first multimodal pipeline for Urdu toxic content detection.
| Modality | Model | F1 Score |
|---|---|---|
| Text only | XLM-RoBERTa | 67% |
| Audio only | Wav2Vec2 | 70% |
| **Multimodal Fusion** | **XLM-RoBERTa + Wav2Vec2** | **79.34%** |

## 🗂️ Project Structure
```
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
```

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

**Video Demo
https://www.loom.com/share/98725ce0fb2a498bbf37f21a81cd5e59

<img width="1331" height="630" alt="image" src="https://github.com/user-attachments/assets/84e8a843-be68-4a55-b072-1b94fb7abd8d" />
<img width="1330" height="633" alt="image" src="https://github.com/user-attachments/assets/ab43a77a-c804-49d6-8b47-46cdf474351c" />
<img width="1330" height="634" alt="image" src="https://github.com/user-attachments/assets/741f4dee-817e-4340-98f8-804e73192d1a" />
<img width="1330" height="635" alt="image" src="https://github.com/user-attachments/assets/b88cac6d-b2db-4d2c-86d0-1cb56133b1c0" />
<img width="1333" height="634" alt="image" src="https://github.com/user-attachments/assets/3cae1b13-cbee-4232-9575-778c81f8f8f0" />
<img width="1318" height="627" alt="image" src="https://github.com/user-attachments/assets/41fc4d96-9a77-44bf-969a-0fd2967362c6" />
<img width="1329" height="634" alt="image" src="https://github.com/user-attachments/assets/8eb199e5-d77d-439f-b6c9-2216737387ae" />



          
## 📝 About

This project focuses on toxic span detection in Urdu text using a 
transformer-based token classification approach. The goal is not only 
to identify whether a sentence contains toxic content, but also to 
**locate and highlight the specific toxic words or spans** within the text.

**No prior work exists on Urdu toxic span detection** — this is the 
first proposed framework for fine-grained toxicity localization in Urdu.

> 📄 A journal paper has been submitted to **Elsevier Applied Soft Computing** 
> and will be published soon.
> linK: https://arxiv.org/abs/2603.05057


So, updates that i made towards the end of the project , was exploring agentic AI part that can be integrated in my NLP project, so I used Lyzr.AI for a real time agent(just for fun), that can call you and basically will take out the toxic spans in the sentence and will be able to give better recommendations on what can be replaced for the toxic words by keeping the context same ,but using politer and user friendly terms, It will also be able to provide you detailed information about NLP, and  its tasks. Loom link:👇
https://www.loom.com/share/ead63d8b123545a890d10bab056705e3
### 🎯 Task Summary

| Component | Details |
|---|---|
| Task | Toxic span detection at word/token level in Urdu |
| Model | XLM-RoBERTa fine-tuned for token classification |
| Dataset | Newly created and manually annotated Urdu toxic span dataset |
| Output | Highlighted toxic words/spans within each input sentence |
| Use Case | Moderation, abusive language detection, explainable toxicity analysis |
| Framework | HuggingFace Transformers + PyTorch |
