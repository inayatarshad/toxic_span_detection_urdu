---
title: URTOX API
sdk: docker
app_port: 7860
pinned: false
---

# URTOX Hugging Face Space API

This folder is a small FastAPI backend scaffold for the open-house deployment.

## Local run

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

Then set the React app environment variable:

```bash
REACT_APP_API_URL=http://localhost:7860
```

## Hugging Face Spaces

Create a new Space with:

- SDK: Docker
- Root files from this folder: `Dockerfile`, `app.py`, `requirements.txt`
- Port: `7860`

On startup, the API downloads `urtox_deploy_artifacts.zip` from:

```text
finalyear226/urdu-toxic-span-detector
```

The Space repo should stay small. Do not commit the `artifacts/` folder to this Space repo; the app downloads those files from the model repo.

Text mode now runs the saved `Urtox_attempt1` XLM-RoBERTa token-classification model and returns BIO toxic-span predictions.

Audio mode transcribes speech with the `openai-whisper` package, runs the transcript through the text toxic-span model, and also runs `facebook/wav2vec2-base` plus the saved `audio_toxic_classifier.pt` head for an audio-level toxic/non-toxic label.

The default ASR model size is `small`, matching the Colab notebook. You can override it with:

```text
WHISPER_MODEL_SIZE=base
```

## Test endpoint

After the Space starts, call:

```bash
POST https://your-space-name.hf.space/detect
Content-Type: application/json

{
  "mode": "text",
  "text": "yeh bad aur toxic jumla hai"
}
```
