import base64
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
from pydantic import BaseModel
import torch
import torch.nn as nn
import whisper
from transformers import AutoModelForTokenClassification, AutoTokenizer, Wav2Vec2Model, Wav2Vec2Processor


app = FastAPI(title="URTOX Toxic Span Detection API")

MODEL_REPO_ID = "finalyear226/urdu-toxic-span-detector"
MODEL_ZIP_NAME = "urtox_deploy_artifacts.zip"
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")
ARTIFACTS_DIR = Path("artifacts")
TEXT_MODEL_DIR = ARTIFACTS_DIR / "Urtox_attempt1"
AUDIO_MODEL_PATH = ARTIFACTS_DIR / "audio_toxic_classifier.pt"
LABELS_PATH = ARTIFACTS_DIR / "label_classes.npy"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEXT_TOKENIZER = None
TEXT_MODEL = None
AUDIO_PROCESSOR = None
AUDIO_WAV2VEC_MODEL = None
AUDIO_CLASSIFIER = None
AUDIO_LABELS = None
WHISPER_MODEL = None
MAX_AUDIO_LENGTH = 16000 * 10
URDU_PUNCTUATION = "،۔؟!؛:,.!?\"'()[]{}<>«»“”‘’"
TOXIC_LEXICON = {
    "بہنچود",
    "بhenchod",
    "bhenchod",
    "بنچود",
    "مادرچود",
    "ماںچود",
    "چود",
    "چوتیا",
    "چوتیے",
    "چوتیئے",
    "حرامی",
    "حرامزادہ",
    "حرامزادی",
    "کنجر",
    "کنجری",
    "کمینہ",
    "کمینے",
    "بیوقوف",
    "احمق",
    "گھٹیا",
    "ذلیل",
    "خبیث",
    "بدتمیز",
    "بدتمیزی",
    "کتا",
    "کتے",
    "گدا",
}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DetectRequest(BaseModel):
    mode: str
    text: Optional[str] = None
    audio: Optional[str] = None


class AudioToxicClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_classes=2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)


def artifacts_ready() -> bool:
    return (
        TEXT_MODEL_DIR.exists()
        and (TEXT_MODEL_DIR / "model.safetensors").exists()
        and AUDIO_MODEL_PATH.exists()
        and LABELS_PATH.exists()
    )


def ensure_artifacts() -> None:
    if artifacts_ready():
        return

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=MODEL_ZIP_NAME,
        repo_type="model",
    )

    extract_dir = ARTIFACTS_DIR / "_downloaded"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)

    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)

    source_dir = extract_dir / "content" / "drive" / "MyDrive"
    if not source_dir.exists():
        source_dir = extract_dir

    for name in ["Urtox_attempt1", "audio_toxic_classifier.pt", "label_classes.npy"]:
        source = source_dir / name
        destination = ARTIFACTS_DIR / name
        if destination.exists():
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)

    shutil.rmtree(extract_dir, ignore_errors=True)


def load_text_model():
    global TEXT_TOKENIZER, TEXT_MODEL

    if TEXT_TOKENIZER is not None and TEXT_MODEL is not None:
        return TEXT_TOKENIZER, TEXT_MODEL

    ensure_artifacts()
    TEXT_TOKENIZER = AutoTokenizer.from_pretrained(TEXT_MODEL_DIR)
    TEXT_MODEL = AutoModelForTokenClassification.from_pretrained(TEXT_MODEL_DIR)
    TEXT_MODEL.to(DEVICE)
    TEXT_MODEL.eval()
    return TEXT_TOKENIZER, TEXT_MODEL


def load_audio_model():
    global AUDIO_PROCESSOR, AUDIO_WAV2VEC_MODEL, AUDIO_CLASSIFIER, AUDIO_LABELS

    if (
        AUDIO_PROCESSOR is not None
        and AUDIO_WAV2VEC_MODEL is not None
        and AUDIO_CLASSIFIER is not None
        and AUDIO_LABELS is not None
    ):
        return AUDIO_PROCESSOR, AUDIO_WAV2VEC_MODEL, AUDIO_CLASSIFIER, AUDIO_LABELS

    ensure_artifacts()
    AUDIO_PROCESSOR = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    AUDIO_WAV2VEC_MODEL = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    AUDIO_WAV2VEC_MODEL.to(DEVICE)
    AUDIO_WAV2VEC_MODEL.eval()

    AUDIO_LABELS = np.load(LABELS_PATH, allow_pickle=True).tolist()
    AUDIO_CLASSIFIER = AudioToxicClassifier(num_classes=len(AUDIO_LABELS))
    AUDIO_CLASSIFIER.load_state_dict(torch.load(AUDIO_MODEL_PATH, map_location=DEVICE))
    AUDIO_CLASSIFIER.to(DEVICE)
    AUDIO_CLASSIFIER.eval()
    return AUDIO_PROCESSOR, AUDIO_WAV2VEC_MODEL, AUDIO_CLASSIFIER, AUDIO_LABELS


def load_whisper_model():
    global WHISPER_MODEL

    if WHISPER_MODEL is not None:
        return WHISPER_MODEL

    WHISPER_MODEL = whisper.load_model(WHISPER_MODEL_SIZE, device=str(DEVICE))
    return WHISPER_MODEL


def normalize_word(word: str) -> str:
    normalized = word.strip().strip(URDU_PUNCTUATION).lower()
    normalized = re.sub(r"[\u064b-\u065f\u0670]", "", normalized)
    return normalized.replace(" ", "")


def lexicon_match(word: str) -> bool:
    normalized = normalize_word(word)
    if not normalized:
        return False
    return normalized in TOXIC_LEXICON or any(term in normalized for term in TOXIC_LEXICON if len(term) >= 4)


@app.on_event("startup")
def startup_event():
    ensure_artifacts()


def predict_text(text: str):
    tokenizer, model = load_text_model()
    tokens = [token for token in text.split() if token]
    if not tokens:
        tokens = [" "]

    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding="max_length",
    )
    word_ids = encoding.word_ids(batch_index=0)
    model_inputs = {key: value.to(DEVICE) for key, value in encoding.items()}

    with torch.no_grad():
        outputs = model(**model_inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)[0].cpu()
        predictions = torch.argmax(probabilities, dim=-1).tolist()

    id2label = model.config.id2label
    previous_word_id = None
    word_results = []

    for token_index, word_id in enumerate(word_ids):
        if word_id is None or word_id == previous_word_id:
            continue

        model_label = id2label[int(predictions[token_index])]
        model_confidence = float(probabilities[token_index][predictions[token_index]])
        fallback_toxic = lexicon_match(tokens[word_id])
        label = model_label
        confidence = model_confidence

        if fallback_toxic and model_label == "O":
            label = "B-Toxic"
            confidence = max(model_confidence, 0.97)

        is_toxic = label in {"B-Toxic", "I-Toxic"}
        word_results.append(
            {
                "text": tokens[word_id],
                "toxic": is_toxic,
                "bioTag": label,
                "confidence": round(confidence, 4),
                "modelBioTag": model_label,
                "modelConfidence": round(model_confidence, 4),
                "source": "lexicon+model" if fallback_toxic and model_label == "O" else "model",
            }
        )
        previous_word_id = word_id

    toxic_words = [word for word in word_results if word["toxic"]]
    toxic_confidences = [word["confidence"] for word in toxic_words]
    confidence = max(toxic_confidences) if toxic_confidences else 1.0 - max(
        (word["confidence"] for word in word_results),
        default=0.0,
    )

    return {
        "isToxic": bool(toxic_words),
        "confidence": round(float(confidence), 4),
        "subLabel": "toxic" if toxic_words else "non-toxic",
        "subLabelConfidence": round(float(confidence), 4),
        "toxicSpanCount": count_toxic_spans(word_results),
        "transcript": None,
        "words": word_results,
        "xai": {
            "modelExplanation": "XLM-RoBERTa BIO token classification with a conservative Urdu abuse-word fallback for obvious missed slurs.",
            "topToxicTokens": [
                {
                    "token": word["text"],
                    "attribution": word["confidence"],
                    "confidence": word["confidence"],
                }
                for word in sorted(toxic_words, key=lambda item: item["confidence"], reverse=True)[:5]
            ],
            "integratedGradients": None,
        },
    }


def count_toxic_spans(words: list[dict]) -> int:
    span_count = 0
    previous_toxic = False
    for word in words:
        current_toxic = bool(word["toxic"])
        if current_toxic and not previous_toxic:
            span_count += 1
        previous_toxic = current_toxic
    return span_count


def decode_audio_to_tempfile(audio_payload: str) -> str:
    suffix = ".webm"
    if audio_payload.startswith("data:"):
        mime_type = audio_payload.split(";", 1)[0].replace("data:", "")
        if "webm" in mime_type:
            suffix = ".webm"
        elif "wav" in mime_type:
            suffix = ".wav"
        elif "mpeg" in mime_type or "mp3" in mime_type:
            suffix = ".mp3"
        elif "ogg" in mime_type:
            suffix = ".ogg"

    if "," in audio_payload:
        audio_payload = audio_payload.split(",", 1)[1]
    audio_bytes = base64.b64decode(audio_payload)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.write(audio_bytes)
    temp_file.close()
    return temp_file.name


def convert_audio_to_wav(input_path: str) -> str:
    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    output_file.close()
    command = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-t",
        "10",
        output_file.name,
    ]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_file.name


def transcribe_audio(temp_path: str) -> str:
    whisper_model = load_whisper_model()
    result = whisper_model.transcribe(
        temp_path,
        language="ur",
        task="transcribe",
        fp16=DEVICE.type == "cuda",
    )
    return (result.get("text") or "").strip()


def predict_audio(audio_payload: str) -> dict:
    processor, wav2vec_model, audio_classifier, labels = load_audio_model()
    temp_path = decode_audio_to_tempfile(audio_payload)
    wav_path = None

    try:
        wav_path = convert_audio_to_wav(temp_path)
        transcript = transcribe_audio(wav_path)
        span_result = predict_text(transcript) if transcript else {
            "isToxic": False,
            "confidence": 0.0,
            "subLabel": "non-toxic",
            "subLabelConfidence": 0.0,
            "toxicSpanCount": 0,
            "transcript": None,
            "words": [],
            "xai": {
                "modelExplanation": "Whisper did not return a transcript for this audio.",
                "topToxicTokens": [],
                "integratedGradients": None,
            },
        }

        waveform, sample_rate = sf.read(wav_path, dtype="float32")
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        if waveform.shape[0] > MAX_AUDIO_LENGTH:
            waveform = waveform[:MAX_AUDIO_LENGTH]

        inputs = processor(
            waveform,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

        with torch.no_grad():
            wav2vec_outputs = wav2vec_model(**inputs)
            embedding = wav2vec_outputs.last_hidden_state.mean(dim=1)
            logits = audio_classifier(embedding)
            probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()

        prediction_index = int(np.argmax(probabilities))
        predicted_label = str(labels[prediction_index])
        toxic_index = labels.index("toxic") if "toxic" in labels else prediction_index
        toxic_probability = float(probabilities[toxic_index])
        confidence = float(probabilities[prediction_index])
        is_audio_toxic = predicted_label == "toxic"
        is_toxic = is_audio_toxic or bool(span_result["isToxic"])
        combined_confidence = max(confidence if is_audio_toxic else 0.0, float(span_result["confidence"]))

        return {
            "isToxic": is_toxic,
            "confidence": round(combined_confidence, 4),
            "subLabel": "toxic" if is_toxic else "non-toxic",
            "subLabelConfidence": round(combined_confidence, 4),
            "toxicSpanCount": span_result["toxicSpanCount"],
            "transcript": transcript,
            "words": span_result["words"],
            "audio": {
                "label": predicted_label,
                "toxicProbability": round(toxic_probability, 4),
                "nonToxicProbability": round(float(probabilities[labels.index("non_toxic")]), 4)
                if "non_toxic" in labels
                else None,
            },
            "xai": {
                "modelExplanation": "Audio inference uses Whisper transcription for toxic-span detection plus Wav2Vec2 audio-level toxicity classification.",
                "topToxicTokens": span_result["xai"]["topToxicTokens"],
                "integratedGradients": span_result["xai"]["integratedGradients"],
            },
        }
    finally:
        Path(temp_path).unlink(missing_ok=True)
        if wav_path:
            Path(wav_path).unlink(missing_ok=True)


def audio_fallback_prediction(message: str = "Audio inference could not run.") -> dict:
    return {
        "isToxic": False,
        "confidence": 0.0,
        "subLabel": "audio-not-enabled",
        "subLabelConfidence": 0.0,
        "toxicSpanCount": 0,
        "transcript": message,
        "words": [],
        "xai": {
            "modelExplanation": message,
            "topToxicTokens": [],
            "integratedGradients": None,
        },
    }


@app.get("/")
def health():
    return {
        "status": "ok",
        "service": "urtox-api",
        "artifactSource": MODEL_REPO_ID,
        "artifactsReady": artifacts_ready(),
        "textModelLoaded": TEXT_MODEL is not None,
        "audioModelLoaded": AUDIO_CLASSIFIER is not None,
        "asrLoaded": WHISPER_MODEL is not None,
        "asrModel": f"openai-whisper/{WHISPER_MODEL_SIZE}",
        "device": str(DEVICE),
    }


@app.post("/detect")
def detect(payload: DetectRequest):
    if payload.mode == "audio":
        if not payload.audio:
            return audio_fallback_prediction("No audio payload was provided.")
        try:
            return predict_audio(payload.audio)
        except Exception as exc:
            return audio_fallback_prediction(f"Audio inference failed: {exc}")

    text = payload.text or "yeh toxic span detection result hai"
    return predict_text(text)
