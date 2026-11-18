"""EMOTION AI inference: frozen HuBERT features + noise-augmented classifier.

The classifier is trained on original RAVDESS training recordings plus their
noise-augmented versions. At inference time, audio is converted to mono 16 kHz,
passed through frozen facebook/hubert-base-ls960, mean-pooled to 768 features,
and scored by the saved Logistic Regression pipeline.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import librosa
import numpy as np
import torch
from transformers import AutoFeatureExtractor, HubertModel


# ---------------------------------------------------------------------------
# Paths and model configuration
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
CLASSIFIER_PATH = MODEL_DIR / "hubert_noise_augmented_classifier.joblib"
MODEL_NAME = "facebook/hubert-base-ls960"
SAMPLE_RATE = 16_000
FEATURE_DIMENSION = 768

CLASS_NAMES = [
    "angry",
    "calm",
    "disgust",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised",
]

MODEL_DISPLAY_NAME = "HuBERT + Logistic Regression"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Load the classifier and frozen HuBERT once at application startup.
# ---------------------------------------------------------------------------
print("[EMOTION AI] Loading noise-augmented Logistic Regression classifier...")
if not CLASSIFIER_PATH.is_file():
    raise FileNotFoundError(
        f"Classifier checkpoint not found: {CLASSIFIER_PATH}. "
        "Run src/train_final_emotion_classifier.py first."
    )

classifier = joblib.load(CLASSIFIER_PATH)

n_features = getattr(classifier, "n_features_in_", None)
if n_features is not None and int(n_features) != FEATURE_DIMENSION:
    raise ValueError(
        f"Expected a {FEATURE_DIMENSION}-feature classifier, found {n_features}."
    )

classifier_classes = np.asarray(classifier.classes_)
expected_class_ids = np.arange(len(CLASS_NAMES))
if not np.array_equal(classifier_classes, expected_class_ids):
    raise ValueError(
        "Classifier class IDs do not match the expected emotion order. "
        f"Expected {expected_class_ids.tolist()}, found {classifier_classes.tolist()}."
    )

print(f"[EMOTION AI] Loading {MODEL_NAME}...")
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
if int(feature_extractor.sampling_rate) != SAMPLE_RATE:
    raise ValueError(
        f"Expected HuBERT sample rate {SAMPLE_RATE}, "
        f"but feature extractor reports {feature_extractor.sampling_rate}."
    )

hubert = HubertModel.from_pretrained(MODEL_NAME).to(DEVICE)
hubert.eval()
for parameter in hubert.parameters():
    parameter.requires_grad_(False)

print("[EMOTION AI] Models loaded successfully.")
print("[EMOTION AI] Device:", DEVICE)
print("[EMOTION AI] Emotion classes:", CLASS_NAMES)


# ---------------------------------------------------------------------------
# Audio preprocessing and HuBERT feature extraction
# ---------------------------------------------------------------------------
def _load_audio(audio_path: str | Path) -> np.ndarray:
    """Load audio as mono float32 and resample it to 16 kHz.

    librosa.load supports the formats Gradio can decode. Gradio is configured
    to provide a WAV filepath, but this function also accepts common formats.
    """
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError("The uploaded audio file is empty.")

    audio, original_sr = librosa.load(
        str(path),
        sr=None,
        mono=True,
        dtype=np.float32,
    )

    if audio.size == 0:
        raise ValueError("No audio samples could be read from this file.")
    if not np.isfinite(audio).all():
        raise ValueError("Audio contains invalid numeric values.")

    if int(original_sr) != SAMPLE_RATE:
        audio = librosa.resample(
            audio,
            orig_sr=int(original_sr),
            target_sr=SAMPLE_RATE,
        ).astype(np.float32, copy=False)

    if audio.size == 0:
        raise ValueError("Audio became empty after resampling.")

    return np.asarray(audio, dtype=np.float32)


def extract_hubert_features(audio_path: str | Path) -> np.ndarray:
    """Return one 768-dimensional mean-pooled HuBERT feature vector."""
    audio = _load_audio(audio_path)

    encoded = feature_extractor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    )
    input_values = encoded.input_values.to(DEVICE)

    with torch.inference_mode():
        outputs = hubert(input_values=input_values, return_dict=True)
        pooled = outputs.last_hidden_state.mean(dim=1)

    features = pooled.squeeze(0).float().cpu().numpy()
    features = np.asarray(features, dtype=np.float32)

    if features.shape != (FEATURE_DIMENSION,):
        raise ValueError(
            f"Expected HuBERT feature shape ({FEATURE_DIMENSION},), "
            f"received {features.shape}."
        )
    if not np.isfinite(features).all():
        raise ValueError("HuBERT produced invalid feature values.")

    return features.reshape(1, -1)


# ---------------------------------------------------------------------------
# Public prediction API used by app.py
# ---------------------------------------------------------------------------
def predict_emotion(audio_path: str | Path) -> dict[str, object]:
    """Predict one of the eight vocal-expression labels for an audio clip.

    Returned scores are the Logistic Regression classifier's class-probability
    estimates. They are not calibrated certainty and do not reveal a person's
    internal emotional state.
    """
    features = extract_hubert_features(audio_path)

    probabilities = classifier.predict_proba(features)[0]
    predicted_class_id = int(classifier.predict(features)[0])
    predicted_emotion = CLASS_NAMES[predicted_class_id]

    scores = {
        CLASS_NAMES[int(class_id)]: float(probability)
        for class_id, probability in zip(classifier.classes_, probabilities)
    }

    # Keep the complete eight-class order stable for the frontend.
    scores = {name: scores[name] for name in CLASS_NAMES}

    return {
        "predicted_emotion": predicted_emotion,
        "scores": scores,
        "model_name": MODEL_DISPLAY_NAME,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run EMOTION AI on one audio recording."
    )
    parser.add_argument("audio", help="Path to an audio file")
    args = parser.parse_args()

    result = predict_emotion(args.audio)
    print("Predicted emotion:", result["predicted_emotion"])
    print("Model:", result["model_name"])
    print("Class probability estimates:")
    for emotion, score in result["scores"].items():
        print(f"  {emotion:10s}: {score * 100:6.2f}%")
