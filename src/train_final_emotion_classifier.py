
from pathlib import Path
import json

import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# ==================================================
# 1. SETTINGS
# ==================================================

HUBERT_FILE = Path("hubert_features.npz")

AUGMENTED_FILE = Path(
    "hubert_augmented_train_features.npz"
)

MODEL_DIR = Path("models")

MODEL_FILE = MODEL_DIR / "hubert_noise_augmented_classifier.joblib"

INFO_FILE = MODEL_DIR / "hubert_noise_augmented_model_info.json"

C_VALUE = 0.03

EMOTIONS = [
    "angry",
    "calm",
    "disgust",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised"
]

LABEL_TO_ID = {
    emotion: index
    for index, emotion in enumerate(EMOTIONS)
}


# ==================================================
# 2. LOAD ORIGINAL TRAINING FEATURES
# ==================================================

print("Loading original HuBERT features...")

original = np.load(HUBERT_FILE)

X_all = original["X"]

y_all = np.array([
    LABEL_TO_ID[str(label)]
    for label in original["y"]
], dtype=np.int64)

train_idx = original["train_idx"]

X_original = X_all[train_idx]
y_original = y_all[train_idx]

original_filenames = original["filenames"][train_idx]

print("Original training features:", X_original.shape)


# ==================================================
# 3. LOAD NOISE-AUGMENTED FEATURES
# ==================================================

print("\nLoading noise-augmented features...")

augmented = np.load(AUGMENTED_FILE)

X_augmented_all = augmented["X"]

y_augmented_all = augmented["y"]

augmentation_ids = augmented["augmentation_ids"]

augmented_indices = augmented["original_indices"]

augmented_filenames = augmented["filenames"]

# ID 0 = background noise only.
noise_mask = augmentation_ids == 0

X_noise_raw = X_augmented_all[noise_mask]

y_noise_raw = y_augmented_all[noise_mask]

noise_indices = augmented_indices[noise_mask]

noise_filenames = augmented_filenames[noise_mask]

print("Noise features found:", X_noise_raw.shape)


# ==================================================
# 4. ALIGN ORIGINAL AND AUGMENTED RECORDINGS
# ==================================================

print("\nVerifying feature alignment...")

if len(noise_indices) != 900:
    raise ValueError("Expected exactly 900 noise recordings.")

if len(np.unique(noise_indices)) != 900:
    raise ValueError("Duplicate original recording indices found.")

# Map each original recording index to its noise feature.
noise_position = {
    int(index): position
    for position, index in enumerate(noise_indices)
}

# Restore the exact ordering of the original training split.
noise_order = np.array([
    noise_position[int(index)]
    for index in train_idx
])

X_noise = X_noise_raw[noise_order]

y_noise = y_noise_raw[noise_order]

noise_filenames = noise_filenames[noise_order]

if not np.array_equal(y_original, y_noise):
    raise ValueError("Original and augmented labels do not match.")

if not np.array_equal(
    original_filenames.astype(str),
    noise_filenames.astype(str)
):
    raise ValueError("Original and augmented filenames do not match.")

if not np.isfinite(X_original).all():
    raise ValueError("Invalid original features.")

if not np.isfinite(X_noise).all():
    raise ValueError("Invalid augmented features.")

print("Feature alignment verified!")


# ==================================================
# 5. COMBINE TRAINING DATA
# ==================================================

X_final = np.vstack([
    X_original,
    X_noise
])

y_final = np.concatenate([
    y_original,
    y_noise
])

print("\nFINAL TRAINING DATA")
print("Combined feature shape:", X_final.shape)
print("Combined label shape:", y_final.shape)


# ==================================================
# 6. DISPLAY EMOTION DISTRIBUTION
# ==================================================

print("\nEmotion distribution:")

for emotion_id, emotion_name in enumerate(EMOTIONS):

    count = np.sum(y_final == emotion_id)

    print(f"{emotion_name:10s}: {count}")


# ==================================================
# 7. TRAIN THE FINAL CLASSIFIER
# ==================================================

print("\nTraining Logistic Regression...")

model = Pipeline([
    ("scaler", StandardScaler()),

    ("classifier", LogisticRegression(
        C=C_VALUE,
        max_iter=3000
    ))
])

model.fit(
    X_final,
    y_final
)

print("Classifier training completed!")


# ==================================================
# 8. SAVE THE MODEL
# ==================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(
    model,
    MODEL_FILE
)

model_info = {
    "model_name": "HuBERT Noise-Augmented Emotion Classifier",
    "feature_extractor": "facebook/hubert-base-ls960",
    "feature_dimension": 768,
    "feature_method": "Frozen HuBERT mean pooling",
    "classifier": "Logistic Regression",
    "C": C_VALUE,
    "training_recordings": int(len(X_final)),
    "original_recordings": int(len(X_original)),
    "noise_augmented_recordings": int(len(X_noise)),
    "emotion_names": EMOTIONS,
    "cross_validation_accuracy": 0.6944,
    "cross_validation_macro_f1": 0.6847
}

with open(INFO_FILE, "w", encoding="utf-8") as file:
    json.dump(
        model_info,
        file,
        indent=4
    )


# ==================================================
# 9. VERIFY SAVED MODEL
# ==================================================

loaded_model = joblib.load(MODEL_FILE)

sample_prediction = loaded_model.predict(
    X_original[:1]
)

print("\n========== FINAL MODEL CHECK ==========")

print("Model file:", MODEL_FILE)
print("Metadata file:", INFO_FILE)
print("Training recordings:", len(X_final))
print("Features per recording:", X_final.shape[1])

print(
    "Sample prediction:",
    EMOTIONS[int(sample_prediction[0])]
)

print("\nFINAL EMOTION CLASSIFIER SAVED SUCCESSFULLY!")
