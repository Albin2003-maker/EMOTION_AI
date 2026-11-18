
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import torch

from transformers import HubertModel, AutoFeatureExtractor


# ==================================================
# 1. SETTINGS
# ==================================================

DATASET_DIR = Path("dataset")

FEATURES_FILE = Path("hubert_features.npz")

OUTPUT_FILE = Path("hubert_augmented_train_features.npz")

MODEL_NAME = "facebook/hubert-base-ls960"

SAMPLE_RATE = 16000

NUM_AUGMENTATIONS = 2

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

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

rng = np.random.default_rng(42)


# ==================================================
# 2. LOAD TRAINING METADATA
# ==================================================

print("Loading RAVDESS training metadata...")

data = np.load(FEATURES_FILE)

train_idx = data["train_idx"]

filenames = data["filenames"]
actors = data["actors"]

raw_labels = data["y"]

y_all = np.array([
    LABEL_TO_ID[str(label)]
    for label in raw_labels
], dtype=np.int64)

train_filenames = filenames[train_idx]
train_labels = y_all[train_idx]
train_actors = actors[train_idx]

if len(train_idx) != 900:
    raise ValueError("Expected 900 training recordings.")

print("Training recordings:", len(train_idx))
print("Training actors:", np.unique(train_actors))


# ==================================================
# 3. LOCATE AUDIO FILES
# ==================================================

print("\nFinding audio files...")

audio_paths = list(DATASET_DIR.rglob("*.wav"))

filename_map = {
    path.name: path
    for path in audio_paths
}

missing = [
    str(name)
    for name in train_filenames
    if str(name) not in filename_map
]

if missing:
    raise FileNotFoundError(
        f"Missing audio files: {missing[:10]}"
    )

print("All 900 training audio files found!")


# ==================================================
# 4. LOAD PRETRAINED FROZEN HUBERT
# ==================================================

print("\nLoading HuBERT...")

feature_extractor = AutoFeatureExtractor.from_pretrained(
    MODEL_NAME
)

hubert = HubertModel.from_pretrained(
    MODEL_NAME
).to(DEVICE)

hubert.eval()

for parameter in hubert.parameters():
    parameter.requires_grad = False

print("HuBERT loaded!")
print("Device:", DEVICE)


# ==================================================
# 5. AUDIO AUGMENTATION FUNCTIONS
# ==================================================

def add_background_noise(audio, min_snr=25, max_snr=35):

    # Calculate the original audio's volume.
    rms = np.sqrt(np.mean(audio ** 2) + 1e-12)

    # Randomly select a signal-to-noise ratio.
    snr_db = rng.uniform(min_snr, max_snr)

    # Calculate how much noise to add.
    noise_level = rms / (10 ** (snr_db / 20))

    noise = rng.normal(
        0,
        noise_level,
        size=audio.shape
    ).astype(np.float32)

    return (audio + noise).astype(np.float32)


def augment_audio(audio, augmentation_id):

    augmented = audio.copy()

    if augmentation_id == 0:

        # Mild background noise.
        augmented = add_background_noise(
            augmented,
            min_snr=30,
            max_snr=35
        )

    elif augmentation_id == 1:

        # Mild speed variation without intentionally
        # changing the pitch.
        rate = rng.uniform(0.95, 1.05)

        augmented = librosa.effects.time_stretch(
            augmented,
            rate=rate
        )

        augmented = add_background_noise(
            augmented,
            min_snr=25,
            max_snr=30
        )

    return augmented.astype(np.float32)


# ==================================================
# 6. EXTRACT HUBERT FEATURES
# ==================================================

features = []
labels = []
speakers = []
original_indices = []
output_filenames = []
augmentation_ids = []

print("\nStarting augmented feature extraction...")

total = len(train_idx) * NUM_AUGMENTATIONS

with torch.inference_mode():

    for index, original_index in enumerate(train_idx, start=1):

        filename = str(filenames[original_index])

        path = filename_map[filename]

        # Read original training audio.
        audio, sr = sf.read(
            path,
            dtype="float32"
        )

        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        # Resample to 16 kHz.
        if sr != SAMPLE_RATE:
            audio = librosa.resample(
                audio,
                orig_sr=sr,
                target_sr=SAMPLE_RATE
            )

        # Generate two augmented versions.
        for augmentation_id in range(NUM_AUGMENTATIONS):

            augmented_audio = augment_audio(
                audio,
                augmentation_id
            )

            # Convert augmented waveform into HuBERT input.
            inputs = feature_extractor(
                augmented_audio,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt"
            )

            input_values = inputs.input_values.to(DEVICE)

            # Extract frame-level speech representations.
            outputs = hubert(
                input_values=input_values
            )

            # Mean pooling: 768-dimensional vector.
            pooled = outputs.last_hidden_state.mean(dim=1)

            vector = pooled.squeeze(0).cpu().numpy()

            features.append(vector)

            labels.append(
                train_labels[index - 1]
            )

            speakers.append(
                train_actors[index - 1]
            )

            original_indices.append(
                original_index
            )

            output_filenames.append(filename)

            augmentation_ids.append(
                augmentation_id
            )

        if index % 25 == 0:
            print(
                f"Processed {index}/{len(train_idx)} originals | "
                f"Generated {len(features)}/{total} augmented features"
            )


# ==================================================
# 7. CONVERT TO NUMPY ARRAYS
# ==================================================

X_augmented = np.asarray(
    features,
    dtype=np.float32
)

y_augmented = np.asarray(
    labels,
    dtype=np.int64
)

speakers = np.asarray(speakers)

original_indices = np.asarray(
    original_indices,
    dtype=np.int64
)

output_filenames = np.asarray(output_filenames)

augmentation_ids = np.asarray(
    augmentation_ids,
    dtype=np.int64
)


# ==================================================
# 8. VERIFY FEATURES
# ==================================================

print("\n========== EXTRACTION RESULTS ==========")

print("Feature shape:", X_augmented.shape)
print("Labels shape:", y_augmented.shape)
print("Speaker shape:", speakers.shape)

print(
    "All feature values finite:",
    np.isfinite(X_augmented).all()
)

if X_augmented.shape != (1800, 768):
    raise ValueError("Unexpected feature shape.")

if not np.isfinite(X_augmented).all():
    raise ValueError("Invalid feature values detected.")

for augmentation_id in range(NUM_AUGMENTATIONS):

    print(
        f"Augmentation {augmentation_id}:",
        np.sum(augmentation_ids == augmentation_id)
    )


# ==================================================
# 9. SAVE AUGMENTED FEATURES
# ==================================================

np.savez_compressed(
    OUTPUT_FILE,
    X=X_augmented,
    y=y_augmented,
    speakers=speakers,
    original_indices=original_indices,
    filenames=output_filenames,
    augmentation_ids=augmentation_ids,
    label_names=np.asarray(EMOTIONS)
)

print("\nSaved:", OUTPUT_FILE)

print("\nMISSION 33 — STEP 1 COMPLETED!")
