
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
import torch
from transformers import AutoFeatureExtractor, HubertModel


# --------------------------------------------------
# 1. PROJECT PATHS AND SETTINGS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = ROOT / "dataset"
SPLIT_PATH = ROOT / "cnn_split.npz"
MEL_PATH = ROOT / "mel_dataset.npz"

OUTPUT_PATH = ROOT / "hubert_features.npz"
CHECKPOINT_PATH = ROOT / "hubert_features_checkpoint.npz"

MODEL_NAME = "facebook/hubert-base-ls960"
SAMPLE_RATE = 16000
SAVE_EVERY = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)


# --------------------------------------------------
# 2. LOAD THE EXISTING DATASET AND SPLIT
# --------------------------------------------------

split = np.load(SPLIT_PATH, allow_pickle=True)
dataset = np.load(MEL_PATH, allow_pickle=True)

filenames = dataset["filenames"]
actors = dataset["actors"]
labels = dataset["y"]

if not np.array_equal(filenames, split["filenames"]):
    raise ValueError("Dataset filenames do not match the saved split!")

if len(filenames) != 1440:
    raise ValueError(f"Expected 1440 recordings, found {len(filenames)}")

print("Total recordings:", len(filenames))
print("Emotion classes:", sorted(set(labels.tolist())))


# --------------------------------------------------
# 3. LOAD THE PRETRAINED HUBERT MODEL
# --------------------------------------------------

print("\nLoading HuBERT feature extractor...")

feature_extractor = AutoFeatureExtractor.from_pretrained(
    MODEL_NAME
)

print("Loading HuBERT model...")

model = HubertModel.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

print("HuBERT loaded successfully!")


# --------------------------------------------------
# 4. CREATE OR RESUME FEATURE EXTRACTION
# --------------------------------------------------

if CHECKPOINT_PATH.exists():

    print("\nPrevious checkpoint found. Resuming...")

    checkpoint = np.load(CHECKPOINT_PATH)

    features = checkpoint["features"]
    completed = checkpoint["completed"]

else:

    print("\nStarting fresh feature extraction...")

    features = np.zeros((len(filenames), 768), dtype=np.float32)
    completed = np.zeros(len(filenames), dtype=bool)


# --------------------------------------------------
# 5. PROCESS ALL 1440 RECORDINGS
# --------------------------------------------------

for index, (actor, filename) in enumerate(zip(actors, filenames)):

    if completed[index]:
        continue

    audio_path = (
        DATASET_DIR /
        f"Actor_{int(actor):02d}" /
        str(filename)
    )

    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    # Read the WAV file.
    audio, original_rate = sf.read(
        audio_path,
        dtype="float32"
    )

    # Convert stereo audio into mono.
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Convert audio to HuBERT's required sample rate.
    if original_rate != SAMPLE_RATE:

        audio = librosa.resample(
            audio,
            orig_sr=original_rate,
            target_sr=SAMPLE_RATE
        )

    audio = np.asarray(audio, dtype=np.float32)

    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError(f"Invalid audio recording: {audio_path}")

    # Prepare the waveform for HuBERT.
    inputs = feature_extractor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt"
    )

    input_values = inputs.input_values.to(device)

    # Extract features without training HuBERT.
    with torch.inference_mode():

        outputs = model(input_values)

        recording_features = (
            outputs.last_hidden_state.mean(dim=1)
        )

    # Store the 768-dimensional feature vector.
    features[index] = (
        recording_features.squeeze(0).cpu().numpy()
    )

    completed[index] = True

    del inputs, input_values, outputs, recording_features

    # Display progress every 50 recordings.
    if completed.sum() % SAVE_EVERY == 0:

        np.savez_compressed(
            CHECKPOINT_PATH,
            features=features,
            completed=completed
        )

        print(
            f"Processed {completed.sum()}/{len(filenames)} recordings"
        )


# --------------------------------------------------
# 6. SAVE THE FINAL FEATURE DATASET
# --------------------------------------------------

if not completed.all():
    raise RuntimeError("Some recordings were not processed!")

np.savez_compressed(
    OUTPUT_PATH,
    X=features,
    y=labels,
    actors=actors,
    filenames=filenames,
    train_idx=split["train_idx"],
    val_idx=split["val_idx"],
    test_idx=split["test_idx"],
    train_actors=split["train_actors"],
    val_actors=split["val_actors"],
    test_actors=split["test_actors"]
)

print("\n--------------------------------")
print("HUBERT FEATURE EXTRACTION COMPLETE")
print("--------------------------------")

print("Feature matrix shape:", features.shape)
print("Training recordings:", len(split["train_idx"]))
print("Validation recordings:", len(split["val_idx"]))
print("Test recordings:", len(split["test_idx"]))
print("Saved to:", OUTPUT_PATH)

print("\nAll 1440 recordings processed successfully!")
