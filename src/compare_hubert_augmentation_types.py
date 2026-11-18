
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


# ==================================================
# 1. SETTINGS
# ==================================================

HUBERT_FILE = Path("hubert_features.npz")

AUGMENTED_FILE = Path("hubert_augmented_train_features.npz")

STRETCH_FILE = Path("hubert_stretch_train_features.npz")

OUTPUT_DIR = Path("models")

N_SPLITS = 5
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
# 2. LOAD ORIGINAL HUBERT FEATURES
# ==================================================

print("Loading original HuBERT features...")

original = np.load(HUBERT_FILE)

X_all = original["X"]

y_all = np.array([
    LABEL_TO_ID[str(label)]
    for label in original["y"]
], dtype=np.int64)

actors_all = original["actors"]
train_idx = original["train_idx"]

X_original = X_all[train_idx]
y_original = y_all[train_idx]
actors = actors_all[train_idx]

print("Original training shape:", X_original.shape)


# ==================================================
# 3. LOAD NOISE AND STRETCH+NOISE FEATURES
# ==================================================

print("\nLoading noise augmentation features...")

augmented = np.load(AUGMENTED_FILE)

X_aug = augmented["X"]
y_aug = augmented["y"]
speakers_aug = augmented["speakers"]

indices_aug = augmented["original_indices"]
augmentation_ids = augmented["augmentation_ids"]

# ID 0 = background noise
# ID 1 = time stretching + background noise

noise_mask = augmentation_ids == 0
stretch_noise_mask = augmentation_ids == 1

X_noise = X_aug[noise_mask]
y_noise = y_aug[noise_mask]
actors_noise = speakers_aug[noise_mask]
indices_noise = indices_aug[noise_mask]

X_stretch_noise = X_aug[stretch_noise_mask]
y_stretch_noise = y_aug[stretch_noise_mask]
actors_stretch_noise = speakers_aug[stretch_noise_mask]
indices_stretch_noise = indices_aug[stretch_noise_mask]

print("Noise features:", X_noise.shape)
print("Stretch + noise features:", X_stretch_noise.shape)


# ==================================================
# 4. LOAD PURE TIME-STRETCH FEATURES
# ==================================================

print("\nLoading pure time-stretch features...")

stretch = np.load(STRETCH_FILE)

X_stretch = stretch["X"]
y_stretch = stretch["y"]
actors_stretch = stretch["speakers"]
indices_stretch = stretch["original_indices"]

print("Pure stretch features:", X_stretch.shape)


# ==================================================
# 5. VERIFY DATA ALIGNMENT
# ==================================================

print("\n========== DATA VALIDATION ==========")

if X_original.shape != (900, 768):
    raise ValueError("Unexpected original feature shape.")

if X_noise.shape != (900, 768):
    raise ValueError("Unexpected noise feature shape.")

if X_stretch_noise.shape != (900, 768):
    raise ValueError("Unexpected stretch+noise feature shape.")

if X_stretch.shape != (900, 768):
    raise ValueError("Unexpected pure stretch feature shape.")

# Confirm that every augmentation comes from
# one of the original training recordings.
for name, indices in [
    ("Noise", indices_noise),
    ("Stretch + noise", indices_stretch_noise),
    ("Pure stretch", indices_stretch)
]:

    if not np.isin(indices, train_idx).all():
        raise ValueError(
            f"{name} contains recordings outside training split."
        )

# Check labels and speakers against original metadata.
for name, indices, labels, speakers in [
    ("Noise", indices_noise, y_noise, actors_noise),
    ("Stretch + noise", indices_stretch_noise,
     y_stretch_noise, actors_stretch_noise),
    ("Pure stretch", indices_stretch, y_stretch,
     actors_stretch)
]:

    if not np.array_equal(labels, y_all[indices]):
        raise ValueError(f"{name} label mismatch!")

    if not np.array_equal(speakers, actors_all[indices]):
        raise ValueError(f"{name} speaker mismatch!")

# Each original recording should have exactly one
# version of each augmentation type.
for name, indices in [
    ("Noise", indices_noise),
    ("Stretch + noise", indices_stretch_noise),
    ("Pure stretch", indices_stretch)
]:

    unique, counts = np.unique(indices, return_counts=True)

    if len(unique) != 900 or not np.all(counts == 1):
        raise ValueError(
            f"Unexpected recording counts for {name}."
        )

# Check all feature arrays.
feature_arrays = [
    X_original,
    X_noise,
    X_stretch_noise,
    X_stretch
]

for array in feature_arrays:

    if not np.isfinite(array).all():
        raise ValueError("Invalid feature values detected.")

print("All alignment checks passed!")
print("Every augmentation matches its original recording.")


# ==================================================
# 6. PREPARE FOUR EXPERIMENTS
# ==================================================

experiments = {
    "Original only": {
        "X": X_original[:0],
        "y": y_original[:0]
    },

    "Original + noise": {
        "X": X_noise,
        "y": y_noise
    },

    "Original + pure stretch": {
        "X": X_stretch,
        "y": y_stretch
    },

    "Original + both augmentations": {
        "X": np.vstack([
            X_noise,
            X_stretch_noise
        ]),
        "y": np.concatenate([
            y_noise,
            y_stretch_noise
        ])
    }
}

print("\nEXPERIMENT CONFIGURATIONS")

for name, experiment in experiments.items():

    print(
        f"{name}: "
        f"{900 + len(experiment['X'])} training recordings"
    )


# ==================================================
# 7. CREATE FIVE SPEAKER-INDEPENDENT FOLDS
# ==================================================

group_kfold = GroupKFold(
    n_splits=N_SPLITS
)

fold_results = []

oof_predictions = {
    name: np.full(len(y_original), -1, dtype=np.int64)
    for name in experiments
}

print("\nStarting four-way comparison...")


# ==================================================
# 8. RUN THE FIVE FOLDS
# ==================================================

for fold, (train_positions, val_positions) in enumerate(
    group_kfold.split(
        X_original,
        y_original,
        groups=actors
    ),
    start=1
):

    print(f"\n========== FOLD {fold} ==========")

    X_train = X_original[train_positions]
    y_train = y_original[train_positions]

    X_val = X_original[val_positions]
    y_val = y_original[val_positions]

    train_actors = np.unique(actors[train_positions])
    val_actors = np.unique(actors[val_positions])

    # Speaker leakage check.
    if set(train_actors) & set(val_actors):
        raise ValueError("Speaker leakage detected!")

    print("Training actors:", train_actors)
    print("Validation actors:", val_actors)

    # ----------------------------------------------
    # Select only augmentations from training actors.
    # ----------------------------------------------

    noise_train_mask = np.isin(
        actors_noise,
        train_actors
    )

    stretch_noise_train_mask = np.isin(
        actors_stretch_noise,
        train_actors
    )

    stretch_train_mask = np.isin(
        actors_stretch,
        train_actors
    )

    fold_augmentations = {
        "Original only": (X_train[:0], y_train[:0]),

        "Original + noise": (
            X_noise[noise_train_mask],
            y_noise[noise_train_mask]
        ),

        "Original + pure stretch": (
            X_stretch[stretch_train_mask],
            y_stretch[stretch_train_mask]
        ),

        "Original + both augmentations": (
            np.vstack([
                X_noise[noise_train_mask],
                X_stretch_noise[stretch_noise_train_mask]
            ]),
            np.concatenate([
                y_noise[noise_train_mask],
                y_stretch_noise[stretch_noise_train_mask]
            ])
        )
    }

    # ----------------------------------------------
    # Train and evaluate each configuration.
    # ----------------------------------------------

    for name, (X_extra, y_extra) in fold_augmentations.items():

        if (
            np.isin(actors_noise, train_actors).sum() != 720
            if name == "Original + noise"
            else False
        ):
            raise ValueError("Unexpected noise training count.")

        if name == "Original + pure stretch":
            if len(X_extra) != len(X_train):
                raise ValueError("Unexpected pure stretch count.")

        if name == "Original + both augmentations":
            if len(X_extra) != 2 * len(X_train):
                raise ValueError("Unexpected combined augmentation count.")

        X_combined = np.vstack([
            X_train,
            X_extra
        ])

        y_combined = np.concatenate([
            y_train,
            y_extra
        ])

        # StandardScaler is fitted only on training recordings.
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                C=C_VALUE,
                max_iter=3000
            ))
        ])

        model.fit(X_combined, y_combined)

        predictions = model.predict(X_val)

        accuracy = accuracy_score(y_val, predictions)

        macro_f1 = f1_score(
            y_val,
            predictions,
            average="macro"
        )

        oof_predictions[name][val_positions] = predictions

        fold_results.append({
            "fold": fold,
            "experiment": name,
            "training_recordings": len(X_combined),
            "accuracy": accuracy,
            "macro_f1": macro_f1
        })

        print(
            f"{name}: Accuracy={accuracy:.4f} | "
            f"Macro-F1={macro_f1:.4f}"
        )


# ==================================================
# 9. CALCULATE POOLED OUT-OF-FOLD RESULTS
# ==================================================

print("\n========== FINAL FOUR-WAY COMPARISON ==========")

summary_results = []

for name, predictions in oof_predictions.items():

    if np.any(predictions == -1):
        raise ValueError(
            f"Missing predictions for {name}."
        )

    accuracy = accuracy_score(
        y_original,
        predictions
    )

    macro_f1 = f1_score(
        y_original,
        predictions,
        average="macro"
    )

    summary_results.append({
        "experiment": name,
        "accuracy": accuracy,
        "macro_f1": macro_f1
    })

    print(f"\n{name}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Macro-F1: {macro_f1:.4f} ({macro_f1 * 100:.2f}%)")


# ==================================================
# 10. DISPLAY FOLD RESULTS AND IMPROVEMENTS
# ==================================================

fold_df = pd.DataFrame(fold_results)
summary_df = pd.DataFrame(summary_results)

print("\n========== POOLED COMPARISON TABLE ==========")

print(summary_df.to_string(
    index=False,
    formatters={
        "accuracy": "{:.4f}".format,
        "macro_f1": "{:.4f}".format
    }
))

print("\n========== AVERAGE FOLD MACRO-F1 ==========")

print(
    fold_df.groupby("experiment")["macro_f1"]
    .agg(["mean", "std"])
    .round(4)
    .to_string()
)


# ==================================================
# 11. SAVE RESULTS
# ==================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

fold_df.to_csv(
    OUTPUT_DIR / "hubert_augmentation_types_folds.csv",
    index=False
)

summary_df.to_csv(
    OUTPUT_DIR / "hubert_augmentation_types_summary.csv",
    index=False
)

for name, predictions in oof_predictions.items():

    filename = (
        name.lower()
        .replace(" ", "_")
        .replace("+", "plus")
    )

    pd.DataFrame({
        "actual_emotion": y_original,
        "predicted_emotion": predictions,
        "actor": actors,
        "correct": y_original == predictions
    }).to_csv(
        OUTPUT_DIR / f"{filename}_oof_predictions.csv",
        index=False
    )

print("\nResults saved successfully!")
print("MISSION 34 COMPLETED!")
