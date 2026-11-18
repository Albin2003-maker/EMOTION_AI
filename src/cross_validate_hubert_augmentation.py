
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

# Keep only the original 900 training recordings.
X_original = X_all[train_idx]
y_original = y_all[train_idx]
actors = actors_all[train_idx]

print("Original training shape:", X_original.shape)
print("Training actors:", np.unique(actors))


# ==================================================
# 3. LOAD AUGMENTED FEATURES
# ==================================================

print("\nLoading augmented HuBERT features...")

augmented = np.load(AUGMENTED_FILE)

X_aug = augmented["X"]
y_aug = augmented["y"]
aug_speakers = augmented["speakers"]

aug_original_indices = augmented["original_indices"]
augmentation_ids = augmented["augmentation_ids"]

print("Augmented feature shape:", X_aug.shape)


# ==================================================
# 4. VERIFY DATA ALIGNMENT
# ==================================================

print("\nVerifying dataset alignment...")

if X_original.shape[1] != X_aug.shape[1]:
    raise ValueError("Feature dimensions do not match!")

if X_aug.shape != (1800, 768):
    raise ValueError("Unexpected augmented feature shape!")

if len(augmented["filenames"]) != len(X_aug):
    raise ValueError("Augmented filename count mismatch!")

if not np.isfinite(X_original).all():
    raise ValueError("Invalid original features found.")

if not np.isfinite(X_aug).all():
    raise ValueError("Invalid augmented features found.")

# Confirm every augmentation came from a training recording.
if not np.isin(
    aug_original_indices,
    train_idx
).all():
    raise ValueError("Augmented data contains non-training recordings.")

# Confirm every original recording has exactly two augmented versions.
unique_indices, counts = np.unique(
    aug_original_indices,
    return_counts=True
)

if len(unique_indices) != 900:
    raise ValueError("Expected 900 unique original recordings.")

if not np.all(counts == 2):
    raise ValueError("Each original must have exactly two augmented versions.")

# Confirm labels and speaker IDs remain correct.
if not np.array_equal(
    y_aug,
    y_all[aug_original_indices]
):
    raise ValueError("Augmented labels do not match original labels.")

if not np.array_equal(
    aug_speakers,
    actors_all[aug_original_indices]
):
    raise ValueError("Augmented speaker IDs do not match.")

if not set(np.unique(augmentation_ids)).issubset({0, 1}):
    raise ValueError("Unexpected augmentation type.")

print("All alignment checks passed!")
print("Unique original recordings:", len(unique_indices))
print("Augmented versions per original:", np.unique(counts))


# ==================================================
# 5. PREPARE FIVE SPEAKER-INDEPENDENT FOLDS
# ==================================================

group_kfold = GroupKFold(
    n_splits=N_SPLITS
)

baseline_results = []
augmented_results = []

baseline_oof = np.full(
    len(y_original),
    -1,
    dtype=np.int64
)

augmented_oof = np.full(
    len(y_original),
    -1,
    dtype=np.int64
)

print("\nStarting five-fold experiment...")


# ==================================================
# 6. RUN FIVE FOLDS
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

    # Original recordings.
    X_train = X_original[train_positions]
    y_train = y_original[train_positions]

    X_val = X_original[val_positions]
    y_val = y_original[val_positions]

    train_actors = np.unique(
        actors[train_positions]
    )

    val_actors = np.unique(
        actors[val_positions]
    )

    # Verify there is no speaker overlap.
    if set(train_actors) & set(val_actors):
        raise ValueError("Speaker leakage detected!")

    print("Training actors:", train_actors)
    print("Validation actors:", val_actors)

    # Select augmented versions belonging ONLY to training actors.
    augmented_mask = np.isin(
        aug_speakers,
        train_actors
    )

    X_aug_train = X_aug[augmented_mask]
    y_aug_train = y_aug[augmented_mask]

    selected_aug_actors = aug_speakers[augmented_mask]

    if set(selected_aug_actors) & set(val_actors):
        raise ValueError("Augmented speaker leakage detected!")

    print("Original training recordings:", len(X_train))
    print("Additional augmented recordings:", len(X_aug_train))
    print("Validation recordings:", len(X_val))

    # --------------------------------------------------
    # MODEL A: ORIGINAL TRAINING DATA
    # --------------------------------------------------

    baseline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=C_VALUE,
            max_iter=3000
        ))
    ])

    baseline.fit(
        X_train,
        y_train
    )

    baseline_predictions = baseline.predict(X_val)

    baseline_accuracy = accuracy_score(
        y_val,
        baseline_predictions
    )

    baseline_f1 = f1_score(
        y_val,
        baseline_predictions,
        average="macro"
    )

    baseline_oof[val_positions] = baseline_predictions

    # --------------------------------------------------
    # MODEL B: ORIGINAL + AUGMENTED TRAINING DATA
    # --------------------------------------------------

    X_combined = np.vstack([
        X_train,
        X_aug_train
    ])

    y_combined = np.concatenate([
        y_train,
        y_aug_train
    ])

    augmented_model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=C_VALUE,
            max_iter=3000
        ))
    ])

    augmented_model.fit(
        X_combined,
        y_combined
    )

    augmented_predictions = augmented_model.predict(X_val)

    augmented_accuracy = accuracy_score(
        y_val,
        augmented_predictions
    )

    augmented_f1 = f1_score(
        y_val,
        augmented_predictions,
        average="macro"
    )

    augmented_oof[val_positions] = augmented_predictions

    # --------------------------------------------------
    # SAVE FOLD RESULTS
    # --------------------------------------------------

    baseline_results.append({
        "fold": fold,
        "accuracy": baseline_accuracy,
        "macro_f1": baseline_f1
    })

    augmented_results.append({
        "fold": fold,
        "accuracy": augmented_accuracy,
        "macro_f1": augmented_f1
    })

    print("\nBASELINE")
    print(f"Accuracy: {baseline_accuracy:.4f}")
    print(f"Macro-F1: {baseline_f1:.4f}")

    print("\nWITH AUGMENTATION")
    print(f"Accuracy: {augmented_accuracy:.4f}")
    print(f"Macro-F1: {augmented_f1:.4f}")


# ==================================================
# 7. VERIFY ALL OUT-OF-FOLD PREDICTIONS
# ==================================================

if np.any(baseline_oof == -1):
    raise ValueError("Baseline predictions are incomplete.")

if np.any(augmented_oof == -1):
    raise ValueError("Augmented predictions are incomplete.")


# ==================================================
# 8. CALCULATE FINAL COMPARISON
# ==================================================

baseline_accuracy = accuracy_score(
    y_original,
    baseline_oof
)

baseline_f1 = f1_score(
    y_original,
    baseline_oof,
    average="macro"
)

augmented_accuracy = accuracy_score(
    y_original,
    augmented_oof
)

augmented_f1 = f1_score(
    y_original,
    augmented_oof,
    average="macro"
)

print("\n========== FINAL FIVE-FOLD COMPARISON ==========")

print("\nBASELINE — ORIGINAL DATA")

print(
    f"Accuracy: {baseline_accuracy:.4f} "
    f"({baseline_accuracy * 100:.2f}%)"
)

print(
    f"Macro-F1: {baseline_f1:.4f} "
    f"({baseline_f1 * 100:.2f}%)"
)

print("\nAUGMENTED — ORIGINAL + MODIFIED DATA")

print(
    f"Accuracy: {augmented_accuracy:.4f} "
    f"({augmented_accuracy * 100:.2f}%)"
)

print(
    f"Macro-F1: {augmented_f1:.4f} "
    f"({augmented_f1 * 100:.2f}%)"
)

print("\n========== IMPROVEMENT ==========")

print(
    "Accuracy change:",
    f"{(augmented_accuracy - baseline_accuracy) * 100:+.2f}",
    "percentage points"
)

print(
    "Macro-F1 change:",
    f"{(augmented_f1 - baseline_f1) * 100:+.2f}",
    "percentage points"
)


# ==================================================
# 9. SAVE RESULTS
# ==================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

baseline_df = pd.DataFrame(baseline_results)
augmented_df = pd.DataFrame(augmented_results)

fold_comparison = baseline_df.merge(
    augmented_df,
    on="fold",
    suffixes=("_baseline", "_augmented")
)

fold_comparison.to_csv(
    OUTPUT_DIR / "hubert_augmentation_fold_results.csv",
    index=False
)

summary = pd.DataFrame([
    {
        "model": "Original HuBERT",
        "accuracy": baseline_accuracy,
        "macro_f1": baseline_f1
    },
    {
        "model": "HuBERT with augmentation",
        "accuracy": augmented_accuracy,
        "macro_f1": augmented_f1
    }
])

summary.to_csv(
    OUTPUT_DIR / "hubert_augmentation_cv_summary.csv",
    index=False
)

print("\nResults saved:")
print("models/hubert_augmentation_fold_results.csv")
print("models/hubert_augmentation_cv_summary.csv")

print("\nMISSION 33 — STEP 2 COMPLETED!")
