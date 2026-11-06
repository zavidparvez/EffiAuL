# ================================================================
# EffiAuL Reviewer-Revision Experimental Framework
# Version 4: four-class reviewer-strengthened EffiAuL benchmark + non-inferiority statistics
# ================================================================
# Implements:
#   - Fixed stratified 70/15/15 Train/Validation/Test split from ONE folder
#   - DenseNet121 baseline
#   - Influence-guided EffiAuL
#   - No-deletion fine-tuning control
#   - Full retraining reference
#   - SCRUB-style teacher/student unlearning (Kurmanji et al., NeurIPS 2023)
#   - SISA (Sharded, Isolated, Sliced, Aggregated) training/unlearning
#   - Influence-function-family baseline using a diagonal-Fisher/Hessian
#     approximation on the classifier head (IF-Diag)
#   - Retain/forget/test utility metrics
#   - Jensen-Shannon divergence (empirical only)
#   - Distance to full-retraining predictions
#   - Forget-set prediction change
#   - Simple loss-based membership-inference evaluation
#   - Runtime and process-memory reporting
#   - Bootstrap AUC confidence intervals
#   - Repeated training seeds and paired significance tests
#
# IMPORTANT SCIENTIFIC NOTES
# ------------------------------------------------
# 1. JSD is an empirical distribution-shift measure, NOT a formal
#    certificate of complete forgetting.
# 2. IF-Diag is an influence-function-family approximation, not an exact
#    inverse-Hessian influence function over the complete DenseNet.
# 3. SISA and SCRUB are implemented in the spirit of their published
#    algorithms, adapted to this TensorFlow/DenseNet medical-image setup.
# 4. Certified removal methods such as Guo et al. (ICML 2020) target
#    linear classifiers; do not label this DenseNet experiment as formally
#    certified removal unless you implement a method with guarantees that
#    apply to this neural-network setting.
# 5. Full retraining on D_train \ D_forget is the reference
#    for empirical comparisons; it is not a formal privacy certificate.
#
# Expected dataset layout:
#
# ./Dataset/CovidxRay4Class/
#     Class_1/
#         image1.jpg
#         ...
#     Class_2/
#         ...
#     [Class_3/... for multiclass]
#
# Install:
# pip install tensorflow scikit-learn scipy pandas matplotlib psutil pillow
# ================================================================

import os
import gc
import json
import time
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psutil
import tensorflow as tf

from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from scipy.spatial.distance import jensenshannon
from scipy.stats import wilcoxon, ttest_rel, ttest_1samp
from types import SimpleNamespace

warnings.filterwarnings("ignore")

# ================================================================
# 1. CONFIGURATION
# ================================================================

MASTER_DATASET_DIR = os.environ.get("EFFIAUL_DATA_DIR", "./Dataset/CovidxRay4Class")
RESULTS_DIR = os.environ.get("EFFIAUL_RESULTS_DIR", "./results/four_class")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SPLIT_SEED = 42

TRAINING_SEEDS = [int(s) for s in os.environ.get("EFFIAUL_SEEDS", "11,22,33,44,55").split(",") if s.strip()]

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

BASELINE_LR = 1e-4
BASELINE_MAX_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5
BASELINE_TRAINABLE_BACKBONE_LAYERS = 30

# Forget-set configuration
N_FORGET = int(os.environ.get("EFFIAUL_N_FORGET", "20"))
# "low", "high", or "random"
# Random is retained to represent an arbitrary deletion request; the SAME
# deletion indices are used across all training seeds for matched comparisons.
FORGET_SELECTION = os.environ.get("EFFIAUL_FORGET_SELECTION", "random")
FORGET_SET_SEED = 2026

# Utility threshold tau = baseline validation weighted-F1 - tolerance
UTILITY_TOLERANCE = 0.02

# EffiAuL StrategySelector
LOW_INFLUENCE_THRESHOLD = 0.33
HIGH_INFLUENCE_THRESHOLD = 0.67

# ------------------------------------------------
# EffiAuL reviewer-strengthened update objective
# ------------------------------------------------
# Retain utility is protected by supervised CE + teacher distillation.
# Forget influence is reduced by a small influence-adaptive ascent term.
# Selection of the final epoch uses ONLY validation utility + forget-set
# behavior; the held-out test set is never used for model selection.
EFFIAUL_RETAIN_CE_WEIGHT = 1.0
EFFIAUL_RETAIN_KD_WEIGHT = 0.50
EFFIAUL_FORGET_ASCENT_BASE_WEIGHT = 0.05
EFFIAUL_GRAD_CLIP_NORM = 5.0
EFFIAUL_MAX_EPOCHS = 8
EFFIAUL_MIN_EPOCHS = 2
EFFIAUL_SELECTION_PATIENCE = 2

# Primary statistical claim should be utility NON-INFERIORITY rather than
# "higher F1", because this dataset is already near the performance ceiling.
F1_NONINFERIORITY_MARGIN = 0.01

# Bootstrap
N_BOOTSTRAP = int(os.environ.get("EFFIAUL_BOOTSTRAP", "1000"))

# SCRUB hyperparameters
SCRUB_ALPHA = 1.0       # retain KL term
SCRUB_GAMMA = 1.0       # retain task-loss term
SCRUB_MAX_LR = 1e-5
SCRUB_MIN_LR = 1e-5
SCRUB_ALTERNATING_EPOCHS = 3
SCRUB_FINAL_MIN_EPOCHS = 2
SCRUB_TEMPERATURE = 1.0

# SISA
SISA_NUM_SHARDS = 3
SISA_NUM_SLICES = 3
SISA_EPOCHS_PER_SLICE = 2
SISA_LR = 1e-4
SISA_TRAINABLE_LAYERS = 30

# Influence-function-family diagonal approximation
IF_DIAG_DAMPING = 1e-3
IF_DIAG_SCALE = 1.0
IF_HEAD_ONLY = True

VALID_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}


# ================================================================
# 2. REPRODUCIBILITY / SYSTEM UTILITIES
# ================================================================

def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def clear_memory():
    tf.keras.backend.clear_session()
    gc.collect()


def current_memory_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ================================================================
# 3. DATASET SCAN AND FIXED 70/15/15 SPLIT
# ================================================================

def scan_master_dataset():
    """
    Scan the four-class chest-X-ray dataset and return ONE master dataframe.

    Supported layouts
    -----------------
    A) Existing layout (no physical reorganisation required):

       ./Dataset/CovidxRay4Class/
           Train/
               Class_A/...
               Class_B/...
               Class_C/...
               Class_D/...
           Validate/
               Class_A/...
               Class_B/...
               Class_C/...
               Class_D/...

       The old Train and Validate folders are combined first, then this script
       creates a NEW fixed stratified 70/15/15 train/validation/test split.

    B) One-master-folder layout:

       ./Dataset/CovidxRay4Class/
           Class_A/...
           Class_B/...
           Class_C/...
           Class_D/...

    This prevents the revised manuscript from describing one dataset while the
    reviewer-facing experiment silently uses another split or class setting.
    """
    root = Path(MASTER_DATASET_DIR)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {MASTER_DATASET_DIR}")

    rows = []

    # Prefer the user's existing Train/Validate directory structure when found.
    legacy_split_dirs = [root / "Train", root / "Validate"]
    existing_split_dirs = [p for p in legacy_split_dirs if p.exists() and p.is_dir()]

    if existing_split_dirs:
        print("Detected existing Train/Validate layout; combining all images before")
        print("creating the fixed reviewer-facing 70/15/15 split.")

        # Require both folders if using this layout so no data are silently omitted.
        missing = [p.name for p in legacy_split_dirs if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Detected a legacy split layout but the following folder(s) are missing: "
                + ", ".join(missing)
            )

        class_name_sets = []
        for split_dir in legacy_split_dirs:
            class_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()])
            class_name_sets.append({p.name for p in class_dirs})

            for class_dir in class_dirs:
                for p in class_dir.rglob("*"):
                    if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS:
                        rows.append({
                            "filepath": str(p.resolve()),
                            "label": class_dir.name,
                            "original_partition": split_dir.name
                        })

        if class_name_sets[0] != class_name_sets[1]:
            raise ValueError(
                "Train and Validate must contain the same class folders. "
                f"Train={sorted(class_name_sets[0])}; "
                f"Validate={sorted(class_name_sets[1])}"
            )

    else:
        # Direct master-folder layout: immediate subdirectories are the classes.
        class_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
        for class_dir in class_dirs:
            for p in class_dir.rglob("*"):
                if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS:
                    rows.append({
                        "filepath": str(p.resolve()),
                        "label": class_dir.name,
                        "original_partition": "master"
                    })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No supported image files were found.")

    # Protect against duplicate image paths.
    duplicate_paths = int(df["filepath"].duplicated().sum())
    if duplicate_paths:
        raise ValueError(
            f"Found {duplicate_paths} duplicate file paths while building the master dataset."
        )

    class_counts = df["label"].value_counts().sort_index()

    # This entry point reproduces the additional four-class validation setting.
    if df["label"].nunique() != 4:
        raise ValueError(
            f"Expected exactly 4 classes, but found {df['label'].nunique()}: "
            f"{sorted(df['label'].unique())}"
        )

    # Check minimum class size for two stratified splits.
    if (class_counts < 4).any():
        raise ValueError(
            "Each class should have at least 4 images for a reliable 70/15/15 "
            "stratified split. Counts:\n" + str(class_counts)
        )

    ensure_dir(RESULTS_DIR)
    df.to_csv(os.path.join(RESULTS_DIR, "master_dataset_manifest.csv"), index=False)

    counts_df = class_counts.rename("count").reset_index().rename(columns={"index": "class"})
    counts_df.to_csv(os.path.join(RESULTS_DIR, "master_dataset_class_counts.csv"), index=False)

    print("\nMASTER FOUR-CLASS DATASET")
    print("=" * 70)
    print(f"Dataset root: {root.resolve()}")
    print(f"Total images: {len(df)}")
    print(f"Classes: {df['label'].nunique()}")
    print(class_counts)
    print("\nIMPORTANT: Use these printed/scanned counts as the source of truth")
    print("for the revised Methods and dataset table.")

    return df


def create_fixed_splits(df):
    if not np.isclose(TRAIN_RATIO + VAL_RATIO + TEST_RATIO, 1.0):
        raise ValueError("TRAIN_RATIO + VAL_RATIO + TEST_RATIO must equal 1.")

    train_df, temp_df = train_test_split(
        df,
        test_size=(VAL_RATIO + TEST_RATIO),
        stratify=df["label"],
        random_state=SPLIT_SEED
    )

    relative_test = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        stratify=temp_df["label"],
        random_state=SPLIT_SEED
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    a = set(train_df.filepath)
    b = set(val_df.filepath)
    c = set(test_df.filepath)
    assert not (a & b)
    assert not (a & c)
    assert not (b & c)

    ensure_dir(RESULTS_DIR)
    train_df.to_csv(os.path.join(RESULTS_DIR, "train_split.csv"), index=False)
    val_df.to_csv(os.path.join(RESULTS_DIR, "validation_split.csv"), index=False)
    test_df.to_csv(os.path.join(RESULTS_DIR, "test_split.csv"), index=False)

    print("\nFIXED STRATIFIED SPLIT")
    print("=" * 70)
    for name, part in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        print(f"\n{name}: {len(part)} ({100*len(part)/len(df):.2f}%)")
        print(part["label"].value_counts())

    return train_df, val_df, test_df


def determine_problem_type(train_df):
    class_names = sorted(train_df["label"].unique())
    n_classes = len(class_names)
    class_mode = "binary" if n_classes == 2 else "categorical"
    return class_names, n_classes, class_mode


# ================================================================
# 4. DATA GENERATORS
# ================================================================

def make_datagens():
    train_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest"
    )
    eval_datagen = ImageDataGenerator(rescale=1.0/255.0)
    return train_datagen, eval_datagen


def dataframe_generator(df, class_names, class_mode, train=False,
                        shuffle=False, seed=42, batch_size=BATCH_SIZE):
    train_datagen, eval_datagen = make_datagens()
    datagen = train_datagen if train else eval_datagen
    return datagen.flow_from_dataframe(
        dataframe=df,
        x_col="filepath",
        y_col="label",
        classes=class_names,
        class_mode=class_mode,
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        validate_filenames=True
    )


def create_main_generators(train_df, val_df, test_df, class_names, class_mode, seed):
    train_gen = dataframe_generator(
        train_df, class_names, class_mode, train=True, shuffle=True, seed=seed
    )
    # batch_size=1 and shuffle=False keep exact row-index mapping for influence scores
    influence_gen = dataframe_generator(
        train_df, class_names, class_mode, train=False, shuffle=False,
        seed=seed, batch_size=1
    )
    val_gen = dataframe_generator(
        val_df, class_names, class_mode, train=False, shuffle=False, seed=seed
    )
    test_gen = dataframe_generator(
        test_df, class_names, class_mode, train=False, shuffle=False, seed=seed
    )
    return train_gen, influence_gen, val_gen, test_gen


# ================================================================
# 5. MODEL
# ================================================================

def compile_model(model, n_classes, lr):
    if n_classes == 2:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr),
            loss="binary_crossentropy",
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.AUC(name="auc")
            ]
        )
    else:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr),
            loss="categorical_crossentropy",
            metrics=[
                tf.keras.metrics.CategoricalAccuracy(name="accuracy")
            ]
        )


def build_densenet_model(n_classes, lr=BASELINE_LR,
                         trainable_backbone_layers=BASELINE_TRAINABLE_BACKBONE_LAYERS):
    """
    Build DenseNet121 as an EXPLICIT NESTED backbone.

    Why this matters:
    -----------------
    The previous implementation constructed the final Keras Model using
    `Model(backbone.input, out)`. In recent TensorFlow/Keras versions this
    can flatten DenseNet's internal layers into the outer model, meaning
    there is no top-level layer whose name contains "densenet". Consequently
    `get_backbone()` failed during EffiAuL with:

        ValueError: DenseNet121 backbone not found.

    Here we explicitly create a new Input tensor and CALL the DenseNet model
    on it. DenseNet121 therefore remains a nested keras.Model layer and can
    be reliably found and reconfigured later.
    """

    inputs = tf.keras.Input(
        shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        name="image_input"
    )

    backbone = DenseNet121(
        weights="imagenet",
        include_top=False,
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    )

    # Freeze entire backbone first, then selectively unfreeze the tail.
    for layer in backbone.layers:
        layer.trainable = False

    n_layers = int(max(0, min(trainable_backbone_layers, len(backbone.layers))))
    if n_layers > 0:
        for layer in backbone.layers[-n_layers:]:
            layer.trainable = True

    # IMPORTANT: calling the backbone on a new Input keeps it as a nested
    # Model inside the final model instead of flattening it.
    x = backbone(inputs)

    x = GlobalAveragePooling2D(name="gap")(x)
    x = Dropout(0.5, name="dropout_1")(x)
    x = Dense(
        256,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.01),
        name="classification_dense"
    )(x)
    x = Dropout(0.5, name="dropout_2")(x)

    if n_classes == 2:
        outputs = Dense(1, activation="sigmoid", name="prediction")(x)
    else:
        outputs = Dense(n_classes, activation="softmax", name="prediction")(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="EffiAuL_DenseNet121"
    )

    # Keep an explicit Python reference as an additional robust fallback.
    model._effiaul_backbone = backbone

    compile_model(model, n_classes, lr)
    return model


def get_backbone(model):
    """
    Return the DenseNet121 backbone robustly across TensorFlow/Keras versions.

    Supports:
      1. the explicit `_effiaul_backbone` reference used by this script;
      2. a nested keras.Model whose name contains 'densenet';
      3. a nested model that structurally looks like DenseNet121.

    A diagnostic error is raised only if all strategies fail.
    """

    # Preferred path for models built by build_densenet_model().
    backbone = getattr(model, "_effiaul_backbone", None)
    if backbone is not None:
        return backbone

    # Standard nested-model lookup.
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "densenet" in layer.name.lower():
            return layer

    # Structural fallback: identify a nested model containing characteristic
    # DenseNet layer names.
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            internal_names = {sub.name for sub in layer.layers}
            if (
                "conv1_conv" in internal_names
                and any(name.startswith("conv5_block") for name in internal_names)
            ):
                return layer

    layer_names = [layer.name for layer in model.layers]
    raise ValueError(
        "DenseNet121 backbone could not be located. "
        f"Top-level model layers are: {layer_names}. "
        "Rebuild the model with the corrected build_densenet_model() function."
    )


def configure_trainable_layers(model, n_classes, n_layers, lr):
    """
    Freeze DenseNet121 and unfreeze exactly the final `n_layers` backbone
    layers, while keeping the classification head trainable.
    """

    backbone = get_backbone(model)

    requested = int(n_layers)
    if requested < 0:
        raise ValueError("n_layers must be >= 0.")

    requested = min(requested, len(backbone.layers))

    for layer in backbone.layers:
        layer.trainable = False

    if requested > 0:
        for layer in backbone.layers[-requested:]:
            layer.trainable = True

    # The backbone is one nested layer in the outer model. All custom head
    # layers remain trainable.
    for layer in model.layers:
        if layer is not backbone:
            layer.trainable = True

    # Recompile after changing trainability so the optimizer tracks the new
    # trainable-variable set correctly.
    compile_model(model, n_classes, lr)

    print(
        f"Configured DenseNet121: {requested}/{len(backbone.layers)} "
        f"backbone layers trainable; learning_rate={lr:.2e}"
    )

    return model


def early_stopping():
    return EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1
    )


def timed_fit(model, train_gen, val_gen, epochs, callbacks=True):
    m0 = current_memory_mb()
    t0 = time.perf_counter()
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[early_stopping()] if callbacks else [],
        verbose=1
    )
    runtime = time.perf_counter() - t0
    memory_delta = current_memory_mb() - m0
    return history, runtime, memory_delta


# ================================================================
# 6. PREDICTION / EVALUATION
# ================================================================

def assert_evaluation_generator(generator, name="evaluation generator"):
    # A shuffled iterator would break correspondence between model.predict()
    # outputs and generator.classes/generator.labels.
    if bool(getattr(generator, "shuffle", False)):
        raise ValueError(
            f"{name} must use shuffle=False for reproducible evaluation."
        )


def get_generator_classes(generator):
    """Return generator labels as a 1-D NumPy integer array.

    Keras/TensorFlow versions are inconsistent here: DataFrameIterator.classes
    may be a Python list in some releases and a NumPy array in others.
    This helper also supports the alternative `labels` attribute used by
    some iterator implementations.
    """
    if hasattr(generator, "classes"):
        labels = generator.classes
    elif hasattr(generator, "labels"):
        labels = generator.labels
    else:
        raise AttributeError(
            "The supplied generator exposes neither 'classes' nor 'labels'. "
            "Use a Keras DirectoryIterator/DataFrameIterator with shuffle=False "
            "for evaluation."
        )

    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    return labels


def get_predictions(model, generator, n_classes):
    assert_evaluation_generator(generator)
    # Evaluation generators in this script are always created with
    # shuffle=False, so prediction order matches the stored class labels.
    generator.reset()
    prob = np.asarray(model.predict(generator, verbose=0))
    y_true = get_generator_classes(generator)

    if n_classes == 2:
        prob = prob.reshape(-1)
        y_pred = (prob >= 0.5).astype(np.int64)
    else:
        if prob.ndim != 2 or prob.shape[1] != n_classes:
            raise ValueError(
                f"Expected multiclass probabilities with shape (N, {n_classes}), "
                f"but received {prob.shape}."
            )
        y_pred = np.argmax(prob, axis=1).astype(np.int64)

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Prediction/label length mismatch: {len(y_pred)} predictions vs "
            f"{len(y_true)} labels. Ensure evaluation generators use shuffle=False."
        )

    return y_true, prob, y_pred


def metrics_from_predictions(y_true, prob, y_pred, n_classes):
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    prob = np.asarray(prob)

    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }

    try:
        if n_classes == 2:
            result["auc"] = float(roc_auc_score(y_true, prob))
        else:
            result["auc"] = float(
                roc_auc_score(y_true, prob, multi_class="ovr", average="weighted")
            )
    except Exception:
        result["auc"] = np.nan

    return result


def evaluate_model(model, generator, n_classes):
    y_true, prob, y_pred = get_predictions(model, generator, n_classes)
    return metrics_from_predictions(y_true, prob, y_pred, n_classes), y_true, prob, y_pred


def bootstrap_auc_ci(y_true, prob, n_classes, seed, n_bootstrap=N_BOOTSTRAP):
    rng = np.random.default_rng(seed)
    values = []
    n = len(y_true)

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        ys = y_true[idx]
        if len(np.unique(ys)) < n_classes:
            continue
        try:
            if n_classes == 2:
                score = roc_auc_score(ys, prob[idx])
            else:
                score = roc_auc_score(
                    ys, prob[idx, :], multi_class="ovr", average="weighted"
                )
            values.append(score)
        except Exception:
            pass

    if not values:
        return np.nan, np.nan
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


# ================================================================
# 7. INFLUENCE-GUIDED SAMPLE SCORING
# ================================================================
# Empirical gradient-norm influence proxy:
# I(z_i) = || grad_theta L(z_i; theta) ||_2
# ================================================================

def task_loss(labels, predictions, n_classes, reduction=True):
    if n_classes == 2:
        labels = tf.reshape(tf.cast(labels, tf.float32), (-1, 1))
        loss = tf.keras.losses.binary_crossentropy(labels, predictions)
    else:
        labels = tf.cast(labels, tf.float32)
        loss = tf.keras.losses.categorical_crossentropy(labels, predictions)
    return tf.reduce_mean(loss) if reduction else loss


def compute_single_influence_score(model, images, labels, n_classes):
    with tf.GradientTape() as tape:
        pred = model(images, training=False)
        loss = task_loss(labels, pred, n_classes)

    grads = tape.gradient(loss, model.trainable_variables)
    grads = [g for g in grads if g is not None]
    if not grads:
        return 0.0

    norm2 = tf.add_n([tf.reduce_sum(tf.square(g)) for g in grads])
    return float(tf.sqrt(norm2).numpy())


def calculate_influence_scores(model, influence_gen, train_df, n_classes):
    influence_gen.reset()
    raw = []
    for i in range(len(train_df)):
        x, y = influence_gen[i]
        raw.append(compute_single_influence_score(model, x, y, n_classes))
        if (i + 1) % 50 == 0 or (i + 1) == len(train_df):
            print(f"Influence score: {i+1}/{len(train_df)}")

    raw = np.asarray(raw, dtype=np.float64)
    norm = (raw - raw.min()) / (raw.max() - raw.min() + 1e-12)

    out = train_df.copy()
    out["sample_index"] = np.arange(len(train_df))
    out["raw_influence"] = raw
    out["normalized_influence"] = norm
    return out, norm


def select_forget_indices(scores, n_forget, strategy, seed):
    if n_forget >= len(scores):
        raise ValueError("N_FORGET must be smaller than training set size.")

    rng = np.random.default_rng(seed)
    strategy = strategy.lower()

    if strategy == "low":
        idx = np.argsort(scores)[:n_forget]
    elif strategy == "high":
        idx = np.argsort(scores)[-n_forget:]
    elif strategy == "random":
        idx = rng.choice(len(scores), size=n_forget, replace=False)
    else:
        raise ValueError("FORGET_SELECTION must be low, high, or random.")

    return np.asarray(idx, dtype=int)


def create_retain_forget_dataframes(train_df, forget_indices):
    forget_df = train_df.iloc[forget_indices].copy().reset_index(drop=True)
    retain_df = train_df.drop(index=forget_indices).copy().reset_index(drop=True)
    return retain_df, forget_df


def strategy_selector(norm_influence, forget_indices, deletion_fraction):
    mean_inf = float(np.mean(norm_influence[forget_indices]))

    if deletion_fraction >= 0.50:
        return {
            "strategy": "full_retraining",
            "mean_forget_influence": mean_inf,
            "trainable_layers": BASELINE_TRAINABLE_BACKBONE_LAYERS,
            "epochs": BASELINE_MAX_EPOCHS,
            "learning_rate": BASELINE_LR
        }
    if mean_inf < LOW_INFLUENCE_THRESHOLD:
        return {
            "strategy": "light_selective_update",
            "mean_forget_influence": mean_inf,
            "trainable_layers": 10,
            "epochs": 3,
            "learning_rate": 1e-5
        }
    if mean_inf < HIGH_INFLUENCE_THRESHOLD:
        return {
            "strategy": "moderate_selective_update",
            "mean_forget_influence": mean_inf,
            "trainable_layers": 20,
            "epochs": 5,
            "learning_rate": 5e-6
        }
    return {
        "strategy": "strong_selective_update",
        "mean_forget_influence": mean_inf,
        "trainable_layers": 30,
        "epochs": 5,
        "learning_rate": 1e-6
    }



# ================================================================
# 7B. REVIEWER-STRENGTHENED EFFIAUL UPDATE
# ================================================================
#
# This implementation makes the components described in the manuscript
# operational:
#
#   1. Influence estimation:
#      mean influence of the requested forget set controls update strength.
#
#   2. Transfer refinement:
#      only a StrategySelector-selected tail of DenseNet is trainable.
#
#   3. Divergence control:
#      retained examples are constrained by teacher/student KL divergence.
#
#   4. Explicit forgetting pressure:
#      a small negative forget-set CE term reduces the original sample fit.
#
#   5. Utility gate:
#      candidate epochs are accepted only if validation F1 >= tau.
#
# The held-out TEST SET is never used to select hyperparameters or epochs.
# ================================================================

def effiaul_influence_adaptive_forget_weight(mean_forget_influence):
    """
    Increase forgetting pressure smoothly with the normalized mean influence
    of the requested samples while remaining close to the conservative base
    value. This rule is fixed before test evaluation.
    """
    m = float(np.clip(mean_forget_influence, 0.0, 1.0))
    return EFFIAUL_FORGET_ASCENT_BASE_WEIGHT * (0.5 + 1.5 * m)


def run_effiaul_update(
    student,
    teacher,
    retain_gen,
    forget_gen,
    val_gen,
    n_classes,
    tau,
    mean_forget_influence,
    learning_rate,
    max_epochs=EFFIAUL_MAX_EPOCHS
):
    """
    Influence-guided selective approximate unlearning.

    Objective per update:
        L = a * L_retain
            + b * KL(teacher_retain || student_retain)
            - c(I_F) * L_forget

    where c(I_F) increases with forget-set influence.

    The negative forget loss performs a controlled ascent step on the
    requested samples. Teacher distillation and validation gating prevent
    uncontrolled utility degradation.
    """

    teacher.trainable = False
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    forget_weight = effiaul_influence_adaptive_forget_weight(
        mean_forget_influence
    )

    best_weights = student.get_weights()
    best_forget_loss = -np.inf
    best_val_f1 = -np.inf
    best_epoch = 0
    epochs_without_improvement = 0

    history = {
        "accuracy": [],
        "val_accuracy": [],
        "loss": [],
        "val_loss": [],
        "val_f1_weighted": [],
        "forget_loss": [],
        "forget_weight": []
    }

    m0 = current_memory_mb()
    t0 = time.perf_counter()

    for epoch in range(int(max_epochs)):
        retain_gen.reset()
        forget_gen.reset()

        epoch_total = []
        correct = 0
        seen = 0

        if len(forget_gen) == 0:
            raise ValueError("Forget generator is empty.")

        for batch_idx in range(len(retain_gen)):
            xr, yr = retain_gen[batch_idx]
            xf, yf = forget_gen[batch_idx % len(forget_gen)]

            with tf.GradientTape() as tape:
                # Retained data: preserve clinical utility.
                student_r = student(xr, training=True)
                teacher_r = teacher(xr, training=False)

                retain_ce = task_loss(
                    yr, student_r, n_classes
                )

                retain_kd = kl_teacher_student(
                    teacher_r, student_r, n_classes, temperature=1.0
                )

                # Forget data: controlled ascent on original labels.
                student_f = student(xf, training=True)
                forget_ce = task_loss(
                    yf, student_f, n_classes
                )

                total_loss = (
                    EFFIAUL_RETAIN_CE_WEIGHT * retain_ce
                    + EFFIAUL_RETAIN_KD_WEIGHT * retain_kd
                    - forget_weight * forget_ce
                )

            grads = tape.gradient(
                total_loss, student.trainable_variables
            )
            pairs = [
                (g, v)
                for g, v in zip(grads, student.trainable_variables)
                if g is not None
            ]

            if pairs:
                clipped, _ = tf.clip_by_global_norm(
                    [g for g, _ in pairs],
                    EFFIAUL_GRAD_CLIP_NORM
                )
                optimizer.apply_gradients(
                    zip(clipped, [v for _, v in pairs])
                )

            epoch_total.append(float(total_loss.numpy()))

            # Approximate train accuracy for the history plot.
            if n_classes == 2:
                pred_label = (
                    np.asarray(student_r).reshape(-1) >= 0.5
                ).astype(np.int64)
                true_label = np.asarray(yr).reshape(-1).astype(np.int64)
            else:
                pred_label = np.argmax(np.asarray(student_r), axis=1)
                true_label = np.argmax(np.asarray(yr), axis=1)

            correct += int(np.sum(pred_label == true_label))
            seen += int(len(true_label))

        # Validation utility gate.
        val_metrics, _, _, _ = evaluate_model(
            student, val_gen, n_classes
        )

        val_gen.reset()
        val_eval = student.evaluate(
            val_gen, verbose=0, return_dict=True
        )

        # Forgetting behavior is used only for selecting the AuL update;
        # it is not a held-out predictive-performance test.
        forget_metrics = evaluate_forget_set(
            student, forget_gen, n_classes
        )

        train_acc = correct / max(seen, 1)
        val_f1 = float(val_metrics["f1_weighted"])
        f_loss = float(forget_metrics["forget_loss"])

        history["accuracy"].append(train_acc)
        history["val_accuracy"].append(float(val_metrics["accuracy"]))
        history["loss"].append(float(np.mean(epoch_total)))
        history["val_loss"].append(float(val_eval["loss"]))
        history["val_f1_weighted"].append(val_f1)
        history["forget_loss"].append(f_loss)
        history["forget_weight"].append(forget_weight)

        print(
            f"EffiAuL epoch {epoch+1}/{max_epochs}: "
            f"val_F1={val_f1:.5f}, "
            f"forget_loss={f_loss:.5f}, "
            f"forget_weight={forget_weight:.5f}"
        )

        # Among utility-valid candidates, prefer stronger forgetting.
        if val_f1 >= tau:
            improved = (
                (f_loss > best_forget_loss + 1e-8)
                or (
                    np.isclose(f_loss, best_forget_loss)
                    and val_f1 > best_val_f1
                )
            )

            if improved:
                best_weights = student.get_weights()
                best_forget_loss = f_loss
                best_val_f1 = val_f1
                best_epoch = epoch + 1
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
        else:
            epochs_without_improvement += 1

        if (
            epoch + 1 >= EFFIAUL_MIN_EPOCHS
            and epochs_without_improvement >= EFFIAUL_SELECTION_PATIENCE
            and best_epoch > 0
        ):
            print(
                "EffiAuL adaptive stopping: no better utility-valid "
                "forgetting checkpoint."
            )
            break

    # If at least one utility-valid state existed, restore it.
    if best_epoch > 0:
        student.set_weights(best_weights)
        selected_by = "utility_gate_plus_forget_loss"
    else:
        # Conservative behavior: retain the last state and let the outer
        # algorithm trigger the manuscript-specified fallback if needed.
        selected_by = "no_tau_satisfying_checkpoint"

    runtime = time.perf_counter() - t0
    memory_delta = current_memory_mb() - m0

    compile_model(student, n_classes, learning_rate)

    metadata = {
        "selected_epoch": int(best_epoch),
        "selection_rule": selected_by,
        "selected_validation_f1": (
            float(best_val_f1) if np.isfinite(best_val_f1) else np.nan
        ),
        "selected_forget_loss": (
            float(best_forget_loss) if np.isfinite(best_forget_loss) else np.nan
        ),
        "forget_weight": float(forget_weight)
    }

    return (
        student,
        SimpleNamespace(history=history),
        runtime,
        memory_delta,
        metadata
    )


# ================================================================
# 8. FORGETTING-SPECIFIC METRICS
# ================================================================

def evaluate_forget_set(model, forget_gen, n_classes):
    metrics, _, prob, _ = evaluate_model(model, forget_gen, n_classes)
    forget_gen.reset()
    ev = model.evaluate(forget_gen, verbose=0, return_dict=True)

    if n_classes == 2:
        confidence = np.maximum(prob, 1.0 - prob)
    else:
        confidence = np.max(prob, axis=1)

    return {
        "forget_loss": float(ev["loss"]),
        "forget_accuracy": metrics["accuracy"],
        "forget_f1_weighted": metrics["f1_weighted"],
        "forget_mean_confidence": float(np.mean(confidence))
    }


def prediction_change(reference_model, modified_model, generator, n_classes):
    _, p0, _ = get_predictions(reference_model, generator, n_classes)
    _, p1, _ = get_predictions(modified_model, generator, n_classes)

    if n_classes == 2:
        d = np.abs(p0 - p1)
    else:
        d = np.mean(np.abs(p0 - p1), axis=1)

    return float(np.mean(d)), float(np.max(d))


def prediction_distance(model_a, model_b, generator, n_classes):
    _, pa, _ = get_predictions(model_a, generator, n_classes)
    _, pb, _ = get_predictions(model_b, generator, n_classes)

    d = pa - pb
    return float(np.mean(np.abs(d))), float(np.sqrt(np.mean(d ** 2)))


def calculate_jsd(model_a, model_b, generator, n_classes):
    _, pa, _ = get_predictions(model_a, generator, n_classes)
    _, pb, _ = get_predictions(model_b, generator, n_classes)

    if n_classes == 2:
        pa = np.column_stack([1.0 - pa, pa])
        pb = np.column_stack([1.0 - pb, pb])

    eps = 1e-10
    values = []
    for a, b in zip(pa, pb):
        a = (a + eps) / np.sum(a + eps)
        b = (b + eps) / np.sum(b + eps)
        values.append(jensenshannon(a, b, base=2) ** 2)
    return float(np.mean(values))


# ================================================================
# 9. SIMPLE LOSS-BASED MEMBERSHIP INFERENCE ATTACK (MIA)
# ================================================================
# This is an empirical privacy diagnostic:
# lower per-sample loss => predict "member".
# Threshold is selected on:
#   members = a reference subset of RETAIN TRAINING data
#   nonmembers = VALIDATION data
# Then applied to the forget set.
# It is not LiRA and should not be described as LiRA.
# ================================================================

def per_sample_losses(model, generator, n_classes):
    generator.reset()
    losses = []
    labels_all = []

    for i in range(len(generator)):
        x, y = generator[i]
        pred = model(x, training=False)
        if n_classes == 2:
            y_tensor = tf.reshape(tf.cast(y, tf.float32), (-1, 1))
            batch_losses = tf.keras.losses.binary_crossentropy(y_tensor, pred)
        else:
            y_tensor = tf.cast(y, tf.float32)
            batch_losses = tf.keras.losses.categorical_crossentropy(y_tensor, pred)

        losses.extend(np.asarray(batch_losses).reshape(-1).tolist())
        labels_all.extend(np.asarray(y).tolist())

    return np.asarray(losses, dtype=float)


def choose_mia_threshold(member_losses, nonmember_losses):
    candidates = np.unique(np.concatenate([member_losses, nonmember_losses]))
    if len(candidates) > 2000:
        candidates = np.quantile(candidates, np.linspace(0, 1, 2000))

    best_t = float(np.median(candidates))
    best_acc = -1.0

    y_true = np.concatenate([
        np.ones(len(member_losses), dtype=int),
        np.zeros(len(nonmember_losses), dtype=int)
    ])
    scores = np.concatenate([member_losses, nonmember_losses])

    for t in candidates:
        y_pred = (scores <= t).astype(int)
        acc = accuracy_score(y_true, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_t = float(t)

    return best_t, float(best_acc)


def mia_forget_membership_rate(model, retain_eval_gen, val_gen, forget_gen, n_classes):
    retain_losses = per_sample_losses(model, retain_eval_gen, n_classes)
    val_losses = per_sample_losses(model, val_gen, n_classes)

    # Balance member/nonmember reference populations.
    m = min(len(retain_losses), len(val_losses))
    retain_ref = retain_losses[:m]
    val_ref = val_losses[:m]

    threshold, attack_ref_acc = choose_mia_threshold(retain_ref, val_ref)

    forget_losses = per_sample_losses(model, forget_gen, n_classes)
    forget_pred_member = (forget_losses <= threshold).astype(int)

    return {
        "mia_threshold": threshold,
        "mia_reference_accuracy": attack_ref_acc,
        "mia_forget_member_rate": float(np.mean(forget_pred_member))
    }


# ================================================================
# 10. SCRUB
# ================================================================
# Teacher/student formulation:
# retain min-step:
#   alpha * KL(teacher || student) + gamma * task_loss(student, y)
# forget max-step:
#   maximize KL(teacher || student)
# Alternates max and min steps, followed by additional retain min-steps.
# ================================================================

def probabilities_for_kl(output, n_classes):
    eps = 1e-7
    if n_classes == 2:
        p = tf.clip_by_value(output, eps, 1.0 - eps)
        return tf.concat([1.0 - p, p], axis=1)
    return tf.clip_by_value(output, eps, 1.0)


def kl_teacher_student(teacher_output, student_output, n_classes, temperature=1.0):
    # Outputs are already probabilities. Temperature is retained in the
    # API for reproducibility but defaults to 1.0 in this implementation.
    pt = probabilities_for_kl(teacher_output, n_classes)
    ps = probabilities_for_kl(student_output, n_classes)

    pt = pt / tf.reduce_sum(pt, axis=1, keepdims=True)
    ps = ps / tf.reduce_sum(ps, axis=1, keepdims=True)

    eps = 1e-7
    kl = tf.reduce_sum(
        pt * (tf.math.log(pt + eps) - tf.math.log(ps + eps)),
        axis=1
    )
    return tf.reduce_mean(kl)


def scrub_max_epoch(student, teacher, forget_gen, n_classes, optimizer):
    forget_gen.reset()
    losses = []

    for i in range(len(forget_gen)):
        x, _ = forget_gen[i]

        with tf.GradientTape() as tape:
            t = teacher(x, training=False)
            s = student(x, training=True)
            kl = kl_teacher_student(t, s, n_classes, SCRUB_TEMPERATURE)
            objective = -kl  # minimize negative KL -> maximize disagreement

        grads = tape.gradient(objective, student.trainable_variables)
        pairs = [(g, v) for g, v in zip(grads, student.trainable_variables) if g is not None]
        optimizer.apply_gradients(pairs)
        losses.append(float(kl.numpy()))

    return float(np.mean(losses)) if losses else np.nan


def scrub_min_epoch(student, teacher, retain_gen, n_classes, optimizer):
    retain_gen.reset()
    values = []

    for i in range(len(retain_gen)):
        x, y = retain_gen[i]

        with tf.GradientTape() as tape:
            t = teacher(x, training=False)
            s = student(x, training=True)
            kl = kl_teacher_student(t, s, n_classes, SCRUB_TEMPERATURE)
            ce = task_loss(y, s, n_classes)
            objective = SCRUB_ALPHA * kl + SCRUB_GAMMA * ce

        grads = tape.gradient(objective, student.trainable_variables)
        pairs = [(g, v) for g, v in zip(grads, student.trainable_variables) if g is not None]
        optimizer.apply_gradients(pairs)
        values.append(float(objective.numpy()))

    return float(np.mean(values)) if values else np.nan


def run_scrub(student, teacher, retain_gen, forget_gen, n_classes):
    teacher.trainable = False

    max_opt = tf.keras.optimizers.Adam(SCRUB_MAX_LR)
    min_opt = tf.keras.optimizers.Adam(SCRUB_MIN_LR)

    t0 = time.perf_counter()
    m0 = current_memory_mb()

    for epoch in range(SCRUB_ALTERNATING_EPOCHS):
        max_kl = scrub_max_epoch(student, teacher, forget_gen, n_classes, max_opt)
        min_loss = scrub_min_epoch(student, teacher, retain_gen, n_classes, min_opt)
        print(
            f"SCRUB alternating epoch {epoch+1}/{SCRUB_ALTERNATING_EPOCHS}: "
            f"forget_KL={max_kl:.5f}, retain_objective={min_loss:.5f}"
        )

    for epoch in range(SCRUB_FINAL_MIN_EPOCHS):
        min_loss = scrub_min_epoch(student, teacher, retain_gen, n_classes, min_opt)
        print(
            f"SCRUB final min epoch {epoch+1}/{SCRUB_FINAL_MIN_EPOCHS}: "
            f"retain_objective={min_loss:.5f}"
        )

    runtime = time.perf_counter() - t0
    memory_delta = current_memory_mb() - m0
    compile_model(student, n_classes, SCRUB_MIN_LR)
    return student, runtime, memory_delta


# ================================================================
# 11. IF-DIAG: INFLUENCE-FUNCTION-FAMILY APPROXIMATION
# ================================================================
# Approximate removal update:
# theta_minus ~= theta + scale/N * H_diag^{-1} * grad L_forget
#
# H_diag is approximated with empirical Fisher E[g^2] + damping.
# For tractability, default updates classifier-head parameters only.
# ================================================================

def classifier_head_variables(model):
    variables = []
    for layer_name in ["classification_dense", "prediction"]:
        layer = model.get_layer(layer_name)
        variables.extend(layer.trainable_variables)
    return variables


def if_diag_unlearning(model, retain_gen, forget_gen, n_classes):
    vars_to_update = classifier_head_variables(model) if IF_HEAD_ONLY else model.trainable_variables

    fisher = [tf.zeros_like(v) for v in vars_to_update]
    n_batches = 0

    # Estimate diagonal Fisher on retained data.
    retain_gen.reset()
    for i in range(len(retain_gen)):
        x, y = retain_gen[i]
        with tf.GradientTape() as tape:
            pred = model(x, training=False)
            loss = task_loss(y, pred, n_classes)
        grads = tape.gradient(loss, vars_to_update)
        fisher = [
            f + (tf.square(g) if g is not None else tf.zeros_like(v))
            for f, g, v in zip(fisher, grads, vars_to_update)
        ]
        n_batches += 1

    fisher = [
        f / max(n_batches, 1) + IF_DIAG_DAMPING
        for f in fisher
    ]

    # Mean forget gradient.
    forget_grads = [tf.zeros_like(v) for v in vars_to_update]
    n_forget_batches = 0

    forget_gen.reset()
    for i in range(len(forget_gen)):
        x, y = forget_gen[i]
        with tf.GradientTape() as tape:
            pred = model(x, training=False)
            loss = task_loss(y, pred, n_classes)
        grads = tape.gradient(loss, vars_to_update)
        forget_grads = [
            acc + (g if g is not None else tf.zeros_like(v))
            for acc, g, v in zip(forget_grads, grads, vars_to_update)
        ]
        n_forget_batches += 1

    forget_grads = [g / max(n_forget_batches, 1) for g in forget_grads]

    # Approximate parameter change for deletion.
    retain_n = max(retain_gen.samples, 1)
    for v, g, hdiag in zip(vars_to_update, forget_grads, fisher):
        delta = IF_DIAG_SCALE * g / (hdiag * retain_n)
        v.assign_add(delta)

    compile_model(model, n_classes, BASELINE_LR)
    return model


def timed_if_diag_unlearning(model, retain_gen, forget_gen, n_classes):
    m0 = current_memory_mb()
    t0 = time.perf_counter()
    model = if_diag_unlearning(model, retain_gen, forget_gen, n_classes)
    return model, time.perf_counter() - t0, current_memory_mb() - m0


# ================================================================
# 12. SISA
# ================================================================

def split_shards_slices(train_df, seed):
    # Stratified shards when possible.
    df = train_df.copy().reset_index(drop=True)
    rng = np.random.default_rng(seed)

    shard_rows = [[] for _ in range(SISA_NUM_SHARDS)]

    # Round-robin class-wise assignment preserves class balance reasonably.
    for label, group in df.groupby("label"):
        indices = group.index.to_numpy()
        rng.shuffle(indices)
        for k, idx in enumerate(indices):
            shard_rows[k % SISA_NUM_SHARDS].append(idx)

    shards = []
    for shard_id, rows in enumerate(shard_rows):
        shard_df = df.loc[sorted(rows)].copy().reset_index(drop=True)
        shard_df["sisa_shard"] = shard_id

        # Randomly slice within shard.
        order = np.arange(len(shard_df))
        rng.shuffle(order)
        slice_ids = np.zeros(len(shard_df), dtype=int)
        chunks = np.array_split(order, SISA_NUM_SLICES)
        for slice_id, chunk in enumerate(chunks):
            slice_ids[chunk] = slice_id
        shard_df["sisa_slice"] = slice_ids
        shards.append(shard_df)

    return shards


def train_sisa(train_df, val_df, class_names, class_mode, n_classes, seed, output_dir):
    shards = split_shards_slices(train_df, seed)
    models = []
    metadata = []
    total_runtime = 0.0
    total_memory = 0.0

    ensure_dir(output_dir)

    for shard_id, shard_df in enumerate(shards):
        print(f"\nSISA training shard {shard_id+1}/{len(shards)}")

        set_seed(seed + shard_id)
        model = build_densenet_model(
            n_classes,
            lr=SISA_LR,
            trainable_backbone_layers=SISA_TRAINABLE_LAYERS
        )

        shard_dir = os.path.join(output_dir, f"shard_{shard_id}")
        ensure_dir(shard_dir)

        # Save the initial state (before slice 0).
        initial_path = os.path.join(shard_dir, "before_slice_0.weights.h5")
        model.save_weights(initial_path)

        cumulative = pd.DataFrame(columns=shard_df.columns)

        for slice_id in range(SISA_NUM_SLICES):
            slice_df = shard_df[shard_df["sisa_slice"] == slice_id].copy()
            cumulative = pd.concat([cumulative, slice_df], ignore_index=True)

            train_gen = dataframe_generator(
                cumulative, class_names, class_mode,
                train=True, shuffle=True, seed=seed + 100*shard_id + slice_id
            )
            val_gen = dataframe_generator(
                val_df, class_names, class_mode,
                train=False, shuffle=False, seed=seed
            )

            m0 = current_memory_mb()
            t0 = time.perf_counter()
            model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=SISA_EPOCHS_PER_SLICE,
                verbose=0
            )
            total_runtime += time.perf_counter() - t0
            total_memory += max(0.0, current_memory_mb() - m0)

            # Save state AFTER current slice, which becomes "before next slice".
            checkpoint = os.path.join(
                shard_dir, f"after_slice_{slice_id}.weights.h5"
            )
            model.save_weights(checkpoint)

        models.append(model)
        metadata.append(shard_df)

    return models, metadata, total_runtime, total_memory


def sisa_predict(models, generator, n_classes):
    assert_evaluation_generator(generator, name="SISA evaluation generator")
    if not models:
        raise ValueError("SISA model list is empty.")

    preds = []
    for m in models:
        generator.reset()
        p = np.asarray(m.predict(generator, verbose=0))
        if n_classes == 2:
            p = p.reshape(-1)
        preds.append(p)

    # All shard models must predict the same examples in exactly the same order.
    expected_shape = preds[0].shape
    if any(p.shape != expected_shape for p in preds):
        raise ValueError(
            "SISA shard prediction shapes differ. Ensure the same evaluation "
            "generator (shuffle=False) is used for every shard."
        )

    avg = np.mean(np.stack(preds, axis=0), axis=0)
    y_true = get_generator_classes(generator)

    if n_classes == 2:
        y_pred = (avg >= 0.5).astype(np.int64)
    else:
        if avg.ndim != 2 or avg.shape[1] != n_classes:
            raise ValueError(
                f"Expected SISA multiclass probabilities with shape (N, {n_classes}), "
                f"but received {avg.shape}."
            )
        y_pred = np.argmax(avg, axis=1).astype(np.int64)

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"SISA prediction/label length mismatch: {len(y_pred)} predictions vs "
            f"{len(y_true)} labels."
        )

    return y_true, avg, y_pred


def sisa_unlearn(
    sisa_models, shard_metadata, forget_df, val_df,
    class_names, class_mode, n_classes, seed, output_dir
):
    # Identify each requested filepath's shard/slice.
    location = {}
    for shard_id, shard_df in enumerate(shard_metadata):
        for _, row in shard_df.iterrows():
            location[row["filepath"]] = (shard_id, int(row["sisa_slice"]))

    affected = {}
    for path in forget_df["filepath"]:
        if path not in location:
            continue
        shard_id, slice_id = location[path]
        affected.setdefault(shard_id, []).append(slice_id)

    m0 = current_memory_mb()
    t0 = time.perf_counter()

    new_models = list(sisa_models)

    for shard_id, slice_ids in affected.items():
        earliest = min(slice_ids)
        shard_df = shard_metadata[shard_id].copy()
        retained_shard = shard_df[
            ~shard_df["filepath"].isin(set(forget_df["filepath"]))
        ].copy()

        set_seed(seed + shard_id)
        model = build_densenet_model(
            n_classes,
            lr=SISA_LR,
            trainable_backbone_layers=SISA_TRAINABLE_LAYERS
        )

        shard_dir = os.path.join(output_dir, f"shard_{shard_id}")

        if earliest == 0:
            checkpoint = os.path.join(shard_dir, "before_slice_0.weights.h5")
        else:
            checkpoint = os.path.join(
                shard_dir, f"after_slice_{earliest-1}.weights.h5"
            )

        model.load_weights(checkpoint)

        # Rebuild cumulative state through affected and subsequent slices.
        if earliest == 0:
            cumulative = pd.DataFrame(columns=retained_shard.columns)
        else:
            cumulative = retained_shard[
                retained_shard["sisa_slice"] < earliest
            ].copy()

        for slice_id in range(earliest, SISA_NUM_SLICES):
            current = retained_shard[
                retained_shard["sisa_slice"] == slice_id
            ].copy()
            cumulative = pd.concat([cumulative, current], ignore_index=True)

            if cumulative.empty:
                continue

            train_gen = dataframe_generator(
                cumulative, class_names, class_mode,
                train=True, shuffle=True,
                seed=seed + 100*shard_id + slice_id
            )
            val_gen = dataframe_generator(
                val_df, class_names, class_mode,
                train=False, shuffle=False, seed=seed
            )

            model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=SISA_EPOCHS_PER_SLICE,
                verbose=0
            )

        new_models[shard_id] = model

    runtime = time.perf_counter() - t0
    memory_delta = current_memory_mb() - m0
    return new_models, runtime, memory_delta


# ================================================================
# 13. PLOTTING
# ================================================================

def save_training_curves(history, prefix):
    plt.figure(figsize=(8, 6))
    plt.plot(history.history.get("accuracy", []), label="Training")
    plt.plot(history.history.get("val_accuracy", []), label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(prefix + "_accuracy.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(history.history.get("loss", []), label="Training")
    plt.plot(history.history.get("val_loss", []), label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(prefix + "_loss.png", dpi=300)
    plt.close()


def save_confusion_matrix(y_true, y_pred, class_names, path, title):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


# ================================================================
# 14. GENERIC RESULT PACKAGING
# ================================================================

def evaluate_standard_method(
    name, model, baseline_model, retrain_model,
    test_gen, retain_eval_gen, val_gen, forget_gen,
    n_classes, seed, runtime, memory_delta, tau, strategy
):
    metrics, y_true, prob, y_pred = evaluate_model(model, test_gen, n_classes)
    ci_low, ci_high = bootstrap_auc_ci(y_true, prob, n_classes, seed)
    forget_metrics = evaluate_forget_set(model, forget_gen, n_classes)

    if name == "Baseline":
        chg_mean, chg_max = 0.0, 0.0
        jsd = 0.0
    else:
        chg_mean, chg_max = prediction_change(
            baseline_model, model, forget_gen, n_classes
        )
        jsd = calculate_jsd(baseline_model, model, forget_gen, n_classes)

    if name == "Full_Retraining":
        dist_mae, dist_rmse = 0.0, 0.0
        forget_dist_mae, forget_dist_rmse = 0.0, 0.0
        retain_dist_mae, retain_dist_rmse = 0.0, 0.0
    else:
        dist_mae, dist_rmse = prediction_distance(
            model, retrain_model, test_gen, n_classes
        )
        # More directly relevant to machine unlearning than test-set distance:
        # how closely does the method behave like exact retraining on the
        # requested forget samples and on retained training samples?
        forget_dist_mae, forget_dist_rmse = prediction_distance(
            model, retrain_model, forget_gen, n_classes
        )
        retain_dist_mae, retain_dist_rmse = prediction_distance(
            model, retrain_model, retain_eval_gen, n_classes
        )

    mia = mia_forget_membership_rate(
        model, retain_eval_gen, val_gen, forget_gen, n_classes
    )

    row = {
        "seed": seed,
        "method": name,
        "test_accuracy": metrics["accuracy"],
        "test_precision_weighted": metrics["precision_weighted"],
        "test_recall_weighted": metrics["recall_weighted"],
        "test_f1_weighted": metrics["f1_weighted"],
        "test_auc": metrics["auc"],
        "auc_95ci_lower": ci_low,
        "auc_95ci_upper": ci_high,
        **forget_metrics,
        "forget_mean_probability_change": chg_mean,
        "forget_max_probability_change": chg_max,
        "distance_to_retrain_mae": dist_mae,
        "distance_to_retrain_rmse": dist_rmse,
        "forget_distance_to_retrain_mae": forget_dist_mae,
        "forget_distance_to_retrain_rmse": forget_dist_rmse,
        "retain_distance_to_retrain_mae": retain_dist_mae,
        "retain_distance_to_retrain_rmse": retain_dist_rmse,
        "forget_set_jsd": jsd,
        **mia,
        "runtime_seconds": runtime,
        "memory_change_mb": memory_delta,
        "tau": tau,
        "meets_utility_threshold": metrics["f1_weighted"] >= tau,
        "strategy": strategy.get("strategy", "not_applicable"),
        "mean_forget_influence": strategy.get("mean_forget_influence", np.nan)
    }

    return row, y_true, y_pred


def evaluate_sisa_method(
    sisa_models, baseline_model, retrain_model,
    test_gen, retain_eval_gen, val_gen, forget_gen,
    n_classes, seed, runtime, memory_delta, tau
):
    y_true, prob, y_pred = sisa_predict(sisa_models, test_gen, n_classes)
    metrics = metrics_from_predictions(y_true, prob, y_pred, n_classes)
    ci_low, ci_high = bootstrap_auc_ci(y_true, prob, n_classes, seed)

    # SISA ensemble forget predictions/metrics.
    yf, pf, pred_f = sisa_predict(sisa_models, forget_gen, n_classes)
    forget_metrics_class = metrics_from_predictions(yf, pf, pred_f, n_classes)

    # Cross-entropy on ensemble predictions.
    eps = 1e-7
    if n_classes == 2:
        pf_clip = np.clip(pf, eps, 1-eps)
        forget_loss = -np.mean(
            yf*np.log(pf_clip) + (1-yf)*np.log(1-pf_clip)
        )
        forget_conf = np.mean(np.maximum(pf, 1-pf))
    else:
        pf_clip = np.clip(pf, eps, 1.0)
        forget_loss = -np.mean(np.log(pf_clip[np.arange(len(yf)), yf]))
        forget_conf = np.mean(np.max(pf, axis=1))

    # Prediction change vs single baseline.
    _, p0f, _ = get_predictions(baseline_model, forget_gen, n_classes)
    if n_classes == 2:
        d = np.abs(p0f - pf)
        pa_js = np.column_stack([1-p0f, p0f])
        pb_js = np.column_stack([1-pf, pf])
    else:
        d = np.mean(np.abs(p0f - pf), axis=1)
        pa_js, pb_js = p0f, pf

    js_values = []
    for a, b in zip(pa_js, pb_js):
        a = (a + eps) / np.sum(a + eps)
        b = (b + eps) / np.sum(b + eps)
        js_values.append(jensenshannon(a, b, base=2)**2)

    # Distance to retraining on test / forget / retain data.
    _, pret, _ = get_predictions(retrain_model, test_gen, n_classes)
    diff = prob - pret

    _, pf_retrain, _ = get_predictions(retrain_model, forget_gen, n_classes)
    forget_diff = pf - pf_retrain

    yr_sisa, pr_sisa, _ = sisa_predict(
        sisa_models, retain_eval_gen, n_classes
    )
    _, pr_retrain, _ = get_predictions(
        retrain_model, retain_eval_gen, n_classes
    )
    retain_diff = pr_sisa - pr_retrain

    # MIA for SISA ensemble using explicit loss arrays.
    yr, pr, _ = sisa_predict(sisa_models, retain_eval_gen, n_classes)
    yv, pv, _ = sisa_predict(sisa_models, val_gen, n_classes)

    def losses_np(y, p):
        if n_classes == 2:
            p = np.clip(p, eps, 1-eps)
            return -(y*np.log(p) + (1-y)*np.log(1-p))
        p = np.clip(p, eps, 1.0)
        return -np.log(p[np.arange(len(y)), y])

    rloss = losses_np(yr, pr)
    vloss = losses_np(yv, pv)
    m = min(len(rloss), len(vloss))
    threshold, ref_acc = choose_mia_threshold(rloss[:m], vloss[:m])
    fl = losses_np(yf, pf)
    mia_rate = np.mean(fl <= threshold)

    row = {
        "seed": seed,
        "method": "SISA",
        "test_accuracy": metrics["accuracy"],
        "test_precision_weighted": metrics["precision_weighted"],
        "test_recall_weighted": metrics["recall_weighted"],
        "test_f1_weighted": metrics["f1_weighted"],
        "test_auc": metrics["auc"],
        "auc_95ci_lower": ci_low,
        "auc_95ci_upper": ci_high,
        "forget_loss": float(forget_loss),
        "forget_accuracy": forget_metrics_class["accuracy"],
        "forget_f1_weighted": forget_metrics_class["f1_weighted"],
        "forget_mean_confidence": float(forget_conf),
        "forget_mean_probability_change": float(np.mean(d)),
        "forget_max_probability_change": float(np.max(d)),
        "distance_to_retrain_mae": float(np.mean(np.abs(diff))),
        "distance_to_retrain_rmse": float(np.sqrt(np.mean(diff**2))),
        "forget_distance_to_retrain_mae": float(np.mean(np.abs(forget_diff))),
        "forget_distance_to_retrain_rmse": float(np.sqrt(np.mean(forget_diff**2))),
        "retain_distance_to_retrain_mae": float(np.mean(np.abs(retain_diff))),
        "retain_distance_to_retrain_rmse": float(np.sqrt(np.mean(retain_diff**2))),
        "forget_set_jsd": float(np.mean(js_values)),
        "mia_threshold": float(threshold),
        "mia_reference_accuracy": float(ref_acc),
        "mia_forget_member_rate": float(mia_rate),
        "runtime_seconds": runtime,
        "memory_change_mb": memory_delta,
        "tau": tau,
        "meets_utility_threshold": metrics["f1_weighted"] >= tau,
        "strategy": "SISA",
        "mean_forget_influence": np.nan
    }
    return row, y_true, y_pred


# ================================================================
# 15. ONE EXPERIMENTAL SEED
# ================================================================

def run_experiment(
    seed, train_df, val_df, test_df,
    class_names, n_classes, class_mode
):
    print("\n" + "="*90)
    print(f"EXPERIMENT SEED {seed}")
    print("="*90)

    seed_dir = os.path.join(RESULTS_DIR, f"Seed_{seed}")
    ensure_dir(seed_dir)
    set_seed(seed)

    train_gen, influence_gen, val_gen, test_gen = create_main_generators(
        train_df, val_df, test_df, class_names, class_mode, seed
    )

    # ------------------------------------------------------------
    # A. BASELINE
    # ------------------------------------------------------------
    print("\n[A] BASELINE")
    baseline = build_densenet_model(n_classes)
    baseline_history, baseline_runtime, baseline_memory = timed_fit(
        baseline, train_gen, val_gen, BASELINE_MAX_EPOCHS
    )
    baseline_weights = os.path.join(seed_dir, "baseline.weights.h5")
    baseline.save_weights(baseline_weights)
    save_training_curves(baseline_history, os.path.join(seed_dir, "baseline"))

    baseline_val_metrics, _, _, _ = evaluate_model(baseline, val_gen, n_classes)
    tau = baseline_val_metrics["f1_weighted"] - UTILITY_TOLERANCE
    print(f"Baseline validation weighted-F1={baseline_val_metrics['f1_weighted']:.5f}")
    print(f"Utility threshold tau={tau:.5f}")

    # ------------------------------------------------------------
    # B. INFLUENCE + FORGET SET
    # ------------------------------------------------------------
    print("\n[B] INFLUENCE SCORING / FORGET SET")
    influence_df, norm_inf = calculate_influence_scores(
        baseline, influence_gen, train_df, n_classes
    )
    influence_df.to_csv(os.path.join(seed_dir, "influence_scores.csv"), index=False)

    forget_idx = select_forget_indices(norm_inf, N_FORGET, FORGET_SELECTION, FORGET_SET_SEED)
    np.save(os.path.join(seed_dir, "forget_indices.npy"), forget_idx)

    selected = influence_df.iloc[forget_idx].copy()
    selected["forget_strategy"] = FORGET_SELECTION
    selected.to_csv(os.path.join(seed_dir, "forgotten_samples.csv"), index=False)

    retain_df, forget_df = create_retain_forget_dataframes(train_df, forget_idx)
    retain_df.to_csv(os.path.join(seed_dir, "retain_set.csv"), index=False)
    forget_df.to_csv(os.path.join(seed_dir, "forget_set.csv"), index=False)

    retain_train_gen = dataframe_generator(
        retain_df, class_names, class_mode, train=True, shuffle=True, seed=seed
    )
    retain_eval_gen = dataframe_generator(
        retain_df, class_names, class_mode, train=False, shuffle=False, seed=seed
    )
    forget_gen = dataframe_generator(
        forget_df, class_names, class_mode, train=False, shuffle=False, seed=seed
    )

    strategy = strategy_selector(
        norm_inf, forget_idx, len(forget_df)/len(train_df)
    )
    with open(os.path.join(seed_dir, "strategy.json"), "w") as f:
        json.dump(strategy, f, indent=2)
    print(json.dumps(strategy, indent=2))

    # ------------------------------------------------------------
    # C. FULL RETRAINING REFERENCE
    # ------------------------------------------------------------
    print("\n[C] FULL RETRAINING REFERENCE")
    clear_memory()
    set_seed(seed)
    retrain = build_densenet_model(n_classes)
    retrain_history, retrain_runtime, retrain_memory = timed_fit(
        retrain, retain_train_gen, val_gen, BASELINE_MAX_EPOCHS
    )
    save_training_curves(retrain_history, os.path.join(seed_dir, "full_retraining"))

    # ------------------------------------------------------------
    # D. EFFIAUL
    # ------------------------------------------------------------
    print("\n[D] EFFIAUL - INFLUENCE-GUIDED SELECTIVE UPDATE")
    clear_memory()
    set_seed(seed)

    # Frozen baseline teacher for divergence control on retained data.
    eff_teacher = build_densenet_model(n_classes)
    eff_teacher.load_weights(baseline_weights)
    eff_teacher.trainable = False

    # Student begins from exactly the same baseline weights.
    effiaul = build_densenet_model(n_classes)
    effiaul.load_weights(baseline_weights)
    effiaul = configure_trainable_layers(
        effiaul,
        n_classes,
        strategy["trainable_layers"],
        strategy["learning_rate"]
    )

    # Use at least the StrategySelector duration, but permit the adaptive
    # update to continue long enough to find a utility-valid forgetting state.
    eff_max_epochs = max(
        int(strategy["epochs"]),
        int(EFFIAUL_MAX_EPOCHS)
    )

    (
        effiaul,
        eff_history,
        eff_runtime,
        eff_memory,
        eff_metadata
    ) = run_effiaul_update(
        effiaul,
        eff_teacher,
        retain_train_gen,
        forget_gen,
        val_gen,
        n_classes,
        tau,
        strategy["mean_forget_influence"],
        strategy["learning_rate"],
        max_epochs=eff_max_epochs
    )

    save_training_curves(
        eff_history, os.path.join(seed_dir, "effiaul")
    )

    with open(
        os.path.join(seed_dir, "effiaul_update_metadata.json"), "w"
    ) as f:
        json.dump(eff_metadata, f, indent=2)

    # Manuscript-consistent utility fallback:
    # if the final selected approximate-unlearning model violates tau,
    # use the already-computed full-retraining reference.
    eff_val_metrics, _, _, _ = evaluate_model(
        effiaul, val_gen, n_classes
    )

    eff_fallback_used = False
    if eff_val_metrics["f1_weighted"] < tau:
        print(
            "EffiAuL validation utility fell below tau; "
            "activating full-retraining fallback."
        )
        effiaul = retrain
        eff_runtime += retrain_runtime
        eff_memory = max(eff_memory, retrain_memory)
        eff_fallback_used = True

    eff_metadata["fallback_used"] = bool(eff_fallback_used)
    with open(
        os.path.join(seed_dir, "effiaul_update_metadata.json"), "w"
    ) as f:
        json.dump(eff_metadata, f, indent=2)

    # ------------------------------------------------------------
    # E. NO-DELETION FINE-TUNING CONTROL
    # ------------------------------------------------------------
    print("\n[E] FINE-TUNING CONTROL (NO DELETION)")
    clear_memory()
    set_seed(seed)
    control = build_densenet_model(n_classes)
    control.load_weights(baseline_weights)
    control = configure_trainable_layers(
        control, n_classes,
        strategy["trainable_layers"], strategy["learning_rate"]
    )
    control_history, control_runtime, control_memory = timed_fit(
        control, train_gen, val_gen, strategy["epochs"]
    )
    save_training_curves(control_history, os.path.join(seed_dir, "finetune_control"))

    # ------------------------------------------------------------
    # F. SCRUB
    # ------------------------------------------------------------
    print("\n[F] SCRUB")
    clear_memory()
    set_seed(seed)
    teacher = build_densenet_model(n_classes)
    teacher.load_weights(baseline_weights)
    teacher.trainable = False

    scrub = build_densenet_model(n_classes)
    scrub.load_weights(baseline_weights)
    scrub = configure_trainable_layers(
        scrub, n_classes, strategy["trainable_layers"], SCRUB_MIN_LR
    )
    scrub, scrub_runtime, scrub_memory = run_scrub(
        scrub, teacher, retain_train_gen, forget_gen, n_classes
    )

    # ------------------------------------------------------------
    # G. IF-DIAG
    # ------------------------------------------------------------
    print("\n[G] IF-DIAG INFLUENCE-FUNCTION-FAMILY BASELINE")
    clear_memory()
    set_seed(seed)
    if_model = build_densenet_model(n_classes)
    if_model.load_weights(baseline_weights)
    if_model, if_runtime, if_memory = timed_if_diag_unlearning(
        if_model, retain_eval_gen, forget_gen, n_classes
    )

    # ------------------------------------------------------------
    # H. SISA
    # ------------------------------------------------------------
    print("\n[H] SISA")
    clear_memory()
    sisa_dir = os.path.join(seed_dir, "sisa")
    sisa_models, sisa_meta, sisa_train_runtime, sisa_train_memory = train_sisa(
        train_df, val_df, class_names, class_mode,
        n_classes, seed, sisa_dir
    )
    sisa_models, sisa_unlearn_runtime, sisa_unlearn_memory = sisa_unlearn(
        sisa_models, sisa_meta, forget_df, val_df,
        class_names, class_mode, n_classes, seed, sisa_dir
    )
    # Report unlearning update cost (not original SISA preparation cost) in main runtime.
    sisa_runtime = sisa_unlearn_runtime
    sisa_memory = sisa_unlearn_memory

    # Save preparation cost separately.
    pd.DataFrame([{
        "seed": seed,
        "sisa_initial_training_seconds": sisa_train_runtime,
        "sisa_unlearning_seconds": sisa_unlearn_runtime,
        "sisa_initial_memory_change_mb": sisa_train_memory,
        "sisa_unlearning_memory_change_mb": sisa_unlearn_memory
    }]).to_csv(os.path.join(seed_dir, "sisa_costs.csv"), index=False)

    # ------------------------------------------------------------
    # I. EVALUATE
    # ------------------------------------------------------------
    print("\n[I] FINAL HELD-OUT TEST + FORGETTING EVALUATION")
    rows = []

    standard_methods = [
        ("Baseline", baseline, baseline_runtime, baseline_memory, {"strategy": "baseline"}),
        ("FineTune_Control", control, control_runtime, control_memory, {"strategy": "no_deletion_control"}),
        ("EffiAuL", effiaul, eff_runtime, eff_memory, strategy),
        ("SCRUB", scrub, scrub_runtime, scrub_memory, {"strategy": "SCRUB"}),
        ("IF_Diag", if_model, if_runtime, if_memory, {"strategy": "IF_Diag"}),
        ("Full_Retraining", retrain, retrain_runtime, retrain_memory, {"strategy": "full_retraining"}),
    ]

    for name, model, runtime, memory, strategy_info in standard_methods:
        print(f"Evaluating {name}")
        row, yt, yp = evaluate_standard_method(
            name, model, baseline, retrain,
            test_gen, retain_eval_gen, val_gen, forget_gen,
            n_classes, seed, runtime, memory, tau, strategy_info
        )
        rows.append(row)
        save_confusion_matrix(
            yt, yp, class_names,
            os.path.join(seed_dir, f"{name}_test_confusion_matrix.png"),
            f"{name} - Held-out Test Set"
        )

    print("Evaluating SISA")
    sisa_row, yt, yp = evaluate_sisa_method(
        sisa_models, baseline, retrain,
        test_gen, retain_eval_gen, val_gen, forget_gen,
        n_classes, seed, sisa_runtime, sisa_memory, tau
    )
    rows.append(sisa_row)
    save_confusion_matrix(
        yt, yp, class_names,
        os.path.join(seed_dir, "SISA_test_confusion_matrix.png"),
        "SISA - Held-out Test Set"
    )

    # Reference-normalized reviewer-facing quantities.
    retrain_row = next(
        row for row in rows if row["method"] == "Full_Retraining"
    )
    baseline_row = next(
        row for row in rows if row["method"] == "Baseline"
    )

    for row in rows:
        row["utility_f1_gap_vs_full_retraining"] = float(
            row["test_f1_weighted"] - retrain_row["test_f1_weighted"]
        )
        row["mia_gap_to_full_retraining"] = float(
            abs(
                row["mia_forget_member_rate"]
                - retrain_row["mia_forget_member_rate"]
            )
        )
        row["forget_distance_gain_vs_baseline"] = float(
            baseline_row["forget_distance_to_retrain_mae"]
            - row["forget_distance_to_retrain_mae"]
        )

    # Computational savings specifically versus gold-standard retraining.
    for row in rows:
        if retrain_runtime > 0 and row["method"] not in ["Baseline", "Full_Retraining"]:
            row["time_reduction_vs_full_retraining_percent"] = float(
                100.0 * (retrain_runtime - row["runtime_seconds"]) / retrain_runtime
            )
        else:
            row["time_reduction_vs_full_retraining_percent"] = np.nan

    pd.DataFrame(rows).to_csv(
        os.path.join(seed_dir, "all_method_results.csv"), index=False
    )
    return rows


# ================================================================
# 16. AGGREGATION / SIGNIFICANCE
# ================================================================

def aggregate_results(df):
    metrics = [
        "test_accuracy",
        "test_precision_weighted",
        "test_recall_weighted",
        "test_f1_weighted",
        "test_auc",
        "forget_loss",
        "forget_accuracy",
        "forget_f1_weighted",
        "forget_mean_confidence",
        "forget_mean_probability_change",
        "distance_to_retrain_mae",
        "distance_to_retrain_rmse",
        "forget_distance_to_retrain_mae",
        "forget_distance_to_retrain_rmse",
        "retain_distance_to_retrain_mae",
        "retain_distance_to_retrain_rmse",
        "utility_f1_gap_vs_full_retraining",
        "mia_gap_to_full_retraining",
        "forget_distance_gain_vs_baseline",
        "forget_set_jsd",
        "mia_reference_accuracy",
        "mia_forget_member_rate",
        "runtime_seconds",
        "memory_change_mb",
        "time_reduction_vs_full_retraining_percent"
    ]

    rows = []
    for method, g in df.groupby("method"):
        row = {"method": method, "number_of_runs": len(g)}
        for metric in metrics:
            if metric not in g:
                continue
            values = pd.to_numeric(g[metric], errors="coerce").dropna()
            if len(values) == 0:
                continue

            mean = float(values.mean())
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            # Normal-approximation CI across runs.
            half = 1.96 * std / np.sqrt(len(values)) if len(values) > 1 else 0.0

            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_95ci_lower"] = mean - half
            row[f"{metric}_95ci_upper"] = mean + half
        rows.append(row)

    return pd.DataFrame(rows)


def paired_tests(df, metric):
    pivot = df.pivot(index="seed", columns="method", values=metric)
    if "EffiAuL" not in pivot.columns:
        return pd.DataFrame()

    comparators = [
        c for c in [
            "Baseline", "FineTune_Control", "SCRUB",
            "IF_Diag", "SISA", "Full_Retraining"
        ]
        if c in pivot.columns
    ]

    rows = []
    for comp in comparators:
        paired = pivot[[comp, "EffiAuL"]].dropna()
        if len(paired) < 2:
            continue

        a = paired[comp].values
        b = paired["EffiAuL"].values

        try:
            ws, wp = wilcoxon(a, b)
        except Exception:
            ws, wp = np.nan, np.nan

        try:
            ts, tp = ttest_rel(a, b)
        except Exception:
            ts, tp = np.nan, np.nan

        rows.append({
            "metric": metric,
            "comparator": comp,
            "method": "EffiAuL",
            "n_pairs": len(a),
            "mean_comparator": float(np.mean(a)),
            "mean_effiaul": float(np.mean(b)),
            "mean_difference_effiaul_minus_comparator": float(np.mean(b-a)),
            "wilcoxon_statistic": ws,
            "wilcoxon_p_value": wp,
            "paired_t_statistic": ts,
            "paired_t_p_value": tp
        })

    return pd.DataFrame(rows)



# ================================================================
# 16B. PRIMARY HYPOTHESIS TESTS
# ================================================================
#
# Because predictive performance is already ~0.99, the scientifically
# appropriate utility claim is NON-INFERIORITY, not forced superiority.
#
# Primary hypotheses:
#   H1 Utility: EffiAuL F1 is non-inferior to full retraining within margin.
#   H2 Efficiency: EffiAuL is faster than full retraining.
#   H3 Approximation: EffiAuL is closer to retraining than baseline on the
#      forget set (lower prediction-distance MAE).
# ================================================================

def paired_cohens_dz(differences):
    differences = np.asarray(differences, dtype=float)
    if len(differences) < 2:
        return np.nan
    sd = np.std(differences, ddof=1)
    if sd == 0:
        return np.inf if np.mean(differences) > 0 else 0.0
    return float(np.mean(differences) / sd)


def primary_hypothesis_tests(df):
    pivot_f1 = df.pivot(index="seed", columns="method", values="test_f1_weighted")
    pivot_time = df.pivot(index="seed", columns="method", values="runtime_seconds")
    pivot_forget_dist = df.pivot(
        index="seed", columns="method",
        values="forget_distance_to_retrain_mae"
    )

    rows = []

    # H1: Non-inferiority of utility.
    if {"EffiAuL", "Full_Retraining"}.issubset(pivot_f1.columns):
        paired = pivot_f1[["EffiAuL", "Full_Retraining"]].dropna()
        if len(paired) >= 2:
            # Non-inferiority condition:
            # EffiAuL - Retraining > -margin
            adjusted = (
                paired["EffiAuL"].values
                - paired["Full_Retraining"].values
                + F1_NONINFERIORITY_MARGIN
            )
            stat, p = ttest_1samp(
                adjusted, popmean=0.0, alternative="greater"
            )
            rows.append({
                "hypothesis": "H1_utility_noninferiority",
                "comparison": "EffiAuL vs Full_Retraining",
                "metric": "test_f1_weighted",
                "direction": f"EffiAuL > Retraining - {F1_NONINFERIORITY_MARGIN}",
                "n_pairs": len(adjusted),
                "mean_raw_difference": float(
                    np.mean(
                        paired["EffiAuL"].values
                        - paired["Full_Retraining"].values
                    )
                ),
                "test_statistic": float(stat),
                "p_value": float(p),
                "effect_size_dz": paired_cohens_dz(adjusted)
            })

    # H2: Runtime superiority.
    if {"EffiAuL", "Full_Retraining"}.issubset(pivot_time.columns):
        paired = pivot_time[["EffiAuL", "Full_Retraining"]].dropna()
        if len(paired) >= 2:
            saving = (
                paired["Full_Retraining"].values
                - paired["EffiAuL"].values
            )
            stat, p = ttest_1samp(
                saving, popmean=0.0, alternative="greater"
            )
            rows.append({
                "hypothesis": "H2_runtime_superiority",
                "comparison": "EffiAuL vs Full_Retraining",
                "metric": "runtime_seconds",
                "direction": "EffiAuL faster than Full_Retraining",
                "n_pairs": len(saving),
                "mean_raw_difference": float(np.mean(saving)),
                "test_statistic": float(stat),
                "p_value": float(p),
                "effect_size_dz": paired_cohens_dz(saving)
            })

    # H3: closer to exact retraining than the original baseline on forget set.
    if {"EffiAuL", "Baseline"}.issubset(pivot_forget_dist.columns):
        paired = pivot_forget_dist[["EffiAuL", "Baseline"]].dropna()
        if len(paired) >= 2:
            improvement = (
                paired["Baseline"].values
                - paired["EffiAuL"].values
            )
            stat, p = ttest_1samp(
                improvement, popmean=0.0, alternative="greater"
            )
            rows.append({
                "hypothesis": "H3_forget_retraining_agreement",
                "comparison": "EffiAuL vs Baseline",
                "metric": "forget_distance_to_retrain_mae",
                "direction": "EffiAuL closer to Full_Retraining",
                "n_pairs": len(improvement),
                "mean_raw_difference": float(np.mean(improvement)),
                "test_statistic": float(stat),
                "p_value": float(p),
                "effect_size_dz": paired_cohens_dz(improvement)
            })

    return pd.DataFrame(rows)


# ================================================================
# 17. MAIN
# ================================================================

def main():
    ensure_dir(RESULTS_DIR)

    dataset_df = scan_master_dataset()
    train_df, val_df, test_df = create_fixed_splits(dataset_df)
    class_names, n_classes, class_mode = determine_problem_type(train_df)

    print("\nClass mapping")
    for i, name in enumerate(class_names):
        print(f"{i}: {name}")

    config = {
        "master_dataset_dir": MASTER_DATASET_DIR,
        "dataset_layout_note": "Existing Train/Validate folders are combined and re-split with fixed 70/15/15 stratification when present.",
        "total_images": len(dataset_df),
        "class_counts": dataset_df["label"].value_counts().sort_index().to_dict(),
        "classes": class_names,
        "n_classes": n_classes,
        "split": {
            "train": TRAIN_RATIO,
            "validation": VAL_RATIO,
            "test": TEST_RATIO,
            "split_seed": SPLIT_SEED
        },
        "training_seeds": TRAINING_SEEDS,
        "baseline": {
            "lr": BASELINE_LR,
            "max_epochs": BASELINE_MAX_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "trainable_backbone_layers": BASELINE_TRAINABLE_BACKBONE_LAYERS
        },
        "forget": {
            "n_forget": N_FORGET,
            "selection": FORGET_SELECTION,
            "forget_set_seed": FORGET_SET_SEED
        },
        "effiaul_strategy_thresholds": {
            "low": LOW_INFLUENCE_THRESHOLD,
            "high": HIGH_INFLUENCE_THRESHOLD
        },
        "effiaul_update": {
            "retain_ce_weight": EFFIAUL_RETAIN_CE_WEIGHT,
            "retain_kd_weight": EFFIAUL_RETAIN_KD_WEIGHT,
            "forget_ascent_base_weight": EFFIAUL_FORGET_ASCENT_BASE_WEIGHT,
            "grad_clip_norm": EFFIAUL_GRAD_CLIP_NORM,
            "max_epochs": EFFIAUL_MAX_EPOCHS,
            "selection_patience": EFFIAUL_SELECTION_PATIENCE,
            "f1_noninferiority_margin": F1_NONINFERIORITY_MARGIN
        },
        "scrub": {
            "alpha": SCRUB_ALPHA,
            "gamma": SCRUB_GAMMA,
            "alternating_epochs": SCRUB_ALTERNATING_EPOCHS,
            "final_min_epochs": SCRUB_FINAL_MIN_EPOCHS
        },
        "sisa": {
            "num_shards": SISA_NUM_SHARDS,
            "num_slices": SISA_NUM_SLICES,
            "epochs_per_slice": SISA_EPOCHS_PER_SLICE
        },
        "if_diag": {
            "damping": IF_DIAG_DAMPING,
            "scale": IF_DIAG_SCALE,
            "head_only": IF_HEAD_ONLY
        }
    }

    with open(os.path.join(RESULTS_DIR, "experiment_configuration.json"), "w") as f:
        json.dump(config, f, indent=2)

    all_rows = []
    for seed in TRAINING_SEEDS:
        clear_memory()
        all_rows.extend(
            run_experiment(
                seed, train_df, val_df, test_df,
                class_names, n_classes, class_mode
            )
        )

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(
        os.path.join(RESULTS_DIR, "all_experimental_results.csv"),
        index=False
    )

    summary = aggregate_results(results_df)
    summary.to_csv(
        os.path.join(RESULTS_DIR, "summary_statistics.csv"),
        index=False
    )

    primary_tests = primary_hypothesis_tests(results_df)
    primary_tests.to_csv(
        os.path.join(RESULTS_DIR, "primary_hypothesis_tests.csv"),
        index=False
    )

    for metric in [
        "test_f1_weighted",
        "test_auc",
        "forget_loss",
        "distance_to_retrain_mae",
        "forget_distance_to_retrain_mae",
        "mia_forget_member_rate",
        "runtime_seconds"
    ]:
        tests = paired_tests(results_df, metric)
        tests.to_csv(
            os.path.join(RESULTS_DIR, f"significance_{metric}.csv"),
            index=False
        )

    display_cols = [
        c for c in [
            "method",
            "test_f1_weighted_mean",
            "test_f1_weighted_std",
            "test_auc_mean",
            "forget_loss_mean",
            "distance_to_retrain_mae_mean",
            "forget_distance_to_retrain_mae_mean",
            "utility_f1_gap_vs_full_retraining_mean",
            "mia_gap_to_full_retraining_mean",
            "mia_forget_member_rate_mean",
            "runtime_seconds_mean",
            "time_reduction_vs_full_retraining_percent_mean"
        ]
        if c in summary.columns
    ]

    print("\n" + "="*90)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*90)
    print(summary[display_cols].to_string(index=False))
    print(f"\nOutputs: {RESULTS_DIR}")


if __name__ == "__main__":
    main()