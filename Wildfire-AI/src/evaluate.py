import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve,
    auc
)

from dataset import prepare_datasets

# ==========================================
# CONFIG
# ==========================================

MODEL_PATH = "../models/10_Layers.h5"
DATASET_PATH = "../data/dataset_all"
RESULTS_DIR = "../results"

BATCH_SIZE = 64

os.makedirs(RESULTS_DIR, exist_ok=True)

# ==========================================
# LOAD MODEL
# ==========================================

print("Loading model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

# ==========================================
# LOAD TEST DATA
# ==========================================

_, _, test_ds = prepare_datasets(
    DATASET_PATH,
    BATCH_SIZE
)

# ==========================================
# PREDICTIONS
# ==========================================

y_true = []
y_scores = []

for images, labels in test_ds:

    logits = model.predict(images, verbose=0)

    probs = tf.sigmoid(logits).numpy().flatten()

    y_scores.extend(probs)

    y_true.extend(labels.numpy())

y_true = np.array(y_true)
y_scores = np.array(y_scores)

y_pred = (y_scores >= 0.5).astype(int)

# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report\n")

print(classification_report(
    y_true,
    y_pred,
    target_names=["Non Fire", "Fire"]
))

# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_true,
    y_pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Non Fire", "Fire"]
)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "confusion_matrix.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================
# ROC CURVE
# ==========================================

fpr, tpr, _ = roc_curve(
    y_true,
    y_scores
)

roc_auc = auc(
    fpr,
    tpr
)

plt.figure(figsize=(6,6))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc:.3f}"
)

plt.plot(
    [0,1],
    [0,1],
    "--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "roc_curve.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================
# SAMPLE PREDICTIONS
# ==========================================

images, labels = next(iter(test_ds))

predictions = model.predict(images, verbose=0)
predictions = tf.sigmoid(predictions).numpy()

plt.figure(figsize=(12,8))

for i in range(min(6, len(images))):

    plt.subplot(2,3,i+1)

    rgb = images[i][:,:,:3].numpy()

    plt.imshow(rgb)

    pred = "Fire" if predictions[i] >= 0.5 else "Non Fire"

    actual = "Fire" if labels[i] == 1 else "Non Fire"

    plt.title(f"P:{pred}\nA:{actual}")

    plt.axis("off")

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "sample_predictions.png"
    ),
    dpi=300
)

plt.close()

print()

print("Evaluation completed.")

print("Results saved inside results/")