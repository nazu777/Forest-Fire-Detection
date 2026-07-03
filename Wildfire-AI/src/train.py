import tensorflow as tf

from model import build_model, apply_swir_weight_transfer
from dataset import prepare_datasets

# ==========================================
# CONFIG
# ==========================================

BASE_DIR = "../data/dataset_all"

INPUT_SHAPE = (128, 128, 4)

ALPHA = 0.5
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.0005

MODEL_SAVE_PATH = "../models/10_Layers.h5"

# ==========================================
# LOAD DATA
# ==========================================

print("=" * 50)
print("Loading datasets...")
print("=" * 50)

train_ds, val_ds, test_ds = prepare_datasets(
    BASE_DIR,
    BATCH_SIZE
)

# ==========================================
# BUILD MODEL
# ==========================================

print("=" * 50)
print("Building model...")
print("=" * 50)

model = build_model(
    input_shape=INPUT_SHAPE,
    alpha=ALPHA
)

model = apply_swir_weight_transfer(
    model,
    alpha=ALPHA
)

# ==========================================
# COMPILE
# ==========================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss=tf.keras.losses.BinaryCrossentropy(
        from_logits=True
    ),
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

model.summary()

# ==========================================
# TRAIN
# ==========================================

print("=" * 50)
print("Training started...")
print("=" * 50)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# ==========================================
# TEST
# ==========================================

print("=" * 50)
print("Evaluating on test set...")
print("=" * 50)

results = model.evaluate(test_ds)

print()

for metric, value in zip(model.metrics_names, results):
    print(f"{metric}: {value:.4f}")

# ==========================================
# SAVE MODEL
# ==========================================

model.save(MODEL_SAVE_PATH)

print()
print(f"Model saved to {MODEL_SAVE_PATH}")