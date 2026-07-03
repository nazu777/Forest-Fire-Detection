import argparse
import numpy as np
import tensorflow as tf
import tifffile


INPUT_SHAPE = (128, 128, 4)

MODEL_PATH = "../models/10_Layers.h5"


def load_image(image_path):
    """
    Reads and preprocesses a Sentinel-2 TIFF image.
    """

    image = tifffile.imread(image_path).astype(np.float32)

    image = tf.image.resize(
        image,
        (128, 128)
    ).numpy()

    image = np.nan_to_num(image)

    if np.max(image) > 2:
        image = image / 10000.0

    image = np.clip(image, 0.0, 1.0)

    image = np.expand_dims(image, axis=0)

    return image


def predict(image_path):

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    image = load_image(image_path)

    logits = model.predict(image, verbose=0)

    probability = tf.sigmoid(logits).numpy()[0][0]

    print("=" * 40)
    print(f"Prediction Score : {probability:.4f}")

    if probability >= 0.5:
        print("Prediction      : 🔥 Forest Fire")
    else:
        print("Prediction      : ✅ Non Forest Fire")

    print("=" * 40)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True,
        help="Path to TIFF image"
    )

    args = parser.parse_args()

    predict(args.image)