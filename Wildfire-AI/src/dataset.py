import os
import glob
import numpy as np
import tensorflow as tf
import tifffile

from sklearn.model_selection import train_test_split

INPUT_SHAPE = (128, 128, 4)


def load_dataset(base_dir):
    """
    Reads all image paths and creates train/validation/test splits.
    """

    fire_paths = glob.glob(
        os.path.join(base_dir, "forest_fire", "**", "*.tif"),
        recursive=True
    )

    non_fire_paths = glob.glob(
        os.path.join(base_dir, "non_forest_fire", "**", "*.tif"),
        recursive=True
    )

    all_paths = fire_paths + non_fire_paths
    all_labels = [1] * len(fire_paths) + [0] * len(non_fire_paths)

    x_temp, x_test, y_temp, y_test = train_test_split(
        all_paths,
        all_labels,
        test_size=0.15,
        stratify=all_labels,
        random_state=42
    )

    x_train, x_val, y_train, y_val = train_test_split(
        x_temp,
        y_temp,
        test_size=0.1765,
        stratify=y_temp,
        random_state=42
    )

    return (
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test
    )


def parse_tif(file_path, label):
    """
    Reads a Sentinel-2 TIFF image and preprocesses it.
    """

    def _read(path):

        path = path.decode("utf-8")

        image = tifffile.imread(path).astype(np.float32)

        image = tf.image.resize(
            image,
            (128, 128)
        ).numpy()

        image = np.nan_to_num(image)

        if np.max(image) > 2:
            image = image / 10000.0

        image = np.clip(image, 0.0, 1.0)

        return image

    image = tf.numpy_function(
        _read,
        [file_path],
        tf.float32
    )

    image.set_shape(INPUT_SHAPE)

    return image, tf.cast(label, tf.float32)


def create_dataset(
    image_paths,
    labels,
    batch_size=64,
    shuffle=False
):
    """
    Creates a TensorFlow Dataset object.
    """

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, labels)
    )

    if shuffle:
        dataset = dataset.shuffle(1000)

    dataset = dataset.map(
        parse_tif,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def prepare_datasets(base_dir, batch_size=64):
    """
    Returns train, validation and test datasets.
    """

    (
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test
    ) = load_dataset(base_dir)

    train_ds = create_dataset(
        x_train,
        y_train,
        batch_size=batch_size,
        shuffle=True
    )

    val_ds = create_dataset(
        x_val,
        y_val,
        batch_size=batch_size
    )

    test_ds = create_dataset(
        x_test,
        y_test,
        batch_size=batch_size
    )

    return train_ds, val_ds, test_ds