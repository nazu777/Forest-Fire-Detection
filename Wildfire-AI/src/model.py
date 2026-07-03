import numpy as np
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    DepthwiseConv2D,
    BatchNormalization,
    ReLU,
    GlobalAveragePooling2D,
    Dense
)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2


def build_model(input_shape=(128, 128, 4), alpha=0.5):
    """
    Builds the 10-layer wildfire classification model.
    """

    inputs = Input(shape=input_shape)

    # Layer 1
    x = Conv2D(
        int(32 * alpha),
        kernel_size=3,
        strides=2,
        padding="same",
        use_bias=False,
        name="Conv1"
    )(inputs)

    # Layer 2
    x = BatchNormalization()(x)

    # Layer 3
    x = ReLU()(x)

    # Layer 4
    x = DepthwiseConv2D(
        kernel_size=3,
        padding="same",
        use_bias=False
    )(x)

    # Layer 5
    x = BatchNormalization()(x)

    # Layer 6
    x = ReLU()(x)

    # Layer 7
    x = Conv2D(
        int(64 * alpha),
        kernel_size=1,
        use_bias=False
    )(x)

    # Layer 8
    x = GlobalAveragePooling2D()(x)

    # Layer 9
    x = Dense(
        32,
        activation="relu"
    )(x)

    # Layer 10
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)

    return model


def apply_swir_weight_transfer(model, alpha=0.5):
    """
    Transfers ImageNet MobileNetV2 weights
    from RGB to 4-channel Sentinel-2 input.
    """

    print("[INFO] Loading ImageNet weights...")

    base_model = MobileNetV2(
        input_shape=(128, 128, 3),
        alpha=alpha,
        include_top=False,
        weights="imagenet"
    )

    base_weights = base_model.layers[1].get_weights()[0]

    # Copy RED channel into SWIR channel
    red_channel = base_weights[:, :, 0:1, :]
    new_weights = np.concatenate(
        [base_weights, red_channel],
        axis=2
    )

    conv1 = model.get_layer("Conv1")

    conv1.set_weights(
        [new_weights[:, :, :, :conv1.filters]]
    )

    print("[INFO] SWIR weight transfer completed.")

    return model