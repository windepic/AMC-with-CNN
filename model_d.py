
import tensorflow as tf
from tensorflow.keras import layers, models, Input

def build_model_d(input_shape=(128, 2), snr_shape=(1,)):
    """
    Residual 1D CNN model with SNR fusion.
    
    Parameters:
        input_shape (tuple): Shape of the IQ signal input.
        snr_shape (tuple): Shape of the SNR auxiliary input.
    
    Returns:
        model (tf.keras.Model): Compiled Keras model.
    """
    # Main input branch (I/Q data)
    iq_input = Input(shape=input_shape, name="iq_input")

    # First Conv block
    x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(iq_input)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # Residual Block 1
    shortcut = x
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.3)(x)

    # Residual Block 2
    shortcut = x
    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.4)(x)

    # Flatten and dense
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    # SNR input branch
    snr_input = Input(shape=snr_shape, name="snr_input")
    snr_dense = layers.Dense(32, activation="relu")(snr_input)

    # Merge branches
    merged = layers.Concatenate()([x, snr_dense])
    merged = layers.Dense(64, activation="relu")(merged)
    merged = layers.Dropout(0.3)(merged)

    output = layers.Dense(11, activation="softmax")(merged)

    model = models.Model(inputs=[iq_input, snr_input], outputs=output)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
