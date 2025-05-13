
from models.model_d import build_model_d
from data.process_data import load_and_preprocess_data
import tensorflow as tf

def train_model():
    X_train, X_test, snr_train, snr_test, y_train, y_test = load_and_preprocess_data()

    model = build_model_d(input_shape=(128, 2), snr_shape=(1,))
    model.summary()

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ]

    history = model.fit(
        [X_train, snr_train], y_train,
        validation_split=0.2,
        batch_size=128,
        epochs=50,
        callbacks=callbacks
    )

    test_loss, test_acc = model.evaluate([X_test, snr_test], y_test)
    print(f"Test Accuracy: {test_acc:.4f}")
    model.save("model_d_amc.h5")

if __name__ == "__main__":
    train_model()
