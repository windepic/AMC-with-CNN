
import numpy as np
from tensorflow.keras.models import load_model
from data.process_data import load_and_preprocess_data
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

def evaluate_model():
    model = load_model("model_d_amc.h5")
    X_train, X_test, snr_train, snr_test, y_train, y_test = load_and_preprocess_data()

    y_pred = model.predict([X_test, snr_test])
    y_pred_classes = np.argmax(y_pred, axis=1)

    acc = accuracy_score(y_test, y_pred_classes)
    print(f"Test Accuracy: {acc:.4f}")

    cm = confusion_matrix(y_test, y_pred_classes)
    np.save("results/confusion_matrix.npy", cm)
    return cm, y_test, y_pred_classes

if __name__ == "__main__":
    if not os.path.exists("results"):
        os.makedirs("results")
    evaluate_model()
