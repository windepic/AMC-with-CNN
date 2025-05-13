
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from data.process_data import load_and_preprocess_data

def plot_accuracy_by_snr(model_path="model_d_amc.h5"):
    model = load_model(model_path)
    X_train, X_test, snr_train, snr_test, y_train, y_test = load_and_preprocess_data()

    snr_values = sorted(set(snr_test.ravel()))
    acc_per_snr = {}

    for snr in snr_values:
        idx = np.where(snr_test.ravel() == snr)[0]
        if len(idx) == 0:
            continue
        preds = model.predict([X_test[idx], snr_test[idx]])
        preds = np.argmax(preds, axis=1)
        acc = np.mean(preds == y_test[idx])
        acc_per_snr[snr] = acc

    plt.figure(figsize=(10, 5))
    plt.plot(list(acc_per_snr.keys()), list(acc_per_snr.values()), marker='o')
    plt.xlabel("SNR")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. SNR")
    plt.grid(True)
    plt.savefig("results/accuracy_by_snr.png")
    plt.close()

def plot_confusion_matrix(cm_path="results/confusion_matrix.npy"):
    cm = np.load(cm_path)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("results/confusion_matrix.png")
    plt.close()

if __name__ == "__main__":
    plot_accuracy_by_snr()
    plot_confusion_matrix()
