
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def load_and_preprocess_data(path="RML2016.10a.npy", snr_label=True, apply_smote=True, augment=False):
    """
    Load and preprocess RadioML2016.10a dataset.
    
    Returns:
        X_train, X_test, snr_train, snr_test, y_train, y_test
    """
    data = np.load(path, allow_pickle=True).item()
    X, y, snrs = [], [], []

    for (mod, snr), signals in data.items():
        for signal in signals:
            X.append(signal)
            y.append(mod)
            snrs.append(snr)

    X = np.array(X)  # shape (n_samples, 2, 128)
    X = np.transpose(X, (0, 2, 1))  # (n_samples, 128, 2)

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y)
    snrs = np.array(snrs).reshape(-1, 1)

    X_train, X_test, snr_train, snr_test, y_train, y_test = train_test_split(
        X, snrs, y, test_size=0.2, stratify=y, random_state=42)

    # Normalize per feature
    scaler = StandardScaler()
    X_train = X_train.reshape(-1, X_train.shape[-1])
    X_train = scaler.fit_transform(X_train).reshape(-1, 128, 2)
    X_test = X_test.reshape(-1, X_test.shape[-1])
    X_test = scaler.transform(X_test).reshape(-1, 128, 2)

    # Apply SMOTE
    if apply_smote:
        flat_X = X_train.reshape((X_train.shape[0], -1))
        smote = SMOTE(random_state=42)
        flat_X, y_train = smote.fit_resample(flat_X, y_train)
        snr_train = np.repeat(snr_train, 1, axis=0)[:flat_X.shape[0]]
        X_train = flat_X.reshape((-1, 128, 2))

    return X_train, X_test, snr_train, snr_test, y_train, y_test
