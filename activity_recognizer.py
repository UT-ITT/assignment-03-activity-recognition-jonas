"""
activity_recognizer.py — ML classifier for activity recognition from DIPPID sensor data.
Loads CSV files, extracts features, trains a Random Forest classifier.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

WINDOW_SIZE = 20    # 2 seconds at ~10Hz (training data downsampled, live data native rate)
STEP_SIZE = 10

ACTIVITIES = ["running", "rowing", "lifting", "jumpingjacks"]


def extract_features(window: np.ndarray) -> np.ndarray:
    """Extract time- and frequency-domain features from a sensor window.

    Args:
        window: (WINDOW_SIZE x 6) array with columns [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]

    Returns:
        1D feature vector
    """
    # Add orientation-invariant magnitude channels: ||acc|| and ||gyro||
    acc_mag = np.linalg.norm(window[:, :3], axis=1, keepdims=True)
    gyro_mag = np.linalg.norm(window[:, 3:], axis=1, keepdims=True)
    window = np.hstack([window, acc_mag, gyro_mag])  # shape: (WINDOW_SIZE, 8)

    features = []
    for col in range(window.shape[1]):
        signal = window[:, col]
        # Time domain
        features.append(np.mean(signal))
        features.append(np.std(signal))
        features.append(np.min(signal))
        features.append(np.max(signal))
        features.append(np.max(signal) - np.min(signal))  # range
        # Frequency domain (FFT magnitudes — top 5)
        fft_mag = np.abs(np.fft.rfft(signal))
        fft_mag = fft_mag[1:]  # skip DC component
        top_k = np.sort(fft_mag)[-5:][::-1]
        features.extend(top_k.tolist())
    return np.array(features)


def load_data(data_dirs: "str | list[str]"):
    """Load all CSV files from one or more directories and extract windowed features.

    Accepts a single directory path or a list of paths. For the shared repo layout
    (one sub-folder per person), each sub-folder is scanned automatically.

    Returns:
        X (np.ndarray): feature matrix
        y (list[str]): activity labels
    """
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]

    # Expand any directory that contains only sub-directories (shared-repo layout)
    expanded = []
    for d in data_dirs:
        entries = os.listdir(d)
        csv_entries = [e for e in entries if e.endswith(".csv")]
        if csv_entries:
            expanded.append(d)  # flat layout — use as-is
        else:
            for sub in entries:
                sub_path = os.path.join(d, sub)
                if os.path.isdir(sub_path):
                    expanded.append(sub_path)

    X, y = [], []
    cols = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    total_files = 0

    for data_dir in expanded:
        csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
        for filename in csv_files:
            parts = filename.replace(".csv", "").split("-")
            if len(parts) < 3:
                continue
            activity = parts[1]
            if activity not in ACTIVITIES:
                continue

            df = pd.read_csv(os.path.join(data_dir, filename))
            if not all(c in df.columns for c in cols):
                print(f"Skipping {filename}: missing columns")
                continue

            data = df[cols].dropna().values[::10]  # downsample 100Hz → ~10Hz
            for start in range(0, len(data) - WINDOW_SIZE + 1, STEP_SIZE):
                window = data[start:start + WINDOW_SIZE]
                X.append(extract_features(window))
                y.append(activity)
            total_files += 1

    if not total_files:
        raise FileNotFoundError(f"No valid CSV files found in: {data_dirs}")

    print(f"Loaded {total_files} CSV files → {len(y)} windows.")
    return np.array(X), y


class ActivityRecognizer:
    """Trains and runs a Random Forest activity classifier."""

    def __init__(self, data_dir: "str | list[str]" = "data"):
        self.data_dirs = data_dir
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.buffer = []
        self._last_sample = None
        self.trained = False

    def train(self):
        """Load data, train classifier, print test accuracy."""
        print("Loading training data...")
        X, y = load_data(self.data_dirs)
        print(f"Loaded {len(y)} windows from {len(set(y))} activities.")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        print("Training classifier...")
        self.classifier.fit(X_train, y_train)
        self.trained = True

        y_pred = self.classifier.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Test accuracy: {acc * 100:.1f}%")
        return acc

    def update(self, acc: dict, gyro: dict) -> str | None:
        """Feed one sensor sample. Returns a prediction once a full window is ready."""
        if not self.trained:
            return None
        sample = [acc["x"], acc["y"], acc["z"], gyro["x"], gyro["y"], gyro["z"]]
        if sample == self._last_sample:
            return None
        self._last_sample = sample
        self.buffer.append(sample)
        if len(self.buffer) < WINDOW_SIZE:
            return None
        window = np.array(self.buffer[-WINDOW_SIZE:])
        features = extract_features(window).reshape(1, -1)
        features = self.scaler.transform(features)
        prediction = self.classifier.predict(features)[0]
        self.buffer = self.buffer[STEP_SIZE:]
        return prediction
