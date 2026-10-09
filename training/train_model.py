"""
now we need to upload the dataset we collected then split it into training/validation/testing
then we need to implement the 3 classifiers
1- KNN 2- SVM 3- Random Forest
after that we evaluate and compare them
after we are happy with the results we can save and use the models
"""



"""
Hand Sign Recognition - Model Training
--------------------------------------
Loads and prepares the collected hand sign dataset.

Run from the project root:
    python -m training.train_model
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.feature_extractor import FeatureExtractor

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

import joblib



DATA_DIR = Path("data/collected")
MODEL_DIR = Path("models")

LETTERS = "ABCDEFGHIKLMNOPQRSTUVWXY"
DIGITS = "0123456789"
LABELS = sorted(LETTERS + DIGITS)

NUM_FEATURES = FeatureExtractor.NUM_FEATURES
RANDOM_STATE = 42


def load_dataset():
    """
    Load all collected CSV files.

    Returns:
        X: NumPy array of shape (num_samples, 63).
        y: NumPy array containing the corresponding labels.
    """
    X_parts = []
    y_parts = []

    expected_columns = [f"feature_{i}" for i in range(NUM_FEATURES)]

    for label in LABELS:
        file_path = DATA_DIR / f"{label}.csv"

        if not file_path.exists():
            raise FileNotFoundError(f"Missing dataset file: {file_path}")

        df = pd.read_csv(file_path)

        if list(df.columns) != expected_columns:
            raise ValueError(f"Invalid feature columns in {file_path}")

        features = df.to_numpy(dtype=np.float32)

        if len(features) == 0:
            raise ValueError(f"No samples found in {file_path}")

        if not np.isfinite(features).all():
            raise ValueError(f"Invalid feature values in {file_path}")

        X_parts.append(features)
        y_parts.extend([label] * len(features))

        print(f"{label}: {len(features)} samples")

    X = np.vstack(X_parts)
    y = np.array(y_parts)

    return X, y


def split_dataset(X, y):
    """
    Split the dataset into 80% training and 20% testing.

    Stratification preserves class proportions.
    """
    return train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )


def train_knn(X_train, y_train):
    """
    Train a KNN classifier with standardized features.
    """
    model = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5)
    )

    model.fit(X_train, y_train)

    return model


def train_svm(X_train, y_train):
    """
    Train a Support Vector Machine classifier.
    """
    model = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=10, gamma="scale")
    )

    model.fit(X_train, y_train)

    return model


def train_random_forest(X_train, y_train):
    """
    Train a Random Forest classifier.
    """
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained classifier on the test dataset.
    """
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"\nAccuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))


def save_model(model, name):
    """
    Save a trained classifier to disk.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    path = MODEL_DIR / f"{name}.joblib"
    joblib.dump(model, path)

    print(f"Model saved: {path}")


def main():
    X, y = load_dataset()

    print("\nDATASET SUMMARY")
    print("-" * 35)
    print(f"Total samples: {len(X)}")
    print(f"Features per sample: {X.shape[1]}")
    print(f"Number of classes: {len(np.unique(y))}")

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    print("\nDATASET SPLIT")
    print("-" * 35)
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")
    print(f"Training labels:  {len(y_train)}")
    print(f"Testing labels:   {len(y_test)}")


    print("\nTRAINING KNN")
    print("-" * 35)

    knn = train_knn(X_train, y_train)

    print("\nEVALUATING KNN")
    print("-" * 35)

    evaluate_model(knn, X_test, y_test)
    save_model(knn, "knn")


    print("\nTRAINING SVM")
    print("-" * 35)

    svm = train_svm(X_train, y_train)

    print("\nEVALUATING SVM")
    print("-" * 35)

    evaluate_model(svm, X_test, y_test)
    save_model(svm, "svm")


    print("\nTRAINING RANDOM FOREST")
    print("-" * 35)

    forest = train_random_forest(X_train, y_train)

    print("\nEVALUATING RANDOM FOREST")
    print("-" * 35)

    evaluate_model(forest, X_test, y_test)
    save_model(forest, "random_forest")

if __name__ == "__main__":
    main()
