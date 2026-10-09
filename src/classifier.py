"""
the idea now is that we have 3 pre-trained models
now we wanna use them so we load them, and then we start collecting frames every
fixed amount of time then convert it to normalized hand landmark features, give it to
the 3 models and get a predication
"""



"""
Hand Sign Classifier
--------------------
Loads the trained KNN, SVM, and Random Forest models.

Each model receives the same 63 normalized hand-landmark features
and produces an independent prediction.
"""

from pathlib import Path

import joblib
import numpy as np

from src.feature_extractor import FeatureExtractor


class Classifier:
    MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

    MODEL_FILES = {
        "KNN": "knn.joblib",
        "SVM": "svm.joblib",
        "Random Forest": "random_forest.joblib",
    }

    def __init__(self):
        self.models = {}

        for name, filename in self.MODEL_FILES.items():
            model_path = self.MODEL_DIR / filename

            if not model_path.is_file():
                raise FileNotFoundError(f"Trained model not found: {model_path}")

            self.models[name] = joblib.load(model_path)

    def predict(self, features):
        """
        Predict a hand sign using all three models.

        Args:
            features: NumPy array containing 63 normalized features.

        Returns:
            Dictionary mapping each model name to its predicted sign.
        """
        features = np.asarray(features, dtype=np.float32)

        if features.shape != (FeatureExtractor.NUM_FEATURES,):
            raise ValueError(
                f"Expected {FeatureExtractor.NUM_FEATURES} features, "
                f"got shape {features.shape}"
            )

        if not np.isfinite(features).all():
            raise ValueError("Features contain invalid numeric values")

        features = features.reshape(1, -1)

        predictions = {}

        for name, model in self.models.items():
            predictions[name] = str(model.predict(features)[0])

        return predictions
