"""ATIS inference module."""

import argparse
from pathlib import Path

import joblib
import pandas as pd

MODEL_DIR = Path("models")


class ATISPredictor:
    """Load trained artifacts and make hazard predictions."""

    def __init__(self, model_path=None, model_dir=MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.model_path = Path(model_path or self.model_dir / "stacking_ensemble.pkl")
        self.model = joblib.load(self.model_path)
        self.imputer = joblib.load(self.model_dir / "imputer.pkl")
        feature_columns_path = self.model_dir / "feature_columns.pkl"
        self.feature_columns = (
            joblib.load(feature_columns_path) if feature_columns_path.exists() else None
        )

    def _prepare_features(self, features):
        frame = pd.DataFrame(features).copy()
        if self.feature_columns:
            missing_columns = [
                column for column in self.feature_columns if column not in frame.columns
            ]
            if missing_columns:
                missing_text = ", ".join(sorted(missing_columns))
                raise ValueError(f"Missing required feature columns: {missing_text}")
            frame = frame.loc[:, self.feature_columns]
        return self.imputer.transform(frame)

    def predict(self, features):
        """Predict classes and probabilities for one or more samples."""
        prepared_features = self._prepare_features(features)
        predictions = self.model.predict(prepared_features)
        probabilities = self.model.predict_proba(prepared_features)[:, 1]
        return predictions, probabilities

    def predict_single(self, feature_dict):
        """Predict for a single sample."""
        prediction, probability = self.predict(pd.DataFrame([feature_dict]))
        return prediction[0], probability[0]

    @staticmethod
    def classify_risk(probability):
        """Map a probability to a human-readable risk label."""
        if probability < 0.25:
            return "Low"
        if probability < 0.50:
            return "Moderate"
        if probability < 0.75:
            return "High"
        return "Critical"


def batch_predict(csv_path, output_path="predictions.csv"):
    """Predict hazard probabilities for every row in a CSV file."""
    predictor = ATISPredictor()
    input_frame = pd.read_csv(csv_path)
    predictions, probabilities = predictor.predict(input_frame)
    results = pd.DataFrame(
        {
            "Index": range(len(predictions)),
            "Prediction": predictions,
            "Probability": probabilities,
            "Risk_Class": [predictor.classify_risk(probability) for probability in probabilities],
        }
    )
    results.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    return results


def build_parser():
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Run ATIS batch inference against a CSV file."
    )
    parser.add_argument("csv_path", help="Path to the input CSV file.")
    parser.add_argument(
        "-o",
        "--output",
        default="predictions.csv",
        help="Path to the output CSV file.",
    )
    return parser


def main(argv=None):
    """CLI entry point for batch inference."""
    args = build_parser().parse_args(argv)
    batch_predict(args.csv_path, args.output)


if __name__ == "__main__":
    main()
