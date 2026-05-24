"""
ATIS Inference Module
Makes predictions using trained models
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path

MODEL_DIR = Path("models")


class ASTISPredictor:
    """Predictor class for ATIS models."""
    
    def __init__(self, model_path=None):
        """Initialize predictor with trained model."""
        if model_path is None:
            model_path = MODEL_DIR / "stacking_ensemble.pkl"
        
        self.model = joblib.load(model_path)
        self.imputer = joblib.load(MODEL_DIR / "imputer.pkl")
        print(f"Loaded model from {model_path}")
    
    def predict(self, features):
        """Make predictions."""
        # Impute missing values
        features_imputed = self.imputer.transform(features)
        
        # Make predictions
        predictions = self.model.predict(features_imputed)
        probabilities = self.model.predict_proba(features_imputed)[:, 1]
        
        return predictions, probabilities
    
    def predict_single(self, feature_dict):
        """Predict for a single sample."""
        # Convert dict to DataFrame
        df = pd.DataFrame([feature_dict])
        
        # Make prediction
        pred, prob = self.predict(df)
        
        return pred[0], prob[0]
    
    def classify_risk(self, probability):
        """Classify risk level based on probability."""
        if probability < 0.25:
            return "Low"
        elif probability < 0.50:
            return "Moderate"
        elif probability < 0.75:
            return "High"
        else:
            return "Critical"


def batch_predict(csv_path, output_path="predictions.csv"):
    """Predict for batch of asteroids from CSV."""
    predictor = ASTISPredictor()
    
    # Load data
    df = pd.read_csv(csv_path)
    
    # Make predictions
    predictions, probabilities = predictor.predict(df)
    
    # Create output dataframe
    results = pd.DataFrame({
        "Index": range(len(predictions)),
        "Prediction": predictions,
        "Probability": probabilities,
        "Risk_Class": [predictor.classify_risk(p) for p in probabilities]
    })
    
    # Save results
    results.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
        batch_predict(csv_file)
    else:
        print("Usage: python inference.py <csv_file>")
