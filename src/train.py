"""
ATIS Training Module
Trains multiple ensemble models for asteroid hazard prediction
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix
)

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import kagglehub

warnings.filterwarnings("ignore")

# Configuration
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("output")
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def download_dataset():
    """Download the asteroid dataset from Kaggle."""
    print("Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("sakhawat18/asteroid-dataset")
    print(f"Dataset path: {path}")
    return path


def load_and_preprocess_data():
    """Load and preprocess the dataset."""
    print("\n=== Loading Data ===")
    
    # Download dataset
    dataset_path = download_dataset()
    
    # Load CSV
    df = pd.read_csv(os.path.join(dataset_path, "dataset.csv"), low_memory=False)
    print(f"Dataset shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum().sort_values(ascending=False).head(10)}")
    
    # Convert target and binary columns
    df["pha"] = df["pha"].map({"Y": 1, "N": 0})
    if "neo" in df.columns:
        df["neo"] = df["neo"].map({"Y": 1, "N": 0})
    
    # Drop non-numeric/non-useful columns
    drop_cols = [
        "id", "spkid", "full_name", "name", "pdes", "prefix",
        "orbit_id", "epoch_cal", "tp_cal", "equinox"
    ]
    for col in drop_cols:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
    
    # Encode categorical columns
    categorical_cols = df.select_dtypes(include="object").columns
    encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    
    # Separate features and target
    X = df.drop("pha", axis=1)
    y = df["pha"]
    
    print(f"\nClass distribution:\n{y.value_counts()}")
    
    return X, y, df, encoders


def split_and_balance_data(X, y):
    """Split data into train/test and apply SMOTE."""
    print("\n=== Splitting Data ===")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Handle missing values
    imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns
    )
    X_test = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns
    )
    
    # Apply SMOTE to handle class imbalance
    print("\n=== Applying SMOTE ===")
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"Training set after SMOTE: {X_train.shape[0]}")
    print(f"Class distribution after SMOTE:\n{y_train.value_counts()}")
    
    return X_train, X_test, y_train, y_test, imputer


def train_ensemble_models(X_train, y_train, X_test, y_test):
    """Train multiple ensemble models."""
    print("\n=== Training Ensemble Models ===")
    
    models = {}
    
    # Random Forest
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=20, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    models["RandomForest"] = rf
    
    # XGBoost
    print("Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=500, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        eval_metric="logloss", verbosity=0
    )
    xgb.fit(X_train, y_train)
    models["XGBoost"] = xgb
    
    # LightGBM
    print("Training LightGBM...")
    lgbm = LGBMClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=8,
        random_state=42, verbose=-1
    )
    lgbm.fit(X_train, y_train)
    models["LightGBM"] = lgbm
    
    # CatBoost
    print("Training CatBoost...")
    cat = CatBoostClassifier(
        iterations=500, learning_rate=0.05, depth=8, verbose=0
    )
    cat.fit(X_train, y_train)
    models["CatBoost"] = cat
    
    return models


def evaluate_models(models, X_test, y_test):
    """Evaluate all models."""
    print("\n=== Model Evaluation ===")
    
    results = {}
    for name, model in models.items():
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        results[name] = accuracy
        print(f"{name} Accuracy: {accuracy:.4f}")
    
    return results


def train_stacking_ensemble(models, X_train, y_train, X_test, y_test):
    """Create and train stacking ensemble."""
    print("\n=== Training Stacking Ensemble ===")
    
    rf = models["RandomForest"]
    xgb = models["XGBoost"]
    lgbm = models["LightGBM"]
    
    stack_model = StackingClassifier(
        estimators=[
            ('rf', rf),
            ('xgb', xgb),
            ('lgbm', lgbm)
        ],
        final_estimator=LogisticRegression(),
        cv=5
    )
    
    stack_model.fit(X_train, y_train)
    
    # Evaluate
    predictions = stack_model.predict(X_test)
    probs = stack_model.predict_proba(X_test)[:, 1]
    
    print("\nStacking Ensemble Metrics:")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision: {precision_score(y_test, predictions):.4f}")
    print(f"Recall: {recall_score(y_test, predictions):.4f}")
    print(f"F1: {f1_score(y_test, predictions):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, probs):.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, predictions)}")
    
    return stack_model


def save_models(models, stack_model, imputer, scaler=None):
    """Save trained models and preprocessing objects."""
    print("\n=== Saving Models ===")
    
    for name, model in models.items():
        model_path = MODEL_DIR / f"{name.lower()}_model.pkl"
        joblib.dump(model, model_path)
        print(f"Saved {name} to {model_path}")
    
    joblib.dump(stack_model, MODEL_DIR / "stacking_ensemble.pkl")
    print(f"Saved Stacking Ensemble to {MODEL_DIR / 'stacking_ensemble.pkl'}")
    
    joblib.dump(imputer, MODEL_DIR / "imputer.pkl")
    print(f"Saved Imputer to {MODEL_DIR / 'imputer.pkl'}")
    
    if scaler:
        joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
        print(f"Saved Scaler to {MODEL_DIR / 'scaler.pkl'}")


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("ATIS - Asteroid Threat Intelligence System")
    print("Training Pipeline")
    print("=" * 60)
    
    try:
        # Load and preprocess data
        X, y, df, encoders = load_and_preprocess_data()
        
        # Split and balance data
        X_train, X_test, y_train, y_test, imputer = split_and_balance_data(X, y)
        
        # Train ensemble models
        models = train_ensemble_models(X_train, y_train, X_test, y_test)
        
        # Evaluate models
        evaluate_models(models, X_test, y_test)
        
        # Train stacking ensemble
        stack_model = train_stacking_ensemble(models, X_train, y_train, X_test, y_test)
        
        # Save models
        save_models(models, stack_model, imputer)
        
        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise


if __name__ == "__main__":
    main()
