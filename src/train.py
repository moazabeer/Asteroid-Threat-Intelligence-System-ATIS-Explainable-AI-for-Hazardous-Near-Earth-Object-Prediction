"""ATIS training module."""

from pathlib import Path
import warnings

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

MODEL_DIR = Path("models")
OUTPUT_DIR = Path("output")
DATASET_SLUG = "sakhawat18/asteroid-dataset"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def download_dataset(dataset_slug=DATASET_SLUG):
    """Download the asteroid dataset from Kaggle."""
    import kagglehub

    print("Downloading dataset from Kaggle...")
    dataset_path = Path(kagglehub.dataset_download(dataset_slug))
    print(f"Dataset path: {dataset_path}")
    return dataset_path


def load_and_preprocess_data():
    """Load the dataset and prepare numeric features."""
    print("\n=== Loading Data ===")
    dataset_path = download_dataset()
    dataset_file = dataset_path / "dataset.csv"
    dataframe = pd.read_csv(dataset_file, low_memory=False)
    print(f"Dataset shape: {dataframe.shape}")
    print(
        "Missing values:\n"
        f"{dataframe.isnull().sum().sort_values(ascending=False).head(10)}"
    )

    dataframe["pha"] = dataframe["pha"].map({"Y": 1, "N": 0})
    if "neo" in dataframe.columns:
        dataframe["neo"] = dataframe["neo"].map({"Y": 1, "N": 0})

    drop_columns = [
        "id",
        "spkid",
        "full_name",
        "name",
        "pdes",
        "prefix",
        "orbit_id",
        "epoch_cal",
        "tp_cal",
        "equinox",
    ]
    dataframe = dataframe.drop(columns=[col for col in drop_columns if col in dataframe.columns])

    for column in dataframe.select_dtypes(include="object").columns:
        encoder = LabelEncoder()
        dataframe[column] = encoder.fit_transform(dataframe[column].astype(str))

    features = dataframe.drop("pha", axis=1)
    target = dataframe["pha"]
    print(f"\nClass distribution:\n{target.value_counts()}")
    return features, target


def split_and_balance_data(features, target):
    """Split the data into train/test partitions and balance the training split."""
    print("\n=== Splitting Data ===")
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=42,
        stratify=target,
    )
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

    print("\n=== Applying SMOTE ===")
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"Training set after SMOTE: {X_train.shape[0]}")
    print(f"Class distribution after SMOTE:\n{y_train.value_counts()}")
    return X_train, X_test, y_train, y_test, imputer


def train_ensemble_models(X_train, y_train):
    """Train the base ensemble models."""
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    print("\n=== Training Ensemble Models ===")
    models = {}

    print("\nTraining Random Forest...")
    random_forest = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
    )
    random_forest.fit(X_train, y_train)
    models["RandomForest"] = random_forest

    print("Training XGBoost...")
    xgboost = XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )
    xgboost.fit(X_train, y_train)
    models["XGBoost"] = xgboost

    print("Training LightGBM...")
    lightgbm = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        random_state=42,
        verbose=-1,
    )
    lightgbm.fit(X_train, y_train)
    models["LightGBM"] = lightgbm

    print("Training CatBoost...")
    catboost = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=8,
        verbose=0,
    )
    catboost.fit(X_train, y_train)
    models["CatBoost"] = catboost
    return models


def evaluate_models(models, X_test, y_test):
    """Evaluate each trained base model."""
    print("\n=== Model Evaluation ===")
    results = {}
    for name, model in models.items():
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        results[name] = accuracy
        print(f"{name} Accuracy: {accuracy:.4f}")
    return results


def train_stacking_ensemble(models, X_train, y_train, X_test, y_test):
    """Train the final stacking ensemble and print its metrics."""
    print("\n=== Training Stacking Ensemble ===")
    stack_model = StackingClassifier(
        estimators=[
            ("rf", models["RandomForest"]),
            ("xgb", models["XGBoost"]),
            ("lgbm", models["LightGBM"]),
        ],
        final_estimator=LogisticRegression(),
        cv=5,
    )
    stack_model.fit(X_train, y_train)

    predictions = stack_model.predict(X_test)
    probabilities = stack_model.predict_proba(X_test)[:, 1]
    print("\nStacking Ensemble Metrics:")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision: {precision_score(y_test, predictions):.4f}")
    print(f"Recall: {recall_score(y_test, predictions):.4f}")
    print(f"F1: {f1_score(y_test, predictions):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, probabilities):.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, predictions)}")
    return stack_model


def save_models(models, stack_model, imputer, feature_columns):
    """Persist trained models and preprocessing artifacts."""
    print("\n=== Saving Models ===")
    for name, model in models.items():
        model_path = MODEL_DIR / f"{name.lower()}_model.pkl"
        joblib.dump(model, model_path)
        print(f"Saved {name} to {model_path}")

    joblib.dump(stack_model, MODEL_DIR / "stacking_ensemble.pkl")
    print(f"Saved Stacking Ensemble to {MODEL_DIR / 'stacking_ensemble.pkl'}")
    joblib.dump(imputer, MODEL_DIR / "imputer.pkl")
    print(f"Saved Imputer to {MODEL_DIR / 'imputer.pkl'}")
    joblib.dump(list(feature_columns), MODEL_DIR / "feature_columns.pkl")
    print(f"Saved feature list to {MODEL_DIR / 'feature_columns.pkl'}")


def main():
    """Run the full training pipeline."""
    print("=" * 60)
    print("ATIS - Asteroid Threat Intelligence System")
    print("Training Pipeline")
    print("=" * 60)

    try:
        features, target = load_and_preprocess_data()
        X_train, X_test, y_train, y_test, imputer = split_and_balance_data(
            features,
            target,
        )
        models = train_ensemble_models(X_train, y_train)
        evaluate_models(models, X_test, y_test)
        stack_model = train_stacking_ensemble(models, X_train, y_train, X_test, y_test)
        save_models(models, stack_model, imputer, features.columns)
        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print("=" * 60)
    except Exception as error:
        print(f"\nError during training: {error}")
        raise


if __name__ == "__main__":
    main()
