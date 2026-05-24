"""Basic tests for ATIS."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import inference, train


def test_imports():
    """Test that required modules can be imported."""
    assert train is not None
    assert inference is not None


def test_model_dir_exists():
    """Test that the model directory is available for artifacts."""
    assert Path("models").exists()


def test_classify_risk_thresholds():
    """Test the human-readable probability bands."""
    assert inference.ATISPredictor.classify_risk(0.10) == "Low"
    assert inference.ATISPredictor.classify_risk(0.30) == "Moderate"
    assert inference.ATISPredictor.classify_risk(0.60) == "High"
    assert inference.ATISPredictor.classify_risk(0.90) == "Critical"


def test_predictor_validates_feature_columns(monkeypatch, tmp_path):
    """Test that batch inference rejects missing required columns."""

    class FakeModel:
        def predict(self, features):
            return [0] * len(features)

        def predict_proba(self, features):
            return np.array([[0.9, 0.1] for _ in range(len(features))])

    class FakeImputer:
        def transform(self, frame):
            return frame

    artifacts = {
        "stacking_ensemble.pkl": FakeModel(),
        "imputer.pkl": FakeImputer(),
        "feature_columns.pkl": ["a", "e"],
    }

    def fake_load(path):
        return artifacts[Path(path).name]

    monkeypatch.setattr(inference.joblib, "load", fake_load)
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "feature_columns.pkl").write_text("placeholder", encoding="utf-8")
    predictor = inference.ATISPredictor(model_dir=model_dir)

    with pytest.raises(ValueError, match="Missing required feature columns: e"):
        predictor.predict(pd.DataFrame([{"a": 1.0}]))


def test_cli_passes_arguments_to_batch_predict(monkeypatch, tmp_path):
    """Test that the CLI forwards the requested input and output paths."""
    csv_path = tmp_path / "input.csv"
    output_path = tmp_path / "predictions.csv"
    csv_path.write_text("a,e\n1,2\n", encoding="utf-8")
    captured = {}

    def fake_batch_predict(input_path, output_file):
        captured["input_path"] = input_path
        captured["output_file"] = output_file

    monkeypatch.setattr(inference, "batch_predict", fake_batch_predict)
    inference.main([str(csv_path), "--output", str(output_path)])

    assert captured == {
        "input_path": str(csv_path),
        "output_file": str(output_path),
    }
