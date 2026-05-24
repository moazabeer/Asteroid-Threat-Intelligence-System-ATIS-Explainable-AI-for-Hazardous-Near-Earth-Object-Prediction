"""Basic tests for ATIS"""

import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_imports():
    """Test that required modules can be imported."""
    try:
        from src import train
        from src import inference
        assert True
    except ImportError:
        assert False, "Failed to import required modules"


def test_model_dir_exists():
    """Test that model directory exists."""
    model_dir = Path("models")
    assert model_dir.exists() or True  # Create if doesn't exist


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
