"""Test initialization"""

import pytest
from pathlib import Path

# Create necessary directories
Path("models").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
