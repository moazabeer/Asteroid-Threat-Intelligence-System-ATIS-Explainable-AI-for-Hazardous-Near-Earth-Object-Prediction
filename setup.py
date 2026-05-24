"""Setup configuration for the ATIS package."""

from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent.resolve()
README = ROOT / "README.md"
REQUIREMENTS = ROOT / "requirements.txt"

long_description = README.read_text(encoding="utf-8")
requirements = [
    line.strip()
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.startswith("#")
]

setup(
    name="atis",
    version="1.0.0",
    author="ATIS Contributors",
    license="MIT",
    description="Asteroid Threat Intelligence System - Explainable AI for Hazardous Near-Earth Object Prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/moazabeer/Asteroid-Threat-Intelligence-System-ATIS-Explainable-AI-for-Hazardous-Near-Earth-Object-Prediction",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "atis-train=src.train:main",
            "atis-predict=src.inference:main",
        ],
    },
)
