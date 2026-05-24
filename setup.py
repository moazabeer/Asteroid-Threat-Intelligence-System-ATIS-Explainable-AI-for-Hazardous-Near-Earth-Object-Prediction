"""Setup configuration for ATIS package"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="atis",
    version="1.0.0",
    author="ATIS Contributors",
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
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "atis-train=src.train:main",
            "atis-predict=src.inference:batch_predict",
        ],
    },
)
