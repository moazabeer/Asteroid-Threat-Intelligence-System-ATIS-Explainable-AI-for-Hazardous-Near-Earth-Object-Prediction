# Dockerfile for ATIS

FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models output logs

# Default command - training
CMD ["python", "-m", "src.train"]

# Labels
LABEL maintainer="ATIS Contributors"
LABEL description="Asteroid Threat Intelligence System - Explainable AI for Hazardous Near-Earth Object Prediction"
LABEL version="1.0.0"
