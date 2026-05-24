# Build and Deploy Documentation

## Quick Start

### Local Development

```bash
# Clone repository
git clone https://github.com/moazabeer/Asteroid-Threat-Intelligence-System-ATIS-Explainable-AI-for-Hazardous-Near-Earth-Object-Prediction.git
cd Asteroid-Threat-Intelligence-System-ATIS-Explainable-AI-for-Hazardous-Near-Earth-Object-Prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run training
python -m src.train

# Run tests
pytest tests/ -v
```

### Docker Deployment

#### Build Docker Image

```bash
docker build -t atis:latest .
```

#### Run Training in Docker

```bash
docker run --rm -v $(pwd)/models:/app/models atis:latest python -m src.train
```

#### Run Inference in Docker

```bash
docker run --rm -v $(pwd)/models:/app/models -v $(pwd)/data:/app/data atis:latest python -m src.inference data/sample.csv
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  atis-train:
    build: .
    container_name: atis-training
    volumes:
      - ./models:/app/models
      - ./output:/app/output
    environment:
      - PYTHONUNBUFFERED=1
    command: python -m src.train

  atis-inference:
    build: .
    container_name: atis-inference
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    environment:
      - PYTHONUNBUFFERED=1
    depends_on:
      - atis-train
```

Run with:

```bash
docker-compose up
```

## Deployment Options

### 1. Kubernetes Deployment

Create `k8s-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: atis-trainer
spec:
  replicas: 1
  selector:
    matchLabels:
      app: atis
  template:
    metadata:
      labels:
        app: atis
    spec:
      containers:
      - name: atis
        image: ghcr.io/moazabeer/atis:latest
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        volumeMounts:
        - name: models
          mountPath: /app/models
        - name: output
          mountPath: /app/output
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: atis-models-pvc
      - name: output
        persistentVolumeClaim:
          claimName: atis-output-pvc
```

Deploy:

```bash
kubectl apply -f k8s-deployment.yaml
```

### 2. AWS Deployment (ECS)

Use AWS ECR for image storage:

```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Tag image
docker tag atis:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/atis:latest

# Push image
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/atis:latest

# Create ECS task definition and service
```

### 3. Google Cloud Deployment (GCP)

```bash
# Push to Google Container Registry
docker tag atis:latest gcr.io/<project-id>/atis:latest
docker push gcr.io/<project-id>/atis:latest

# Deploy on Cloud Run
gcloud run deploy atis \
  --image gcr.io/<project-id>/atis:latest \
  --platform managed \
  --region us-central1 \
  --memory 8Gi \
  --cpu 4
```

### 4. Azure Deployment

```bash
# Login to Azure Container Registry
az acr login --name <registry-name>

# Tag and push image
docker tag atis:latest <registry-name>.azurecr.io/atis:latest
docker push <registry-name>.azurecr.io/atis:latest

# Deploy to Azure Container Instances
az container create \
  --resource-group <group-name> \
  --name atis-container \
  --image <registry-name>.azurecr.io/atis:latest \
  --memory 8 \
  --cpu 4
```

## GitHub Container Registry (GHCR)

The project automatically pushes Docker images to GHCR on each push to main:

```bash
# Pull latest image
docker pull ghcr.io/moazabeer/atis:latest

# Run container
docker run -v $(pwd)/models:/app/models ghcr.io/moazabeer/atis:latest
```

## CI/CD Pipeline

The project includes GitHub Actions workflows:

1. **build.yml** - Runs on every push/PR:
   - Tests across Python 3.9, 3.10, 3.11
   - Linting with flake8
   - Coverage reporting
   - Docker image build and push to GHCR

2. **release.yml** - Runs on version tags (v*):
   - Creates GitHub release
   - Builds and pushes tagged Docker images

## Environment Variables

Set these for Docker deployment:

- `PYTHONUNBUFFERED=1` - Enable Python output buffering
- Any Kaggle API keys if needed for data download

## Production Deployment Checklist

- [ ] Review and update `requirements.txt` versions
- [ ] Set up Kaggle API credentials if using automated data fetch
- [ ] Configure persistent volumes for models and output
- [ ] Set up monitoring and logging
- [ ] Configure resource limits appropriately
- [ ] Set up automated backups for trained models
- [ ] Test deployment in staging environment
- [ ] Document access credentials and deployment parameters
- [ ] Set up alerting for failed runs
- [ ] Configure auto-scaling if using cloud platforms

## Troubleshooting

### Import Errors
Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Out of Memory
Increase Docker memory allocation or reduce batch sizes.

### Model Not Found
Ensure models directory is properly mounted in Docker containers.

### Kaggle Dataset Download Fails
Set up Kaggle API credentials in `~/.kaggle/kaggle.json`

## Support

For issues or questions, please open an issue on GitHub.
