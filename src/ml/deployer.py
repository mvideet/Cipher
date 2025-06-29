"""
Model deployment component for creating Docker containers
"""

import hashlib
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import docker
import structlog

from ..core.config import settings
from ..models.schema import DeploymentResult

logger = structlog.get_logger()


class Deployer:
    """Handles model deployment to Docker containers"""
    
    def __init__(self):
        try:
            self.docker_client = docker.from_env()
            logger.info("Docker client initialized")
        except Exception as e:
            logger.error("Failed to initialize Docker client", error=str(e))
            self.docker_client = None
    
    async def deploy_model(self, run_id: str) -> DeploymentResult:
        """Deploy a trained model as a Docker container"""
        
        if not self.docker_client:
            raise ValueError("Docker not available")
        
        logger.info("Starting model deployment", run_id=run_id)
        start_time = time.time()
        
        # Find model files
        model_dir = Path(settings.MODELS_DIR) / run_id
        if not model_dir.exists():
            raise ValueError(f"Model directory not found: {model_dir}")
        
        # Find best model (assuming it's the one with highest score in filename or latest)
        model_files = list(model_dir.glob("*_model.pkl"))
        if not model_files:
            raise ValueError("No model files found in model directory")
        
        # Use the first model file (in production, select best based on metadata)
        model_file = model_files[0]
        
        # Create deployment context
        deploy_dir = self._create_deployment_context(run_id, model_file)
        
        # Build Docker image
        docker_tag = self._generate_docker_tag(run_id)
        image = self._build_docker_image(deploy_dir, docker_tag)
        
        # Calculate metrics
        build_time = time.time() - start_time
        image_size_mb = image.attrs['Size'] / (1024 * 1024)
        
        # Generate model hash
        model_hash = self._calculate_model_hash(model_file)
        
        # Generate deployment command
        deployment_command = f"docker run -p 8000:8000 {docker_tag}"
        
        # Create audit record
        self._create_audit_record(run_id, {
            "docker_tag": docker_tag,
            "model_hash": model_hash,
            "build_time_seconds": build_time,
            "image_size_mb": image_size_mb,
            "deployment_command": deployment_command
        })
        
        # Cleanup build context
        shutil.rmtree(deploy_dir, ignore_errors=True)
        
        logger.info("Model deployment completed", 
                   run_id=run_id, 
                   docker_tag=docker_tag,
                   build_time=build_time)
        
        return DeploymentResult(
            run_id=run_id,
            docker_tag=docker_tag,
            image_size_mb=image_size_mb,
            build_time_seconds=build_time,
            model_hash=model_hash,
            deployment_command=deployment_command
        )
    
    def _create_deployment_context(self, run_id: str, model_file: Path) -> Path:
        """Create Docker build context with model and app files"""
        
        deploy_dir = Path(settings.TEMP_DIR) / f"deploy_{run_id}"
        deploy_dir.mkdir(exist_ok=True)
        
        app_dir = deploy_dir / "app"
        app_dir.mkdir(exist_ok=True)
        
        # Copy model file
        shutil.copy2(model_file, app_dir / "model.pkl")
        
        # Create FastAPI app
        self._create_fastapi_app(app_dir)
        
        # Create requirements.txt
        self._create_requirements_file(app_dir)
        
        # Create Dockerfile
        self._create_dockerfile(deploy_dir)
        
        return deploy_dir
    
    def _create_fastapi_app(self, app_dir: Path):
        """Create the FastAPI application for serving the model"""
        
        app_code = '''import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn

# Load model at startup
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI(title="AutoML Model API", version="1.0.0")

class PredictionRequest(BaseModel):
    inputs: List[Dict[str, Any]]

class PredictionResponse(BaseModel):
    predictions: List[float]

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        # Convert inputs to DataFrame
        df = pd.DataFrame(request.inputs)
        
        # Make predictions
        predictions = model.predict(df)
        
        # Convert to list and ensure JSON serializable
        pred_list = [float(pred) for pred in predictions]
        
        return PredictionResponse(predictions=pred_list)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

@app.get("/model/info")
def model_info():
    """Get model information"""
    try:
        model_type = type(model).__name__
        
        # Try to get feature names if available
        feature_names = None
        if hasattr(model, 'feature_names_in_'):
            feature_names = model.feature_names_in_.tolist()
        
        return {
            "model_type": model_type,
            "feature_names": feature_names,
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
        
        with open(app_dir / "app.py", "w") as f:
            f.write(app_code)
    
    def _create_requirements_file(self, app_dir: Path):
        """Create requirements.txt file"""
        
        requirements = [
            "fastapi>=0.104.1",
            "uvicorn>=0.24.0",
            "pandas>=2.1.0",
            "scikit-learn>=1.3.0",
            "lightgbm>=4.1.0",
            "numpy>=1.25.0",
            "pydantic>=2.5.0"
        ]
        
        with open(app_dir / "requirements.txt", "w") as f:
            f.write("\n".join(requirements))
    
    def _create_dockerfile(self, deploy_dir: Path):
        """Create Dockerfile"""
        
        dockerfile_content = '''FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app/ .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
'''
        
        with open(deploy_dir / "Dockerfile", "w") as f:
            f.write(dockerfile_content)
    
    def _generate_docker_tag(self, run_id: str) -> str:
        """Generate Docker tag for the model"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        return f"automl/model_{run_id}:{timestamp}"
    
    def _build_docker_image(self, build_context: Path, tag: str):
        """Build Docker image"""
        
        logger.info("Building Docker image", tag=tag)
        
        try:
            image, build_logs = self.docker_client.images.build(
                path=str(build_context),
                tag=tag,
                rm=True,
                forcerm=True
            )
            
            logger.info("Docker image built successfully", tag=tag, image_id=image.short_id)
            return image
            
        except docker.errors.BuildError as e:
            logger.error("Docker build failed", error=str(e))
            raise ValueError(f"Docker build failed: {str(e)}")
    
    def _calculate_model_hash(self, model_file: Path) -> str:
        """Calculate SHA256 hash of model file"""
        
        with open(model_file, 'rb') as f:
            content = f.read()
            return hashlib.sha256(content).hexdigest()
    
    def _create_audit_record(self, run_id: str, deployment_info: Dict[str, Any]):
        """Create audit record for deployment"""
        
        audit_dir = Path(settings.RUNS_DIR)
        audit_dir.mkdir(exist_ok=True)
        
        audit_file = audit_dir / f"run-{run_id}.json"
        
        audit_data = {
            "run_id": run_id,
            "deployment_timestamp": datetime.utcnow().isoformat(),
            "deployment_info": deployment_info,
            "version": "0.1.0"
        }
        
        with open(audit_file, 'w') as f:
            json.dump(audit_data, f, indent=2)
        
        logger.info("Audit record created", audit_file=str(audit_file)) 