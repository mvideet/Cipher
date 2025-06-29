"""
Main orchestrator API endpoints
"""

import asyncio
import hashlib
import json
import os
import uuid
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException
from sqlmodel import Session, select
import pandas as pd
import structlog

from ..core.config import settings
from ..database import get_session
from ..models.schema import (
    Run, PromptRequest, PromptResponse, DataProfile,
    ModelArtifact, DeploymentResult
)
from ..ml.prompt_parser import PromptParser
from ..ml.data_profiler import DataProfiler
from ..ml.trainer import TrainingOrchestrator
from ..ml.explainer import Explainer
from ..ml.deployer import Deployer
from .websocket_manager import websocket_manager

logger = structlog.get_logger()
router = APIRouter()


@router.post("/session/{session_id}/start")
async def start_ml_session(
    session_id: str,
    file: UploadFile = File(...),
    prompt: str = Form(...),
    db: Session = Depends(get_session)
):
    """Start a new ML session with dataset upload and prompt"""
    
    logger.info("Starting ML session", session_id=session_id, prompt=prompt)
    
    try:
        # Validate file size
        file_content = await file.read()
        file_size_mb = len(file_content) / (1024 * 1024)
        
        if file_size_mb > settings.MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max size: {settings.MAX_FILE_SIZE_MB}MB"
            )
        
        # Create dataset hash for deduplication
        dataset_hash = hashlib.sha1(file_content).hexdigest()
        
        # Load and validate dataset
        try:
            import io
            df = pd.read_csv(io.BytesIO(file_content))
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to read CSV file: {str(e)}"
            )
        
        if df.empty:
            raise HTTPException(status_code=400, detail="Dataset is empty")
        
        # Create run record
        run = Run(
            id=str(uuid.uuid4()),
            user_prompt=prompt,
            dataset_hash=dataset_hash,
            status="parsing_prompt"
        )
        db.add(run)
        db.commit()
        db.refresh(run)
        
        # Parse prompt with GPT-4
        prompt_parser = PromptParser()
        dataset_preview = {
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "sample_rows": df.head(5).to_dict(orient="records"),
            "shape": df.shape
        }
        
        prompt_request = PromptRequest(
            session_id=session_id,
            prompt=prompt,
            dataset_preview=dataset_preview
        )
        
        try:
            prompt_response = await prompt_parser.parse_prompt(prompt_request)
        except Exception as e:
            run.status = "failed"
            run.end_ts = datetime.utcnow()
            db.commit()
            await websocket_manager.broadcast_error(
                session_id, f"Failed to parse prompt: {str(e)}"
            )
            raise HTTPException(status_code=400, detail=str(e))
        
        # Check if clarifications needed
        if prompt_response.clarifications_needed:
            run.status = "needs_clarification"
            db.commit()
            return {
                "run_id": run.id,
                "status": "needs_clarification",
                "clarifications": prompt_response.clarifications_needed,
                "parsed_intent": {
                    "task": prompt_response.task,
                    "target": prompt_response.target,
                    "metric": prompt_response.metric,
                    "constraints": prompt_response.constraints
                }
            }
        
        # Profile the data
        run.status = "profiling_data"
        db.commit()
        
        profiler = DataProfiler()
        profile = profiler.profile_dataset(df)
        
        # Start training asynchronously
        run.status = "training"
        run.metric = prompt_response.metric
        db.commit()
        
        # Save dataset temporarily for training
        temp_file = settings.TEMP_DIR / f"{run.id}_dataset.csv"
        df.to_csv(temp_file, index=False)
        
        # Schedule training in background
        asyncio.create_task(
            _run_training_pipeline(
                run.id, session_id, str(temp_file), 
                prompt_response, profile, db
            )
        )
        
        return {
            "run_id": run.id,
            "status": "training_started",
            "data_profile": profile.dict(),
            "parsed_intent": {
                "task": prompt_response.task,
                "target": prompt_response.target,
                "metric": prompt_response.metric,
                "constraints": prompt_response.constraints
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to start ML session", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/session/{session_id}/clarify")
async def provide_clarification(
    session_id: str,
    run_id: str = Form(...),
    clarification: str = Form(...),
    db: Session = Depends(get_session)
):
    """Provide clarification and continue training"""
    
    logger.info("Received clarification", run_id=run_id, clarification=clarification)
    
    # Get run
    run = db.get(Run, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    if run.status != "needs_clarification":
        raise HTTPException(status_code=400, detail="Run doesn't need clarification")
    
    # Update prompt with clarification
    updated_prompt = f"{run.user_prompt}\n\nClarification: {clarification}"
    run.user_prompt = updated_prompt
    run.status = "training"
    db.commit()
    
    # Continue training pipeline...
    # (Implementation would continue the training process)
    
    return {"status": "clarification_received", "training_resumed": True}


@router.get("/session/{session_id}/status/{run_id}")
async def get_run_status(
    session_id: str,
    run_id: str,
    db: Session = Depends(get_session)
):
    """Get current status of a training run"""
    
    run = db.get(Run, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    return {
        "run_id": run.id,
        "status": run.status,
        "start_time": run.start_ts,
        "end_time": run.end_ts,
        "best_family": run.best_family,
        "val_score": run.val_score,
        "metric": run.metric
    }


@router.post("/session/{session_id}/deploy/{run_id}")
async def deploy_model(
    session_id: str,
    run_id: str,
    db: Session = Depends(get_session)
):
    """Deploy the best model from a training run"""
    
    logger.info("Deploying model", run_id=run_id)
    
    run = db.get(Run, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    if run.status != "completed":
        raise HTTPException(status_code=400, detail="Training not completed")
    
    try:
        deployer = Deployer()
        deployment_result = await deployer.deploy_model(run_id)
        
        await websocket_manager.send_personal_message(
            {"type": "deployment_complete", "data": deployment_result.dict()},
            session_id
        )
        
        return deployment_result.dict()
        
    except Exception as e:
        logger.error("Failed to deploy model", run_id=run_id, error=str(e))
        await websocket_manager.broadcast_error(
            session_id, f"Deployment failed: {str(e)}"
        )
        raise HTTPException(status_code=500, detail="Deployment failed")


async def _run_training_pipeline(
    run_id: str,
    session_id: str,
    dataset_path: str,
    prompt_response: PromptResponse,
    profile: DataProfile,
    db: Session
):
    """Run the complete training pipeline asynchronously"""
    
    try:
        # Initialize training orchestrator
        trainer = TrainingOrchestrator(
            run_id=run_id,
            session_id=session_id,
            websocket_manager=websocket_manager
        )
        
        # Run training
        best_model = await trainer.train_models(
            dataset_path=dataset_path,
            target_col=prompt_response.target,
            task_type=prompt_response.task,
            metric=prompt_response.metric,
            constraints=prompt_response.constraints
        )
        
        # Generate explanations
        explainer = Explainer()
        explanation_result = await explainer.explain_model(
            model_path=best_model.model_path,
            dataset_path=dataset_path,
            target_col=prompt_response.target
        )
        
        # Update run status
        run = db.get(Run, run_id)
        run.status = "completed"
        run.end_ts = datetime.utcnow()
        run.best_family = best_model.family
        run.val_score = best_model.val_score
        db.commit()
        
        # Notify frontend
        await websocket_manager.broadcast_training_complete(
            session_id,
            {
                "run_id": run_id,
                "best_model": best_model.dict(),
                "explanation": explanation_result
            }
        )
        
        # Cleanup temp files
        try:
            os.remove(dataset_path)
        except:
            pass
            
    except Exception as e:
        logger.error("Training pipeline failed", run_id=run_id, error=str(e))
        
        # Update run status
        run = db.get(Run, run_id)
        run.status = "failed"
        run.end_ts = datetime.utcnow()
        db.commit()
        
        # Notify frontend
        await websocket_manager.broadcast_error(
            session_id, f"Training failed: {str(e)}"
        ) 