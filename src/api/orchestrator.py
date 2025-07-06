"""
Main orchestrator API endpoints
"""

import asyncio
import hashlib
import json
import os
import uuid
import traceback
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path

import pandas as pd
import numpy as np
from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException
from sqlmodel import Session, select
import structlog

from ..core.config import settings
from ..database import get_session, engine
from ..models.schema import (
    Run, PromptRequest, PromptResponse, DataProfile,
    ModelArtifact, DeploymentResult, EnhancedRun
)
from ..ml.prompt_parser import PromptParser
from ..ml.data_profiler import DataProfiler
from ..ml.enhanced_trainer import EnhancedTrainingOrchestrator
from ..ml.explainer import Explainer
from ..ml.deployer import Deployer
from .websocket_manager import websocket_manager
from ..ml.query_suggester import QuerySuggester

logger = structlog.get_logger()
router = APIRouter()

# Global storage for training tasks
training_tasks = {}


def convert_numpy_types(obj):
    """Convert numpy data types to native Python types for JSON serialization"""
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


def generate_session_id() -> str:
    """Generate a unique session ID"""
    return f"session_{uuid.uuid4().hex[:10]}_{int(datetime.now().timestamp() * 1000)}"


def generate_run_id() -> str:
    """Generate a unique run ID"""
    return str(uuid.uuid4())


async def save_uploaded_file(file: UploadFile, run_id: str) -> str:
    """Save uploaded file and return path"""
    # Create temp directory if it doesn't exist
    settings.TEMP_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save file
    file_path = settings.TEMP_DIR / f"{run_id}_{file.filename}"
    
    content = await file.read()
    with open(file_path, 'wb') as f:
        f.write(content)
    
    return str(file_path)


@router.post("/session/{session_id}/start")
async def start_ml_session(
    session_id: str,
    file: UploadFile = File(...),
    prompt: str = Form(...),
    db: Session = Depends(get_session)
):
    """Start a new ML session with enhanced training (LLM-guided ensemble model selection)"""
    
    logger.info("Starting ML session with enhanced training", session_id=session_id, prompt=prompt)
    
    try:
        # Save uploaded file to a temporary location first
        run_id = str(uuid.uuid4())
        temp_file_path = await save_uploaded_file(file, run_id)
        
        # Always use enhanced training mode
        asyncio.create_task(
            _process_and_train_pipeline(
                run_id=run_id,
                session_id=session_id,
                temp_file_path=temp_file_path,
                prompt=prompt,
                enhanced=True
            )
        )
        
        # Return a response to the client immediately
        return {
            "run_id": run_id,
            "status": "processing_started",
            "message": "File uploaded. Processing and enhanced training will start shortly."
        }
        
    except Exception as e:
        logger.error("Failed to start ML session", error=str(e), exc_info=True)
        # Use traceback to get more detailed error info
        detailed_error = traceback.format_exc()
        await websocket_manager.broadcast_error(
            session_id, f"Failed to initiate session: {str(e)}\n{detailed_error}"
        )
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


async def _process_and_train_pipeline(
    run_id: str,
    session_id: str,
    temp_file_path: str,
    prompt: str,
    enhanced: bool
):
    """Full pipeline: data processing and model training in the background."""
    
    # This function now runs entirely in the background
    with Session(engine) as db:
        try:
            await websocket_manager.broadcast_training_status(
                session_id, {"status": "preprocessing", "message": "Reading and processing dataset..."}
            )

            # --- Start of previously blocking code ---
            with open(temp_file_path, 'rb') as f:
                file_content = f.read()

            file_size_mb = len(file_content) / (1024 * 1024)
            if file_size_mb > settings.MAX_FILE_SIZE_MB:
                raise ValueError(f"File too large. Max size: {settings.MAX_FILE_SIZE_MB}MB")

            # Generate a unique hash of the file contents to identify duplicate datasets
            dataset_hash = hashlib.sha1(file_content).hexdigest()

            import io
            df = pd.read_csv(io.BytesIO(file_content))
            if df.empty:
                raise ValueError("Dataset is empty")
            
            # Create run record - always use EnhancedRun since we only use enhanced training
            run = EnhancedRun(
                id=run_id,
                user_prompt=prompt, 
                dataset_hash=dataset_hash,
                status="parsing_prompt"
            )
            
            # Add the run to the database session
            db.add(run)
            # Commit the transaction to save it
            db.commit()
            # Refresh to get the latest state from the database
            db.refresh(run)

            # Parse prompt
            await websocket_manager.broadcast_training_status(
                session_id, {"status": "parsing_prompt", "message": "Parsing prompt with AI..."}
            )
            prompt_parser = PromptParser()
            
            # Create a preview of the dataset for the prompt parser
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
            # PromptParser is imported from src/ml/prompt_parser.py
            # It uses GPT-4 to parse the user's natural language prompt into structured ML task parameters
            # Returns a PromptResponse object with task type, target column, metric, etc.
            # await is used to pause execution and wait for an asynchronous operation to complete
            # In this case, we wait for prompt_parser.parse_prompt() to finish before continuing
            prompt_response = await prompt_parser.parse_prompt(prompt_request)
            
            if prompt_response.clarifications_needed:
                run.status = "needs_clarification"
                db.commit()
                # Notify frontend about clarification (this part of the UI flow might need adjustment)
                await websocket_manager.broadcast_to_session(session_id, {
                    "type": "clarification_needed",
                    "data": {
                        "run_id": run.id,
                        "clarifications": prompt_response.clarifications_needed,
                        "parsed_intent": prompt_response.dict()
                    }
                })
                return # Stop processing until user clarifies

            # Profile the data
            await websocket_manager.broadcast_training_status(
                session_id, {"status": "profiling_data", "message": "Analyzing data profile..."}
            )
            profiler = DataProfiler()
            profile = profiler.profile_dataset(df)

            run.status = "training"
            run.metric = prompt_response.metric
            db.commit()
            # --- End of previously blocking code ---

            # Always use enhanced training pipeline
            await _run_enhanced_training_pipeline(
                run.id, session_id, temp_file_path, 
                prompt_response, profile, db
            )

        except Exception as e:
            logger.error("Background processing/training failed", run_id=run_id, error=str(e), exc_info=True)
            detailed_error = traceback.format_exc()
            
            try:
                # Update run status to failed
                run = db.get(Run, run_id)
                if run:
                    run.status = "failed"
                    run.end_ts = datetime.utcnow()
                    db.commit()
            except Exception as db_error:
                logger.error("Failed to update run status to failed", run_id=run_id, error=str(db_error))

            await websocket_manager.broadcast_error(
                session_id, f"Processing failed: {str(e)}\n\n{detailed_error}"
            )
        finally:
            # Cleanup temp file
            try:
                os.remove(temp_file_path)
            except OSError:
                pass





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



async def _run_enhanced_training_pipeline(
    run_id: str,
    session_id: str,
    dataset_path: str,
    prompt_response: PromptResponse,
    profile: DataProfile,
    db: Session
):
    """Run the enhanced training pipeline with LLM-guided model selection"""
    
    try:
        # Initialize enhanced training orchestrator
        trainer = EnhancedTrainingOrchestrator(
            run_id=run_id,
            session_id=session_id,
            websocket_manager=websocket_manager
        )
        
        # Run enhanced ensemble training
        best_ensemble = await trainer.train_ensemble_models(
            dataset_path=dataset_path,
            target_col=prompt_response.target,
            task_type=prompt_response.task,
            metric=prompt_response.metric,
            constraints=prompt_response.constraints,
            data_profile=profile
        )
        
        # Generate explanations for ensemble
        explainer = Explainer()
        explanation_result = await explainer.explain_model(
            model_path=best_ensemble.model_path,
            dataset_path=dataset_path,
            target_col=prompt_response.target
        )
        
        # Update run status with fresh database session
        with Session(engine) as fresh_db:
            run = fresh_db.get(EnhancedRun, run_id)
            if run:
                run.status = "completed"
                run.end_ts = datetime.utcnow()
                run.best_family = best_ensemble.family
                run.val_score = best_ensemble.val_score
                fresh_db.add(run)
                fresh_db.commit()
        
        # Notify frontend
        await websocket_manager.broadcast_training_complete(
            session_id,
            {
                "run_id": run_id,
                "best_ensemble": best_ensemble.dict(),
                "explanation": explanation_result,
                "enhanced_results": {
                    "ensemble_method": best_ensemble.family,
                    "models_tested": "Multiple architectures per model type",
                    "selection_method": "LLM-guided with neural architecture search"
                }
            }
        )
        
        # Cleanup temp files
        try:
            os.remove(dataset_path)
        except:
            pass
            
    except Exception as e:
        logger.error("Enhanced training pipeline failed", run_id=run_id, error=str(e))
        
        # Update run status with fresh database session
        try:
            with Session(engine) as fresh_db:
                run = fresh_db.get(EnhancedRun, run_id)
                if run:
                    run.status = "failed"
                    run.end_ts = datetime.utcnow()
                    fresh_db.add(run)
                    fresh_db.commit()
        except Exception as db_error:
            logger.error("Failed to update run status", error=str(db_error))
        
        # Notify frontend
        await websocket_manager.broadcast_error(
            session_id, f"Enhanced training failed: {str(e)}"
        )


@router.post("/get-model-recommendations")
async def get_model_recommendations(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    enhanced: str = Form(...),
    session_id: str = Form(None),
    db: Session = Depends(get_session)
):
    """Get AI model recommendations for user selection"""
    try:
        logger.info("Getting model recommendations for user selection", 
                   file_name=file.filename,
                   enhanced=enhanced)
        
        # Use provided session_id or generate new one
        if not session_id:
            session_id = generate_session_id()
        run_id = generate_run_id()
        
        # Read file content for hashing and processing
        file_content = await file.read()
        
        # Save file content directly (since file stream is consumed)
        settings.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        file_path = settings.TEMP_DIR / f"{run_id}_{file.filename}"
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
        # Load and profile data
        df = pd.read_csv(file_path)
        profiler = DataProfiler()
        data_profile = profiler.profile_dataset(df)
        
        # Parse intent to get basic task info
        parser = PromptParser()
        
        # Create dataset preview
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
        
        parsed_intent = await parser.parse_prompt(prompt_request)
        
        if parsed_intent.clarifications_needed:
            return {
                "status": "needs_clarification",
                "run_id": run_id,
                "clarifications": parsed_intent.clarifications_needed
            }
        
        # Handle clustering tasks
        if parsed_intent.task == "clustering":
            from ..ml.clustering_trainer import ClusteringTrainer
            
            # Send initial response
            response_data = {
                "status": "clustering_started", 
                "run_id": run_id,
                "session_id": session_id,
                "parsed_intent": {
                    "task": parsed_intent.task,
                    "target": parsed_intent.target,
                    "metric": parsed_intent.metric
                },
                "data_profile": {
                    "n_rows": data_profile.n_rows,
                    "n_cols": data_profile.n_cols,
                    "issues": data_profile.issues
                }
            }
            
            # Start clustering in background
            logger.info("Starting clustering pipeline", run_id=run_id, session_id=session_id)
            asyncio.create_task(
                _run_clustering_pipeline(
                    run_id, session_id, file_path,
                    parsed_intent.constraints or {},
                    websocket_manager
                )
            )
            
            return response_data
        
        # Get LLM model recommendations
        if enhanced.lower() == 'true':
            from ..ml.model_selector import ModelSelector
            model_selector = ModelSelector()
            
            ensemble_strategy = await model_selector.recommend_ensemble(
                data_profile=data_profile,
                task_type=parsed_intent.task,
                target_column=parsed_intent.target,
                constraints=parsed_intent.constraints
            )
            
            # Format recommendations for frontend
            recommendations = []
            model_type_counters = {}  # Track count per model type for unique IDs
            
            for model_rec in ensemble_strategy.recommended_models:
                # Generate deterministic ID based on model type, not array position
                model_type = model_rec.model_type
                if model_type not in model_type_counters:
                    model_type_counters[model_type] = 0
                else:
                    model_type_counters[model_type] += 1
                
                model_id = f"{model_type}_{model_type_counters[model_type]}"
                
                recommendations.append({
                    "id": model_id,
                    "model_type": model_rec.model_type,
                    "model_family": model_rec.model_family,
                    "name": model_rec.model_type.replace("_", " ").title(),
                    "complexity_score": model_rec.complexity_score,
                    "expected_training_time": model_rec.expected_training_time,
                    "training_time_estimate": model_rec.training_time_estimate,
                    "memory_usage": model_rec.memory_usage,
                    "interpretability": model_rec.interpretability,
                    "reasoning": model_rec.reasoning,
                    "pros": model_rec.pros or [],
                    "cons": model_rec.cons or [],
                    "best_for": model_rec.best_for,
                    "architectures": len(model_rec.architectures),
                    "selected": True  # Default to selected
                })
            
            return {
                "status": "recommendations_ready",
                "run_id": run_id,
                "session_id": session_id,
                "parsed_intent": {
                    "task": parsed_intent.task,
                    "target": parsed_intent.target,
                    "metric": parsed_intent.metric
                },
                "data_profile": {
                    "n_rows": data_profile.n_rows,
                    "n_cols": data_profile.n_cols,
                    "issues": data_profile.issues
                },
                "recommendations": recommendations,
                "ensemble_strategy": {
                    "method": ensemble_strategy.ensemble_method,
                    "reasoning": ensemble_strategy.reasoning,
                    "diversity_score": ensemble_strategy.diversity_score
                },
                "estimated_total_time": "15-45 minutes",
                "file_path": str(file_path)
            }
        else:
            # Always use enhanced training - no longer support standard mode
            from ..ml.model_selector import ModelSelector
            model_selector = ModelSelector()
            
            ensemble_strategy = await model_selector.recommend_ensemble(
                data_profile=data_profile,
                task_type=parsed_intent.task,
                target_column=parsed_intent.target,
                constraints=parsed_intent.constraints
            )
            
            # Format recommendations for frontend
            recommendations = []
            model_type_counters = {}  # Track count per model type for unique IDs
            
            for model_rec in ensemble_strategy.recommended_models:
                # Generate deterministic ID based on model type, not array position
                model_type = model_rec.model_type
                if model_type not in model_type_counters:
                    model_type_counters[model_type] = 0
                else:
                    model_type_counters[model_type] += 1
                
                model_id = f"{model_type}_{model_type_counters[model_type]}"
                
                recommendations.append({
                    "id": model_id,
                    "model_type": model_rec.model_type,
                    "model_family": model_rec.model_family,
                    "name": model_rec.model_type.replace("_", " ").title(),
                    "complexity_score": model_rec.complexity_score,
                    "expected_training_time": model_rec.expected_training_time,
                    "training_time_estimate": model_rec.training_time_estimate,
                    "memory_usage": model_rec.memory_usage,
                    "interpretability": model_rec.interpretability,
                    "reasoning": model_rec.reasoning,
                    "pros": model_rec.pros or [],
                    "cons": model_rec.cons or [],
                    "best_for": model_rec.best_for,
                    "architectures": len(model_rec.architectures),
                    "selected": True  # Default to selected
                })
            
            return {
                "status": "recommendations_ready",
                "run_id": run_id,
                "session_id": session_id,
                "parsed_intent": {
                    "task": parsed_intent.task,
                    "target": parsed_intent.target,
                    "metric": parsed_intent.metric
                },
                "data_profile": {
                    "n_rows": data_profile.n_rows,
                    "n_cols": data_profile.n_cols,
                    "issues": data_profile.issues
                },
                "recommendations": recommendations,
                "ensemble_strategy": {
                    "method": ensemble_strategy.ensemble_method,
                    "reasoning": ensemble_strategy.reasoning,
                    "diversity_score": ensemble_strategy.diversity_score
                },
                "estimated_total_time": "15-45 minutes",
                "file_path": str(file_path)
            }
            
    except Exception as e:
        logger.error("Model recommendations failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Model recommendations failed: {str(e)}")


@router.post("/start-training-with-selection")
async def start_training_with_selection(request: dict):
    """Start training with user-selected models"""
    try:
        run_id = request["run_id"]
        session_id = request["session_id"]
        selected_models = request["selected_models"]  # List of selected model IDs
        file_path = request["file_path"]
        parsed_intent = request["parsed_intent"]
        
        logger.info("Starting training with user model selection", 
                   run_id=run_id,
                   selected_models=len(selected_models))
        
        # Load data
        df = pd.read_csv(file_path)
        profiler = DataProfiler()
        data_profile = profiler.profile_dataset(df)
        
        # Start enhanced training in background with selected models
        websocket_manager_instance = websocket_manager
        
        # Initialize constraints
        constraints = {}
        
        task = asyncio.create_task(
            _run_enhanced_training_with_selection(
                run_id, session_id, file_path, parsed_intent["target"],
                parsed_intent["task"], parsed_intent["metric"], 
                constraints, data_profile, selected_models, websocket_manager_instance
            )
        )
        
        # Store task for potential cancellation
        training_tasks[run_id] = task
        
        return {
            "status": "training_started",
            "run_id": run_id,
            "message": f"Training started with {len(selected_models)} selected models"
        }
        
    except Exception as e:
        logger.error("Training with selection failed", error=str(e), traceback=traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


async def _run_clustering_pipeline(
    run_id: str,
    session_id: str,
    dataset_path: str,
    constraints: Dict[str, Any],
    websocket_manager_instance
):
    """Run clustering pipeline"""
    try:
        logger.info("🔍 Starting clustering pipeline", 
                   run_id=run_id, 
                   session_id=session_id)
        
        from ..ml.clustering_trainer import ClusteringTrainer
        trainer = ClusteringTrainer(run_id, session_id, websocket_manager_instance)
        
        # Default clustering algorithms if none specified
        selected_algorithms = constraints.get("selected_algorithms", ["kmeans", "dbscan", "hierarchical"])
        
        result = await trainer.train_clustering_models(
            dataset_path=dataset_path,
            constraints=constraints,
            selected_algorithms=selected_algorithms
        )
        
        logger.info("✅ Clustering completed successfully", 
                   run_id=run_id,
                   algorithm=result.algorithm,
                   silhouette_score=f"{result.silhouette_score:.4f}",
                   n_clusters=result.n_clusters)
        
    except Exception as e:
        logger.error("❌ Clustering pipeline failed", 
                    run_id=run_id, 
                    session_id=session_id,
                    error=str(e))
        await websocket_manager_instance.broadcast_error(session_id, f"Clustering failed: {str(e)}")


async def _run_enhanced_training_with_selection(
    run_id: str,
    session_id: str, 
    dataset_path: str,
    target_col: str,
    task_type: str,
    metric: str,
    constraints: Dict[str, Any],
    data_profile: DataProfile,
    selected_models: List[str],
    websocket_manager_instance
):
    """Run enhanced training with user-selected models"""
    try:
        logger.info("🚀 Starting enhanced training with selection", 
                   run_id=run_id, 
                   session_id=session_id,
                   selected_models=selected_models,
                   target_col=target_col,
                   task_type=task_type,
                   metric=metric,
                   data_profile_rows=data_profile.n_rows,
                   data_profile_cols=data_profile.n_cols)
        
        # Route to appropriate trainer based on task type
        if task_type == "forecasting":
            # Use time series trainer for forecasting tasks
            from ..ml.timeseries_trainer import TimeSeriesTrainer
            trainer = TimeSeriesTrainer(run_id, session_id, websocket_manager_instance)
        else:
            # Use enhanced trainer for classification/regression tasks
            from ..ml.enhanced_trainer import EnhancedTrainingOrchestrator
            trainer = EnhancedTrainingOrchestrator(run_id, session_id, websocket_manager_instance)
        
        # Convert frontend model IDs to backend model types
        model_type_mapping = {
            # Classification/Regression models
            "linear_regression": "linear_regression",
            "logistic_regression": "logistic_regression", 
            "random_forest": "random_forest",
            "xgboost": "xgboost",
            "lightgbm": "lightgbm",
            "catboost": "catboost",
            "hist_gradient_boosting": "hist_gradient_boosting",
            "neural_network": "neural_network",
            "svm": "svm",
            "naive_bayes": "naive_bayes",
            "knn": "knn",
            "extra_trees": "extra_trees",
            
            # Time Series Forecasting models
            "arima": "arima",
            "prophet": "prophet",
            "exponential_smoothing": "exponential_smoothing",
            "lstm_ts": "lstm_ts",
            "seasonal_decompose": "seasonal_decompose",
            
            # Clustering models
            "kmeans": "kmeans",
            "dbscan": "dbscan",
            "hierarchical": "hierarchical",
            "spectral": "spectral",
            "gaussian_mixture": "gaussian_mixture",
            
            # Anomaly detection
            "isolation_forest": "isolation_forest"
        }
        
        # Extract model types from selected model IDs
        selected_model_types = []
        for model_id in selected_models:
            # Extract base model type from ID (remove numeric suffixes like "_0", "_1", etc.)
            # Handle cases like "arima_1", "prophet_0", "random_forest_2", "exponential_smoothing_1"
            
            # First, try exact match
            if model_id in model_type_mapping:
                selected_model_types.append(model_type_mapping[model_id])
                continue
            
            # Remove trailing numbers to get base type (e.g., "arima_1" -> "arima")
            import re
            base_type = re.sub(r'_\d+$', '', model_id)
            
            if base_type in model_type_mapping:
                selected_model_types.append(model_type_mapping[base_type])
            else:
                # For compound names like "exponential_smoothing_1", try the full compound name
                # Split and check if it's a multi-word model type
                parts = model_id.split('_')
                if len(parts) > 2:
                    # Try combinations like "exponential_smoothing" from "exponential_smoothing_1"
                    for i in range(len(parts)-1, 0, -1):
                        compound_type = '_'.join(parts[:i])
                        if compound_type in model_type_mapping:
                            selected_model_types.append(model_type_mapping[compound_type])
                            break
                else:
                    # Fallback: try partial matching
                    for frontend_name, backend_name in model_type_mapping.items():
                        if frontend_name in model_id.lower():
                            selected_model_types.append(backend_name)
                            break
        
        logger.info("🔍 Mapped selected models", 
                   frontend_ids=selected_models, 
                   backend_types=selected_model_types,
                   neural_networks_selected=any("neural_network" in t for t in selected_model_types))
        
        # Add selected models to constraints
        enhanced_constraints = constraints.copy()
        enhanced_constraints["selected_models"] = selected_model_types
        
        logger.info("🎯 Starting enhanced trainer with constraints", 
                   constraints=enhanced_constraints)
        
        # Send initial progress update
        await websocket_manager_instance.broadcast_training_status(session_id, {
            "event": "training_started",
            "message": f"Starting training with {len(selected_model_types)} model types",
            "selected_models": selected_model_types,
            "progress": 0
        })
        
        # Call appropriate training method based on task type
        if task_type == "forecasting":
            # For forecasting, we need to detect the date column
            df = pd.read_csv(dataset_path)
            date_columns = []
            for col in df.columns:
                if col.lower() in ['date', 'time', 'timestamp', 'datetime'] or df[col].dtype == 'datetime64[ns]':
                    date_columns.append(col)
            
            if not date_columns:
                # Try to detect date columns by attempting to parse them
                for col in df.columns:
                    if col != target_col and df[col].dtype == 'object':
                        try:
                            pd.to_datetime(df[col].head(10))
                            date_columns.append(col)
                            break
                        except:
                            continue
            
            if not date_columns:
                raise ValueError("No date column found for time series forecasting")
            
            date_column = date_columns[0]
            forecast_horizon = enhanced_constraints.get("forecast_horizon", 30)
            
            # Add selected models to constraints for time series trainer
            enhanced_constraints["selected_models"] = selected_model_types
            
            result = await trainer.train_forecast_models(
                dataset_path=dataset_path,
                date_column=date_column,
                target_column=target_col,
                forecast_horizon=forecast_horizon,
                data_profile=data_profile,
                constraints=enhanced_constraints
            )
        else:
            result = await trainer.train_ensemble_models(
                dataset_path=dataset_path,
                target_col=target_col,
                task_type=task_type,
                metric=metric,
                constraints=enhanced_constraints,
                data_profile=data_profile
            )
        
        if task_type == "forecasting":
            # Time series trainer returns ModelArtifact with different format
            logger.info("✅ Time series training completed successfully with user selection", 
                       run_id=run_id,
                       best_model_type=result.family,
                       performance_rmse=result.val_score)
            
            # Get forecast data from the trainer
            forecast_data = None
            if hasattr(trainer, 'forecast_data'):
                forecast_data = trainer.forecast_data
            
            # Get all model performances from the trainer
            all_model_performances = []
            if hasattr(trainer, 'all_model_performances'):
                all_model_performances = trainer.all_model_performances
            
            # Send completion notification for time series
            completion_data = {
                "forecasting_results": True,  # Mark as forecasting
                "run_id": run_id,
                "best_model": {
                    "family": result.family,
                    "model_type": result.family,
                    "val_score": result.val_score,
                    "train_score": result.train_score
                },
                "model_path": result.model_path,
                "selected_models": selected_model_types,
                "training_method": "time_series_forecasting",
                "all_models": all_model_performances  # Include all model performances
            }
            
            # Add forecast data if available
            if forecast_data:
                completion_data["forecast_data"] = forecast_data
            
            await websocket_manager_instance.broadcast_training_complete(session_id, completion_data)
        else:
            # Enhanced trainer returns ensemble results
            logger.info("✅ Enhanced training completed successfully with user selection", 
                       run_id=run_id,
                       final_score=f"{result.val_score:.4f}",
                       best_model_family=result.family,
                       train_score=f"{result.train_score:.4f}")
            
            # Send completion notification
            await websocket_manager_instance.broadcast_training_complete(session_id, {
                "enhanced_results": True,  # Mark as enhanced since we used enhanced trainer
                "run_id": run_id,
                "best_model": {
                    "family": result.family,
                    "val_score": result.val_score,
                    "train_score": result.train_score
                },
                "model_path": result.model_path,
                "selected_models": selected_model_types,
                "ensemble_method": "user_selected"
            })
        
        logger.info("🎉 Training completion notification sent", 
                   session_id=session_id, 
                   run_id=run_id)
        
    except Exception as e:
        logger.error("❌ Enhanced training with selection failed", 
                    run_id=run_id, 
                    session_id=session_id,
                    error=str(e), 
                    error_type=type(e).__name__,
                    traceback=traceback.format_exc())
        await websocket_manager_instance.broadcast_error(session_id, f"Enhanced training failed: {str(e)}")
    finally:
        # Clean up
        if run_id in training_tasks:
            del training_tasks[run_id]
            logger.info("🧹 Cleaned up training task", run_id=run_id) 


@router.post("/generate-query-suggestions")
async def generate_query_suggestions(
    file: UploadFile = File(...),
    max_suggestions: int = 5
):
    """Generate intelligent query suggestions based on uploaded dataset"""
    try:
        logger.info("Generating query suggestions", 
                   file_name=file.filename,
                   max_suggestions=max_suggestions)
        
        # Read and save file temporarily
        file_content = await file.read()
        settings.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        temp_file_path = settings.TEMP_DIR / f"temp_suggestions_{file.filename}"
        
        with open(temp_file_path, 'wb') as f:
            f.write(file_content)
        
        # Load dataset
        try:
            df = pd.read_csv(temp_file_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not read CSV file: {str(e)}")
        
        # Validate dataset
        if df.empty:
            raise HTTPException(status_code=400, detail="Dataset is empty")
        
        if len(df.columns) < 2:
            raise HTTPException(status_code=400, detail="Dataset must have at least 2 columns")
        
        # Generate suggestions
        suggester = QuerySuggester()
        suggestions = await suggester.generate_suggestions(df, max_suggestions)
        
        # Get basic dataset info
        dataset_info = {
            "shape": {"rows": int(len(df)), "columns": int(len(df.columns))},
            "columns": df.columns.tolist(),
            "column_types": {str(k): str(v) for k, v in df.dtypes.astype(str).to_dict().items()},
            "sample_data": df.head(3).to_dict(orient="records"),
            "has_nulls": bool(df.isnull().any().any()),
            "memory_usage": int(df.memory_usage(deep=True).sum())
        }
        
        # Cleanup temp file
        try:
            os.remove(temp_file_path)
        except:
            pass
        
        logger.info(f"Generated {len(suggestions)} query suggestions successfully")
        
        response = {
            "status": "success",
            "suggestions": suggestions,
            "dataset_info": dataset_info,
            "message": f"Generated {len(suggestions)} intelligent query suggestions"
        }
        
        return convert_numpy_types(response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Query suggestion generation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to generate suggestions: {str(e)}")


@router.post("/analyze-dataset")
async def analyze_dataset(
    file: UploadFile = File(...),
    include_suggestions: bool = True
):
    """Analyze dataset and optionally include query suggestions"""
    try:
        logger.info("Analyzing dataset", 
                   file_name=file.filename,
                   include_suggestions=include_suggestions)
        
        # Read and save file temporarily
        file_content = await file.read()
        settings.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        temp_file_path = settings.TEMP_DIR / f"temp_analysis_{file.filename}"
        
        with open(temp_file_path, 'wb') as f:
            f.write(file_content)
        
        # Load and profile dataset
        df = pd.read_csv(temp_file_path)
        profiler = DataProfiler()
        data_profile = profiler.profile_dataset(df)
        
        response = {
            "status": "success",
            "dataset_info": {
                "filename": file.filename,
                "shape": {"rows": int(len(df)), "columns": int(len(df.columns))},
                "columns": df.columns.tolist(),
                "sample_data": df.head(5).to_dict(orient="records"),
                "column_types": {str(k): str(v) for k, v in df.dtypes.astype(str).to_dict().items()}
            },
            "data_profile": {
                "n_rows": int(data_profile.n_rows),
                "n_cols": int(data_profile.n_cols),
                "issues": data_profile.issues
            }
        }
        
        # Generate suggestions if requested
        if include_suggestions:
            suggester = QuerySuggester()
            suggestions = await suggester.generate_suggestions(df, max_suggestions=5)
            response["suggestions"] = suggestions
            response["suggestions_count"] = len(suggestions)
        
        # Cleanup temp file
        try:
            os.remove(temp_file_path)
        except:
            pass
        
        logger.info("Dataset analysis completed successfully")
        return convert_numpy_types(response)
        
    except Exception as e:
        logger.error("Dataset analysis failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Dataset analysis failed: {str(e)}") 