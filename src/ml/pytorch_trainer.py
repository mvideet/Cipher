"""
PyTorch neural network trainer with NAS integration
"""

import asyncio
import pickle
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor  # Change back to threads but with better isolation
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import structlog

from ..core.config import settings
from ..models.schema import ModelArtifact, DataProfile
from .pytorch_models import PyTorchModelFactory, BaseNeuralNet
from .pytorch_nas import PyTorchNAS

logger = structlog.get_logger()


class PyTorchTrainer:
    """PyTorch trainer with Neural Architecture Search using improved isolation"""
    
    def __init__(self, run_id: str, session_id: str, websocket_manager):
        self.run_id = run_id
        self.session_id = session_id
        self.websocket_manager = websocket_manager
        self.start_time = time.time()
        self.max_time_seconds = settings.MAX_TRAINING_TIME_MINUTES * 60
        
        # PyTorch-specific settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    async def train_pytorch_models(
        self,
        dataset_path: str,
        target_col: str,
        task_type: str,
        metric: str,
        constraints: Dict[str, Any],
        data_profile: DataProfile,
        search_strategy: str = "progressive"
    ) -> List[ModelArtifact]:
        """Train PyTorch models using Neural Architecture Search with improved threading"""
        
        # Store task type for detailed reporting
        self._task_type = task_type
        self._metric = metric
        
        logger.info("🧠 Starting PyTorch neural network training with improved isolation", 
                   run_id=self.run_id, 
                   task=task_type, 
                   metric=metric,
                   search_strategy=search_strategy)
        
        # Load and prepare data
        df = pd.read_csv(dataset_path)
        X_processed, y_processed = self._prepare_pytorch_data(df, target_col, constraints)
        
        # Split data
        test_size = min(0.2, max(0.1, 10 / len(y_processed)))
        stratify_param = y_processed if task_type == "classification" and len(np.unique(y_processed)) > 1 else None
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_processed, y_processed, 
            test_size=test_size, 
            random_state=42, 
            stratify=stratify_param
        )
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_train_tensor = torch.LongTensor(y_train) if task_type == "classification" else torch.FloatTensor(y_train)
        y_val_tensor = torch.LongTensor(y_val) if task_type == "classification" else torch.FloatTensor(y_val)
        y_train_tensor = y_train_tensor.to(self.device)
        y_val_tensor = y_val_tensor.to(self.device)
        
        logger.info("📊 PyTorch data prepared", 
                   train_size=len(X_train),
                   val_size=len(X_val),
                   features=X_train.shape[1],
                   device=str(self.device))
        
        # Initialize NAS engine
        nas_engine = PyTorchNAS(
            task_type=task_type,
            metric=metric,
            max_time_minutes=settings.MAX_TRAINING_TIME_MINUTES // 4  # Shorter time for stability
        )
        
        # Search architectures for different model types using thread pool
        architecture_types = ["simple_mlp"]  # Start with just one to avoid complexity
        trained_models = []
        
        # Use thread pool with limited concurrency
        loop = asyncio.get_running_loop()
        
        for arch_type in architecture_types:
            try:
                logger.info(f"🔍 Running NAS for {arch_type} in thread...")
                
                # Run architecture search in a thread with timeout
                nas_result = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,  # Use default thread pool
                        self._run_nas_sync,
                        nas_engine, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, arch_type, search_strategy
                    ),
                    timeout=300  # 5 minute timeout per architecture
                )
                
                if "error" in nas_result:
                    logger.error(f"NAS failed for {arch_type}: {nas_result['error']}")
                    continue
                
                # Train final model with best architecture
                best_config = nas_result["best_config"]
                model_artifact = await self._train_final_model(
                    best_config, X_train_tensor, X_val_tensor, 
                    y_train_tensor, y_val_tensor, task_type, metric
                )
                
                if model_artifact:
                    trained_models.append(model_artifact)
                    
                    await self._send_model_completion_update(
                        f"pytorch_{arch_type}", 
                        model_artifact.val_score,
                        nas_result["search_strategy"]
                    )
                
                # Check time budget
                if time.time() - self.start_time > self.max_time_seconds * 0.6:
                    logger.info("⏰ Time budget running low, stopping architecture search")
                    break
                    
            except asyncio.TimeoutError:
                logger.error(f"Timeout during {arch_type} architecture search")
                continue
            except Exception as e:
                logger.error(f"Failed to train {arch_type}", error=str(e))
                continue
        
        if not trained_models:
            # If PyTorch fails, return an empty list rather than crashing
            logger.warning("All PyTorch model training failed, returning empty list")
            return []
        
        logger.info("🎯 PyTorch training completed", 
                   n_models=len(trained_models),
                   best_score=f"{max(m.val_score for m in trained_models):.4f}")
        
        return trained_models
    
    def _run_nas_sync(self, nas_engine, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, 
                     architecture_type, search_strategy):
        """Synchronous wrapper for NAS that can be run in a thread"""
        try:
            return nas_engine.search_architecture(
                X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor,
                architecture_type=architecture_type,
                search_strategy=search_strategy
            )
        except Exception as e:
            logger.error(f"NAS error for {architecture_type}: {str(e)}")
            return {"error": str(e)}

    async def _train_final_model(
        self,
        config: Dict[str, Any],
        X_train: torch.Tensor,
        X_val: torch.Tensor,
        y_train: torch.Tensor,
        y_val: torch.Tensor,
        task_type: str,
        metric: str
    ) -> Optional[ModelArtifact]:
        """Train final model with best configuration using async approach"""
        
        try:
            architecture_type = config.pop("architecture_type")
            
            # Create model
            input_dim = X_train.shape[1]
            output_dim = len(torch.unique(y_train)) if task_type == "classification" else 1
            
            model = PyTorchModelFactory.create_model(
                architecture_type, input_dim, output_dim, task_type, config
            )
            model.to(self.device)
            
            # Get training components
            optimizer = PyTorchModelFactory.get_optimizer(
                model, 
                config.get("optimizer", "adam"),
                config.get("learning_rate", 1e-3)
            )
            
            scheduler = PyTorchModelFactory.get_scheduler(
                optimizer,
                config.get("scheduler", "none")
            )
            
            loss_fn = model.get_loss_function(config.get("loss_type", "default"))
            
            # Shorter training for stability
            model.train()
            epochs = min(50, config.get("epochs", 30))  # Much shorter for stability
            best_val_score = float('-inf') if metric in ["accuracy", "precision", "recall", "f1"] else float('inf')
            patience_counter = 0
            patience = 10  # Reduced patience
            
            # Send training start update
            await self._send_training_start_update(architecture_type, epochs, X_train.shape[0], X_val.shape[0])
            
            loop = asyncio.get_running_loop()

            for epoch in range(epochs):
                epoch_start_time = time.time()
                
                # Training step in thread
                def _train_step():
                    optimizer.zero_grad()
                    outputs = model(X_train)
                    
                    if task_type == "classification":
                        loss = loss_fn(outputs, y_train)
                    else:
                        loss = loss_fn(outputs.squeeze(), y_train)
                    
                    loss.backward()
                    optimizer.step()
                    return loss.item()

                loss_item = await loop.run_in_executor(None, _train_step)
                
                if scheduler:
                    scheduler.step()
                
                # Evaluation step in thread
                def _eval_step():
                    train_score = self._evaluate_pytorch_model(model, X_train, y_train, task_type)
                    val_score = self._evaluate_pytorch_model(model, X_val, y_val, task_type)
                    return train_score, val_score

                train_score, val_score = await loop.run_in_executor(None, _eval_step)
                
                epoch_time = time.time() - epoch_start_time
                current_lr = optimizer.param_groups[0]['lr']
                
                # Send epoch update
                await self._send_epoch_update(
                    architecture_type, epoch + 1, epochs, 
                    loss_item, train_score, val_score, 
                    current_lr, epoch_time, best_val_score
                )
                
                is_better = (val_score > best_val_score) if metric in ["accuracy", "precision", "recall", "f1"] else (val_score < best_val_score)
                
                if is_better:
                    best_val_score = val_score
                    patience_counter = 0
                    await self._send_improvement_update(architecture_type, epoch + 1, val_score)
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    await self._send_early_stopping_update(architecture_type, epoch + 1, patience)
                    break
            
            # Final evaluation
            final_val_score = self._evaluate_pytorch_model(model, X_val, y_val, task_type)
            final_train_score = self._evaluate_pytorch_model(model, X_train, y_train, task_type)
            
            # Save model
            model_dir = Path(settings.MODELS_DIR) / self.run_id
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / f"pytorch_{architecture_type}.pth"
            
            torch.save({
                "model": model,
                "config": config,
                "model_type": architecture_type,
                "task_type": task_type,
                "device": str(self.device),
                "input_dim": input_dim,
                "output_dim": output_dim
            }, model_path)
            
            logger.info(f"✅ Final {architecture_type} model trained", 
                       val_score=f"{final_val_score:.4f}")
            
            return ModelArtifact(
                run_id=self.run_id,
                family=f"pytorch_{architecture_type}",
                model_path=str(model_path),
                val_score=final_val_score,
                train_score=final_train_score
            )
            
        except Exception as e:
            logger.error(f"Failed to train final model", error=str(e))
            return None

    def _prepare_pytorch_data(self, df: pd.DataFrame, target_col: str, 
                            constraints: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for PyTorch training with robust preprocessing"""
        
        # Handle target column
        y = df[target_col].copy()
        
        # Handle excluded columns
        exclude_cols = constraints.get("exclude_cols", [])
        feature_cols = [col for col in df.columns if col != target_col and col not in exclude_cols]
        X = df[feature_cols].copy()
        
        # Handle missing values
        for col in X.columns:
            if X[col].dtype in ['object', 'category']:
                X[col] = X[col].fillna('missing')
            else:
                X[col] = X[col].fillna(X[col].median())
        
        # Handle categorical features
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        # One-hot encode categorical features
        if len(categorical_cols) > 0:
            # Limit categories to prevent explosion
            for col in categorical_cols:
                value_counts = X[col].value_counts()
                top_categories = value_counts.head(20).index
                X[col] = X[col].where(X[col].isin(top_categories), 'other')
            
            # One-hot encode
            X_categorical = pd.get_dummies(X[categorical_cols], prefix_sep='_', dummy_na=False)
            X_combined = pd.concat([X[numeric_cols], X_categorical], axis=1)
        else:
            X_combined = X[numeric_cols]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_combined)
        
        # Handle target variable
        if df[target_col].dtype in ['object', 'category']:
            # Classification
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
        else:
            # Regression - ensure numeric
            y_encoded = pd.to_numeric(y, errors='coerce').fillna(y.median()).values
        
        # Save preprocessing info
        preprocessing_info = {
            "scaler": scaler,
            "feature_names": X_combined.columns.tolist(),
            "categorical_cols": categorical_cols.tolist(),
            "numeric_cols": numeric_cols.tolist(),
            "label_encoder": label_encoder if df[target_col].dtype in ['object', 'category'] else None
        }
        
        # Save preprocessing pipeline
        model_dir = Path(settings.MODELS_DIR) / self.run_id
        model_dir.mkdir(parents=True, exist_ok=True)
        preprocessing_path = model_dir / "pytorch_preprocessing.pkl"
        
        with open(preprocessing_path, 'wb') as f:
            pickle.dump(preprocessing_info, f)
        
        return X_scaled, y_encoded
    
    async def _send_model_completion_update(self, model_name: str, val_score: float, 
                                          search_strategy: str):
        """Send model completion update via WebSocket"""
        
        elapsed = time.time() - self.start_time
        await self.websocket_manager.broadcast_family_completion(
            self.session_id,
            {
                "event": "pytorch_model_complete",
                "model": model_name,
                "val_metric": val_score,
                "search_strategy": search_strategy,
                "elapsed_s": elapsed
            }
        )
    
    async def _send_training_start_update(self, architecture_type: str, total_epochs: int, 
                                        train_size: int, val_size: int):
        """Send training start update via WebSocket"""
        
        if self.websocket_manager:
            await self.websocket_manager.broadcast_to_session(
                self.session_id,
                {
                    "event": "training_start",
                    "model": f"pytorch_{architecture_type}",
                    "total_epochs": total_epochs,
                    "train_samples": train_size,
                    "val_samples": val_size,
                    "device": str(self.device),
                    "message": f"🚀 Starting {architecture_type} training - {total_epochs} epochs"
                }
            )
    
    async def _send_epoch_update(self, architecture_type: str, epoch: int, total_epochs: int,
                               loss: float, train_score: float, val_score: float,
                               learning_rate: float, epoch_time: float, best_val_score: float):
        """Send detailed epoch update via WebSocket"""
        
        if self.websocket_manager:
            # Progress percentage
            progress = (epoch / total_epochs) * 100
            
            # Format metrics for display
            is_classification = hasattr(self, '_task_type') and self._task_type == "classification"
            score_format = ".4f" if is_classification else ".6f"
            
            await self.websocket_manager.broadcast_to_session(
                self.session_id,
                {
                    "event": "epoch_update",
                    "model": f"pytorch_{architecture_type}",
                    "epoch": epoch,
                    "total_epochs": total_epochs,
                    "progress": round(progress, 1),
                    "loss": round(loss, 6),
                    "train_score": round(train_score, 4),
                    "val_score": round(val_score, 4),
                    "best_val_score": round(best_val_score, 4),
                    "learning_rate": f"{learning_rate:.2e}",
                    "epoch_time": round(epoch_time, 2),
                    "message": f"📊 Epoch {epoch}/{total_epochs} | Loss: {loss:.4f} | Val: {val_score:.4f} | Best: {best_val_score:.4f}"
                }
            )
    
    async def _send_improvement_update(self, architecture_type: str, epoch: int, new_best_score: float):
        """Send improvement notification via WebSocket"""
        
        if self.websocket_manager:
            await self.websocket_manager.broadcast_to_session(
                self.session_id,
                {
                    "event": "model_improvement",
                    "model": f"pytorch_{architecture_type}",
                    "epoch": epoch,
                    "new_best_score": round(new_best_score, 4),
                    "message": f"🎯 New best validation score: {new_best_score:.4f} at epoch {epoch}"
                }
            )
    
    async def _send_early_stopping_update(self, architecture_type: str, epoch: int, patience: int):
        """Send early stopping notification via WebSocket"""
        
        if self.websocket_manager:
            await self.websocket_manager.broadcast_to_session(
                self.session_id,
                {
                    "event": "early_stopping",
                    "model": f"pytorch_{architecture_type}",
                    "stopped_at_epoch": epoch,
                    "patience": patience,
                    "message": f"⏹️ Early stopping at epoch {epoch} (patience: {patience})"
                }
            )

    def _evaluate_pytorch_model(self, model, X: torch.Tensor, y: torch.Tensor, task_type: str) -> float:
        """Evaluate PyTorch model"""
        
        model.eval()
        with torch.no_grad():
            outputs = model(X)
            
            if task_type == "classification":
                predictions = torch.argmax(outputs, dim=1)
                accuracy = (predictions == y).float().mean().item()
                return accuracy
            else:
                mse = torch.mean((outputs.squeeze() - y) ** 2).item()
                return -mse  # Return negative MSE for consistency


class PyTorchEnsemble(nn.Module):
    """PyTorch ensemble that combines multiple neural networks"""
    
    def __init__(self, models: List[BaseNeuralNet], task_type: str):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.task_type = task_type
        self.num_models = len(models)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ensemble"""
        
        outputs = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                output = model(x)
                outputs.append(output)
        
        # Stack outputs
        stacked_outputs = torch.stack(outputs, dim=0)  # (num_models, batch_size, output_dim)
        
        if self.task_type == "classification":
            # Average probabilities
            averaged_outputs = torch.mean(stacked_outputs, dim=0)
            return averaged_outputs
        else:
            # Average predictions for regression
            averaged_outputs = torch.mean(stacked_outputs, dim=0)
            return averaged_outputs
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make predictions (for sklearn compatibility)"""
        
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            
            if self.task_type == "classification":
                return torch.argmax(outputs, dim=1)
            else:
                return outputs.squeeze() 