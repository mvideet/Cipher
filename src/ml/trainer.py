"""
Training orchestrator with multiple model families and Optuna hyperparameter optimization
"""

import asyncio
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import optuna
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error
import lightgbm as lgb
import structlog

from ..core.config import settings
from ..models.schema import ModelArtifact

logger = structlog.get_logger()


class TrainingOrchestrator:
    """Orchestrates training of multiple model families with hyperparameter optimization"""
    
    def __init__(self, run_id: str, session_id: str, websocket_manager):
        self.run_id = run_id
        self.session_id = session_id
        self.websocket_manager = websocket_manager
        self.start_time = time.time()
        self.max_time_seconds = settings.MAX_TRAINING_TIME_MINUTES * 60
        
    async def train_models(
        self,
        dataset_path: str,
        target_col: str,
        task_type: str,
        metric: str,
        constraints: Dict[str, Any]
    ) -> ModelArtifact:
        """Train multiple model families and return the best one"""
        
        logger.info("Starting model training", 
                   run_id=self.run_id, 
                   task_type=task_type, 
                   metric=metric)
        
        # Load and prepare data
        df = pd.read_csv(dataset_path)
        X, y = self._prepare_data(df, target_col, constraints)
        
        # Check for class imbalance issues in classification
        if task_type == "classification":
            class_counts = y.value_counts()
            min_class_count = class_counts.min()
            
            logger.info("Class distribution", class_counts=class_counts.to_dict())
            
            # Check if we have enough samples for cross-validation
            if min_class_count < 2:
                raise ValueError(f"Class imbalance detected: The least populated class has only {min_class_count} sample(s). "
                               f"Each class needs at least 2 samples for proper validation. "
                               f"Please provide more balanced data or add more samples.")
            
            # For severely imbalanced small datasets, suggest synthetic data generation
            if len(y) < 200 and min_class_count < 10:
                logger.warning(f"Very small and imbalanced dataset detected. "
                             f"Consider using synthetic data generation to improve balance.")
            
            # For very small datasets, use different validation strategy
            if len(y) < 100 or min_class_count < 5:
                logger.warning("Small dataset detected, using simple train/test split without stratification")
                # Use stratify only if we have enough samples
                stratify_param = y if min_class_count >= 5 else None
            else:
                stratify_param = y
        else:
            stratify_param = None
        
        # Split data
        test_size = min(0.2, max(0.1, 10 / len(y)))  # Adaptive test size for small datasets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42,
            stratify=stratify_param
        )
        
        logger.info("Data split completed", 
                   train_size=len(X_train), 
                   val_size=len(X_val))
        
        # Define model families to train
        families = ["baseline", "lightgbm", "mlp"]
        
        # Train models concurrently
        tasks = []
        for family in families:
            task = asyncio.create_task(
                self._train_family(
                    family, X_train, X_val, y_train, y_val,
                    task_type, metric, constraints
                )
            )
            tasks.append(task)
        
        # Wait for all training to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Find best model
        best_model = None
        best_score = float('-inf') if metric in ['accuracy', 'precision', 'recall', 'f1'] else float('inf')
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Training failed for {families[i]}", error=str(result))
                continue
            
            model_artifact = result
            if self._is_better_score(model_artifact.val_score, best_score, metric):
                best_score = model_artifact.val_score
                best_model = model_artifact
        
        if best_model is None:
            raise ValueError("All model training failed")
        
        logger.info("Training completed", 
                   best_family=best_model.family,
                   best_score=best_model.val_score)
        
        return best_model
    
    def _prepare_data(self, df: pd.DataFrame, target_col: str, constraints: Dict[str, Any]):
        """Prepare features and target"""
        
        # Drop excluded columns
        exclude_cols = constraints.get("exclude_cols", [])
        feature_cols = [col for col in df.columns if col != target_col and col not in exclude_cols]
        
        X = df[feature_cols]
        y = df[target_col]
        
        return X, y
    
    async def _train_family(
        self,
        family: str,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        task_type: str,
        metric: str,
        constraints: Dict[str, Any]
    ) -> ModelArtifact:
        """Train a specific model family with hyperparameter optimization"""
        
        logger.info(f"Training {family} family", run_id=self.run_id)
        
        # Create preprocessing pipeline
        preprocessor = self._create_preprocessor(X_train, constraints)
        
        # Create study
        study = optuna.create_study(
            direction="maximize" if metric in ['accuracy', 'precision', 'recall', 'f1'] else "minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        )
        
        # Define objective function
        def objective(trial):
            return self._objective_function(
                trial, family, preprocessor, X_train, X_val, y_train, y_val,
                task_type, metric
            )
        
        # Run optimization
        max_trials = min(settings.MAX_OPTUNA_TRIALS, 20)
        
        # Reduce trials for very small datasets to avoid overfitting
        if len(X_train) < 100:
            max_trials = min(max_trials, 5)
            logger.info(f"Small dataset detected, reducing Optuna trials to {max_trials}")
        
        # Run Optuna study in a separate thread to avoid blocking asyncio event loop
        loop = asyncio.get_running_loop()
        
        for trial_num in range(max_trials):
            if time.time() - self.start_time > self.max_time_seconds:
                logger.info(f"Time budget exceeded for {family}")
                break
            
            try:
                # This is a blocking call, so we run it in a thread
                await loop.run_in_executor(
                    None,  # Use default executor
                    lambda: study.optimize(objective, n_trials=1, timeout=300)
                )
                
                # Send progress update
                await self._send_trial_update(family, trial_num + 1, study.best_value)
                
            except Exception as e:
                logger.error(f"Trial {trial_num} failed for {family}", error=str(e))
                continue
        
        if len(study.trials) == 0:
            raise ValueError(f"No successful trials for {family}")
        
        # Train final model with best params
        best_params = study.best_params
        final_model = self._create_model(family, task_type, best_params)
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", final_model)
        ])
        
        pipeline.fit(X_train, y_train)
        
        # Evaluate
        val_pred = pipeline.predict(X_val)
        train_pred = pipeline.predict(X_train)
        
        val_score = self._calculate_metric(y_val, val_pred, metric)
        train_score = self._calculate_metric(y_train, train_pred, metric)
        
        # Save model
        model_dir = Path(settings.MODELS_DIR) / self.run_id
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"{family}_model.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(pipeline, f)
        
        # Send completion notification
        await self._send_family_completion(family, val_score)
        
        return ModelArtifact(
            run_id=self.run_id,
            family=family,
            model_path=str(model_path),
            val_score=val_score,
            train_score=train_score
        )
    
    def _create_preprocessor(self, X: pd.DataFrame, constraints: Dict[str, Any]):
        """Create preprocessing pipeline"""
        
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Add feature selection if requested
        feature_limit = constraints.get("feature_limit")
        if feature_limit:
            # Estimate the number of features after preprocessing
            # For categorical features, estimate based on unique values
            estimated_features = len(numeric_features)
            for cat_col in categorical_features:
                estimated_features += min(X[cat_col].nunique(), 20)  # Cap at 20 for high cardinality
            
            # Don't select more features than we'll have
            actual_limit = min(feature_limit, estimated_features, len(X.columns) - 1)
            
            if actual_limit > 0:
                preprocessor = Pipeline([
                    ("transform", preprocessor),
                    ("select", SelectKBest(score_func=f_classif, k=actual_limit))
                ])
                logger.info(f"Feature selection enabled: selecting {actual_limit} features")
        
        return preprocessor
    
    def _objective_function(
        self, trial, family: str, preprocessor, X_train, X_val, y_train, y_val,
        task_type: str, metric: str
    ) -> float:
        """Optuna objective function"""
        
        # Get model with trial parameters
        model = self._create_model(family, task_type, self._suggest_params(trial, family, task_type))
        
        # Create pipeline
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])
        
        # Train and evaluate
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)
        
        score = self._calculate_metric(y_val, y_pred, metric)
        return score
    
    def _suggest_params(self, trial, family: str, task_type: str) -> Dict[str, Any]:
        """Suggest hyperparameters for Optuna trial"""
        
        if family == "baseline":
            if task_type == "classification":
                return {
                    "C": trial.suggest_float("C", 1e-3, 1e3, log=True),
                    "max_iter": 1000
                }
            else:
                return {}
        
        elif family == "lightgbm":
            return {
                "num_leaves": trial.suggest_int("num_leaves", 8, 512),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                "n_estimators": trial.suggest_int("n_estimators", 100, 600),
                "random_state": 42
            }
        
        elif family == "mlp":
            layer_sizes = trial.suggest_categorical("hidden_layer_sizes", [
                (128,), (128, 64), (256, 128, 64)
            ])
            return {
                "hidden_layer_sizes": layer_sizes,
                "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
                "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True),
                "max_iter": 500,
                "random_state": 42
            }
        
        return {}
    
    def _create_model(self, family: str, task_type: str, params: Dict[str, Any]):
        """Create model instance with parameters"""
        
        if family == "baseline":
            if task_type == "classification":
                return LogisticRegression(**params, random_state=42)
            else:
                return LinearRegression(**params)
        
        elif family == "lightgbm":
            if task_type == "classification":
                return lgb.LGBMClassifier(**params, verbose=-1)
            else:
                return lgb.LGBMRegressor(**params, verbose=-1)
        
        elif family == "mlp":
            if task_type == "classification":
                return MLPClassifier(**params)
            else:
                return MLPRegressor(**params)
        
        raise ValueError(f"Unknown model family: {family}")
    
    def _calculate_metric(self, y_true, y_pred, metric: str) -> float:
        """Calculate specified metric"""
        
        if metric == "accuracy":
            return float(accuracy_score(y_true, y_pred))
        elif metric == "precision":
            return float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        elif metric == "recall":
            return float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
        elif metric == "f1":
            return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        elif metric == "rmse":
            return float(np.sqrt(mean_squared_error(y_true, y_pred)))
        elif metric == "mae":
            return float(mean_absolute_error(y_true, y_pred))
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def _is_better_score(self, new_score: float, current_best: float, metric: str) -> bool:
        """Check if new score is better than current best"""
        if metric in ['accuracy', 'precision', 'recall', 'f1']:
            return new_score > current_best
        else:  # Lower is better for rmse, mae
            return new_score < current_best
    
    async def _send_trial_update(self, family: str, trial: int, best_score: float):
        """Send trial progress update via WebSocket"""
        
        elapsed = time.time() - self.start_time
        await self.websocket_manager.broadcast_trial_update(
            self.session_id,
            {
                "event": "trial_complete",
                "family": family,
                "trial": trial,
                "val_metric": best_score,
                "elapsed_s": elapsed
            }
        )

    async def _send_family_completion(self, family: str, val_score: float):
        """Send family completion notification via WebSocket"""
        
        elapsed = time.time() - self.start_time
        await self.websocket_manager.broadcast_family_completion(
            self.session_id,
            {
                "event": "family_complete",
                "family": family,
                "val_metric": val_score,
                "elapsed_s": elapsed
            }
        ) 