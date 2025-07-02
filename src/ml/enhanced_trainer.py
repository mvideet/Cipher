"""
Enhanced training orchestrator with LLM-guided model selection and ensemble support
"""

import asyncio
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import optuna
from sklearn.ensemble import VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error
import structlog

from ..core.config import settings
from ..models.schema import ModelArtifact, DataProfile
from .model_selector import ModelSelector, EnsembleStrategy, ModelRecommendation
from .trainer import TrainingOrchestrator  # Import original for fallback

logger = structlog.get_logger()


class EnhancedTrainingOrchestrator:
    """Enhanced training orchestrator with LLM-guided model selection"""
    
    def __init__(self, run_id: str, session_id: str, websocket_manager):
        self.run_id = run_id
        self.session_id = session_id
        self.websocket_manager = websocket_manager
        self.start_time = time.time()
        self.max_time_seconds = settings.MAX_TRAINING_TIME_MINUTES * 60
        self.model_selector = ModelSelector()
        
    async def train_ensemble_models(
        self,
        dataset_path: str,
        target_col: str,
        task_type: str,
        metric: str,
        constraints: Dict[str, Any],
        data_profile: DataProfile
    ) -> ModelArtifact:
        """Train ensemble of models with LLM guidance"""
        
        logger.info("🚀 Starting enhanced ensemble training", 
                   run_id=self.run_id, 
                   task=task_type, 
                   metric=metric)
        
        # Load and prepare data
        df = pd.read_csv(dataset_path)
        X, y = self._prepare_data(df, target_col, constraints)
        
        # PyTorch training removed from enhanced trainer
        
        # Get LLM recommendations for ensemble strategy
        try:
            from .model_selector import ModelSelector
            
            model_selector = ModelSelector()
            ensemble_strategy = await model_selector.recommend_ensemble(
                data_profile=data_profile,
                task_type=task_type,
                target_column=target_col,
                constraints=constraints
            )
            
            # Filter recommendations based on user selection if provided
            selected_models = constraints.get("selected_models", [])
            if selected_models:
                logger.info("👤 User selected specific models", selected_models=selected_models)
                
                # Filter to only selected model types
                filtered_models = []
                for model_rec in ensemble_strategy.recommended_models:
                    # Check if this model type was selected by the user
                    if model_rec.model_type in selected_models or any(
                        selected_model == model_rec.model_type or 
                        selected_model.startswith(model_rec.model_type) 
                        for selected_model in selected_models
                    ):
                        filtered_models.append(model_rec)
                        logger.info("✅ Including user-selected model", model_type=model_rec.model_type)
                    else:
                        logger.info("❌ Excluding non-selected model", model_type=model_rec.model_type)
                
                if filtered_models:
                    ensemble_strategy.recommended_models = filtered_models
                    logger.info("🎯 Filtered models based on user selection", 
                               selected=len(filtered_models),
                               original=len(ensemble_strategy.recommended_models))
                else:
                    logger.warning("⚠️ No models matched user selection, keeping all LLM recommendations")
            else:
                logger.info("🤖 Using all LLM recommendations (no user selection provided)")
            
            logger.info("🧠 LLM ensemble strategy received", 
                       n_models=len(ensemble_strategy.recommended_models),
                       strategy=ensemble_strategy.ensemble_method)
            
        except ImportError as e:
            logger.error("❌ Missing dependency for LLM recommendations", error=str(e))
            
            # Fallback to original trainer
            original_trainer = TrainingOrchestrator(self.run_id, self.session_id, self.websocket_manager)
            return await original_trainer.train_models(dataset_path, target_col, task_type, metric, constraints)
        
        except Exception as e:
            logger.error("❌ LLM recommendations failed, using fallback", error=str(e))
            
            # Enhanced fallback: Create a basic ensemble strategy for larger datasets
            if data_profile.n_rows > 1000:
                logger.info("🔄 Creating robust ensemble strategy for large dataset")
                fallback_models = self._create_large_dataset_strategy(data_profile, task_type)
            else:
                # Fallback to original trainer for smaller datasets
                original_trainer = TrainingOrchestrator(self.run_id, self.session_id, self.websocket_manager)
                return await original_trainer.train_models(dataset_path, target_col, task_type, metric, constraints)
                
            # Continue with fallback ensemble strategy
            ensemble_strategy = EnsembleStrategy(
                recommended_models=fallback_models,
                ensemble_method="voting",
                diversity_score=0.8,
                reasoning="Fallback strategy for large dataset when LLM unavailable"
            )
        
        # Data validation and splitting
        test_size = min(0.2, max(0.1, 10 / len(y)))
        
        # Convert y to numpy array for stratification to avoid DataFrame column selection issues
        y_array = np.array(y) if hasattr(y, 'values') else y
        stratify_param = y_array if task_type == "classification" and len(np.unique(y_array)) > 1 else None
        
        # Debug logging for data validation
        logger.info("🔍 Data validation", 
                   target_unique=np.unique(y).tolist(),
                   target_distribution=dict(zip(*np.unique(y, return_counts=True))),
                   feature_count=X.shape[1],
                   sample_count=len(y))
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify_param
        )
        
        # Ensure data types are preserved as pandas DataFrames/Series
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train, columns=X.columns)
        if not isinstance(X_val, pd.DataFrame):
            X_val = pd.DataFrame(X_val, columns=X.columns)
        if not isinstance(y_train, pd.Series):
            y_train = pd.Series(y_train, name=y.name if hasattr(y, 'name') else 'target')
        if not isinstance(y_val, pd.Series):
            y_val = pd.Series(y_val, name=y.name if hasattr(y, 'name') else 'target')
        
        logger.info("📊 Data prepared", 
                   train_size=len(X_train), 
                   val_size=len(X_val),
                   features=X.shape[1],
                   train_target_dist=dict(zip(*np.unique(y_train, return_counts=True))),
                   val_target_dist=dict(zip(*np.unique(y_val, return_counts=True))))
        
        # Configure Optuna logging to reduce noise
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        # Send initial status update
        await self._send_training_status_update("preparing_models", {
            "message": "Starting model architecture search",
            "train_size": len(X_train),
            "val_size": len(X_val),
            "features": X.shape[1]
        })
        
        # Train individual models for each recommendation
        trained_models = []
        
        total_architectures = sum(len(model_rec.architectures) for model_rec in ensemble_strategy.recommended_models)
        logger.info(f"🏗️ Training {total_architectures} model architectures...")
        
        # Send model training start update
        await self._send_training_status_update("training_models", {
            "message": f"Training {total_architectures} model architectures",
            "total_models": total_architectures,
            "progress": 0
        })
        
        # Train models sequentially to avoid resource contention
        completed_count = 0
        for model_rec in ensemble_strategy.recommended_models:
            # Check if this is a neural network - train them one at a time
            is_neural_network = model_rec.model_type == "neural_network"
            
            if is_neural_network:
                logger.info("🧠 Training neural networks sequentially to avoid resource contention")
                
            for arch in model_rec.architectures:
                try:
                    logger.info(f"🔄 Training architecture {completed_count + 1}/{total_architectures}: {model_rec.model_type}_{arch['name']}")
                    
                    # Train the model
                    result = await self._train_model_architecture(
                        model_rec, arch, X_train, X_val, y_train, y_val,
                        task_type, metric, constraints
                    )
                    
                    trained_models.append(result)
                    completed_count += 1
                    
                    # Send progress update
                    await self._send_training_status_update("model_completed", {
                        "message": f"Model {completed_count}/{total_architectures} completed",
                        "completed": completed_count,
                        "total": total_architectures,
                        "progress": int((completed_count / total_architectures) * 100)
                    })
                    
                except Exception as e:
                    logger.error(f"❌ Training failed for {model_rec.model_type}_{arch['name']}", error=str(e))
                    completed_count += 1
                    continue
        
        if not trained_models:
            raise ValueError("All model training failed")
        
        logger.info("🎯 Model training complete", 
                   successful=len(trained_models),
                   failed=total_architectures - len(trained_models))
        
        # Use only sklearn models (PyTorch removed)
        all_trained_models = trained_models
        
        # Create ensemble
        logger.info("🔗 Creating ensemble...")
        
        # Send ensemble creation update
        await self._send_training_status_update("creating_ensemble", {
            "message": f"Creating ensemble from {len(all_trained_models)} models",
            "sklearn_models": len(trained_models),
            "total_models": len(all_trained_models)
        })
        
        best_ensemble = await self._create_ensemble(
            all_trained_models, ensemble_strategy, X_train, X_val, y_train, y_val,
            task_type, metric
        )
        
        logger.info("🎉 Enhanced training completed", 
                   final_score=f"{best_ensemble.val_score:.4f}",
                   sklearn_models=len(trained_models),
                   total_models=len(all_trained_models))
        
        return best_ensemble
    
    def _create_large_dataset_strategy(self, data_profile: DataProfile, task_type: str) -> List['ModelRecommendation']:
        """Create a robust model strategy for large datasets when LLM is unavailable"""
        
        from .model_selector import ModelRecommendation
        
        logger.info("🔧 Creating fallback strategy for large dataset", 
                   n_rows=data_profile.n_rows, 
                   n_cols=data_profile.n_cols)
        
        fallback_models = []
        
        # Model 1: XGBoost (excellent for structured data)
        xgb_architectures = [
            {
                "name": "fast_xgb",
                "description": "Fast XGBoost for quick results",
                "config": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6},
                "complexity": "medium"
            },
            {
                "name": "tuned_xgb",
                "description": "Well-tuned XGBoost",
                "config": {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 8},
                "complexity": "high"
            },
            {
                "name": "precise_xgb",
                "description": "Precise XGBoost for best performance",
                "config": {"n_estimators": 500, "learning_rate": 0.01, "max_depth": 10},
                "complexity": "high"
            }
        ]
        
        fallback_models.append(ModelRecommendation(
            model_type="xgboost",
            model_family="tree_based",
            complexity_score=7.0,
            expected_training_time="medium",
            parameter_estimates={"n_estimators": 300, "learning_rate": 0.05},
            reasoning="XGBoost is robust for large structured datasets",
            architectures=xgb_architectures,
            pros=["Handles missing values", "Feature importance", "Robust performance"],
            cons=["Can overfit", "Hyperparameter sensitive"],
            best_for="Structured data with mixed feature types",
            training_time_estimate="5-15 minutes",
            memory_usage="medium",
            interpretability="medium"
        ))
        
        # Model 2: Random Forest (robust baseline)
        rf_architectures = [
            {
                "name": "balanced_rf",
                "description": "Balanced Random Forest",
                "config": {"n_estimators": 100, "max_depth": 15, "min_samples_split": 5},
                "complexity": "medium"
            },
            {
                "name": "deep_rf",
                "description": "Deep Random Forest",
                "config": {"n_estimators": 200, "max_depth": None, "min_samples_split": 2},
                "complexity": "high"
            },
            {
                "name": "wide_rf",
                "description": "Wide Random Forest",
                "config": {"n_estimators": 300, "max_depth": 20, "min_samples_split": 3},
                "complexity": "high"
            }
        ]
        
        fallback_models.append(ModelRecommendation(
            model_type="random_forest",
            model_family="tree_based",
            complexity_score=6.0,
            expected_training_time="fast",
            parameter_estimates={"n_estimators": 200, "max_depth": 20},
            reasoning="Random Forest is stable and works well with large datasets",
            architectures=rf_architectures,
            pros=["Stable", "Handles overfitting", "Good baseline"],
            cons=["Can be memory intensive", "Less precise than boosting"],
            best_for="Robust baseline with good interpretability",
            training_time_estimate="3-10 minutes",
            memory_usage="medium",
            interpretability="high"
        ))
        
        # Model 3: LightGBM (fast and efficient for large data)
        lgb_architectures = [
            {
                "name": "fast_lgb",
                "description": "Fast LightGBM",
                "config": {"n_estimators": 100, "learning_rate": 0.1, "num_leaves": 31},
                "complexity": "medium"
            },
            {
                "name": "efficient_lgb",
                "description": "Efficient LightGBM",
                "config": {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 63},
                "complexity": "high"
            },
            {
                "name": "precise_lgb",
                "description": "Precise LightGBM",
                "config": {"n_estimators": 500, "learning_rate": 0.02, "num_leaves": 127},
                "complexity": "high"
            }
        ]
        
        fallback_models.append(ModelRecommendation(
            model_type="lightgbm",
            model_family="tree_based",
            complexity_score=7.0,
            expected_training_time="fast",
            parameter_estimates={"n_estimators": 300, "learning_rate": 0.05},
            reasoning="LightGBM is very efficient for large datasets",
            architectures=lgb_architectures,
            pros=["Very fast", "Memory efficient", "Good performance"],
            cons=["Can overfit on small data", "Less stable than RF"],
            best_for="Large datasets requiring speed and efficiency",
            training_time_estimate="2-8 minutes",
            memory_usage="low",
            interpretability="medium"
        ))
        
        # Add neural network for complex patterns if dataset is large enough
        if data_profile.n_rows > 5000:
            nn_architectures = [
                {
                    "name": "simple_mlp",
                    "description": "Simple MLP for large data",
                    "config": {
                        "hidden_layer_sizes": (128, 64),
                        "activation": "relu",
                        "alpha": 0.001,
                        "max_iter": 500
                    },
                    "complexity": "medium"
                },
                {
                    "name": "deep_mlp",
                    "description": "Deep MLP for complex patterns",
                    "config": {
                        "hidden_layer_sizes": (256, 128, 64),
                        "activation": "relu",
                        "alpha": 0.0001,
                        "max_iter": 1000
                    },
                    "complexity": "high"
                },
                {
                    "name": "wide_mlp",
                    "description": "Wide MLP for feature interactions",
                    "config": {
                        "hidden_layer_sizes": (512, 256),
                        "activation": "relu",
                        "alpha": 0.00001,
                        "max_iter": 800
                    },
                    "complexity": "high"
                }
            ]
            
            fallback_models.append(ModelRecommendation(
                model_type="neural_network",
                model_family="neural",
                complexity_score=8.0,
                expected_training_time="slow",
                parameter_estimates={"hidden_layer_sizes": (256, 128), "max_iter": 800},
                reasoning="Neural network can capture complex patterns in large datasets",
                architectures=nn_architectures,
                pros=["Captures non-linear patterns", "Flexible", "Good for complex data"],
                cons=["Requires more data", "Slower training", "Less interpretable"],
                best_for="Large datasets with complex feature interactions",
                training_time_estimate="10-30 minutes",
                memory_usage="high",
                interpretability="low"
            ))
        
        logger.info("✅ Created fallback strategy", n_models=len(fallback_models))
        return fallback_models
    
    async def _train_model_architecture(
        self,
        model_rec: ModelRecommendation,
        architecture: Dict[str, Any],
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        task_type: str,
        metric: str,
        constraints: Dict[str, Any]
    ) -> ModelArtifact:
        """Train a specific model architecture"""
        
        model_name = f"{model_rec.model_type}_{architecture['name']}"
        logger.info(f"Training {model_name}", complexity=architecture['complexity'])
        
        # Neural networks use sklearn MLP (PyTorch removed)
        # Continue with normal sklearn model training
        
        # Create preprocessing pipeline for sklearn models
        preprocessor = self._create_preprocessor(X_train, constraints, task_type)
        
        # Create and configure model
        model = self._create_model_instance(
            model_rec.model_type, task_type, architecture['config']
        )
        
        # Create pipeline
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])
        
        # Optuna optimization for this specific architecture
        study = optuna.create_study(
            direction="maximize" if metric in ['accuracy', 'precision', 'recall', 'f1'] else "minimize",
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        def objective(trial):
            # Fine-tune the architecture parameters
            tuned_params = self._tune_architecture_params(
                trial, model_rec.model_type, architecture['config']
            )
            
            # Create model with tuned params
            tuned_model = self._create_model_instance(
                model_rec.model_type, task_type, tuned_params
            )
            
            tuned_pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", tuned_model)
            ])
            
            # Train and evaluate
            tuned_pipeline.fit(X_train, y_train)
            y_pred = tuned_pipeline.predict(X_val)
            score = self._calculate_metric(y_val, y_pred, metric)
            
            return score
        
        # Run optimization with time budget per model
        max_trials = min(20, max(5, 60 // len(architecture.get('config', {}))))
        
        loop = asyncio.get_running_loop()
        for trial_num in range(max_trials):
            if time.time() - self.start_time > self.max_time_seconds * 0.8:  # Reserve time for ensemble
                break
            
            try:
                await loop.run_in_executor(
                    None,
                    lambda: study.optimize(objective, n_trials=1, timeout=120)
                )
                await self._send_architecture_trial_update(model_name, trial_num + 1, study.best_value)
            except Exception as e:
                logger.error(f"Trial {trial_num} failed for {model_name}", error=str(e))
                continue
        
        # Train final model with best params
        if len(study.trials) > 0:
            best_params = study.best_params
            final_config = {**architecture['config'], **best_params}
        else:
            final_config = architecture['config']
        
        final_model = self._create_model_instance(model_rec.model_type, task_type, final_config)
        final_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", final_model)
        ])
        
        final_pipeline.fit(X_train, y_train)
        
        # Evaluate
        val_pred = final_pipeline.predict(X_val)
        train_pred = final_pipeline.predict(X_train)
        
        val_score = self._calculate_metric(y_val, val_pred, metric)
        train_score = self._calculate_metric(y_train, train_pred, metric)
        
        # Save model
        model_dir = Path(settings.MODELS_DIR) / self.run_id
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"{model_name}_model.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(final_pipeline, f)
        
        await self._send_architecture_completion(model_name, val_score)
        
        return ModelArtifact(
            run_id=self.run_id,
            family=model_name,
            model_path=str(model_path),
            val_score=val_score,
            train_score=train_score
        )
    
    # PyTorch architecture training method removed completely
    
    async def _create_ensemble(
        self,
        trained_models: List[ModelArtifact],
        ensemble_strategy: EnsembleStrategy,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        task_type: str,
        metric: str
    ) -> ModelArtifact:
        """Create ensemble from trained models"""
        
        logger.info("Creating ensemble", 
                   method=ensemble_strategy.ensemble_method,
                   n_models=len(trained_models))
        
        # Load trained models (PyTorch support removed)
        estimators = []
        
        for model_artifact in trained_models:
            try:
                # Load sklearn models
                with open(model_artifact.model_path, 'rb') as f:
                    model_or_info = pickle.load(f)
                
                # Handle sklearn pipeline/model
                estimators.append((model_artifact.family, model_or_info))
                logger.info(f"Added sklearn model {model_artifact.family} to ensemble")
                
            except Exception as e:
                logger.error(f"Failed to load model {model_artifact.family}", error=str(e))
                continue
        
        if not estimators:
            raise ValueError("No valid sklearn models available for ensemble creation")
        
        # Create ensemble based on strategy
        if ensemble_strategy.ensemble_method == "voting":
            if task_type == "classification":
                ensemble = VotingClassifier(estimators=estimators, voting='soft')
            else:
                ensemble = VotingRegressor(estimators=estimators)
                
        elif ensemble_strategy.ensemble_method == "stacking":
            # Use best individual model as meta-learner
            best_model = max(trained_models, key=lambda x: x.val_score if metric in ['accuracy', 'precision', 'recall', 'f1'] else -x.val_score)
            
            with open(best_model.model_path, 'rb') as f:
                meta_learner = pickle.load(f)
            
            if task_type == "classification":
                ensemble = StackingClassifier(estimators=estimators, final_estimator=meta_learner)
            else:
                ensemble = StackingRegressor(estimators=estimators, final_estimator=meta_learner)
        
        else:  # Default to voting
            if task_type == "classification":
                ensemble = VotingClassifier(estimators=estimators, voting='hard')
            else:
                ensemble = VotingRegressor(estimators=estimators)
        
        # Send ensemble training update
        await self._send_training_status_update("ensemble_training", {
            "message": f"Training {ensemble_strategy.ensemble_method} ensemble",
            "method": ensemble_strategy.ensemble_method,
            "n_models": len(trained_models)
        })
        
        # Train ensemble
        ensemble.fit(X_train, y_train)
        
        # Evaluate ensemble
        val_pred = ensemble.predict(X_val)
        train_pred = ensemble.predict(X_train)
        
        val_score = self._calculate_metric(y_val, val_pred, metric)
        train_score = self._calculate_metric(y_train, train_pred, metric)
        
        # Wrap ensemble in a pipeline for consistency with explainer
        ensemble_pipeline = Pipeline([('model', ensemble)])
        
        # Save ensemble pipeline
        model_dir = Path(settings.MODELS_DIR) / self.run_id
        model_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        ensemble_path = model_dir / "ensemble_model.pkl"
        
        with open(ensemble_path, 'wb') as f:
            pickle.dump(ensemble_pipeline, f)
        
        logger.info("🎯 Ensemble created", 
                   method=ensemble_strategy.ensemble_method,
                   val_score=f"{val_score:.4f}",
                   n_models=len(trained_models))
        
        return ModelArtifact(
            run_id=self.run_id,
            family=f"ensemble_{ensemble_strategy.ensemble_method}",
            model_path=str(ensemble_path),
            val_score=val_score,
            train_score=train_score
        )
    
    def _create_model_instance(self, model_type: str, task_type: str, config: Dict[str, Any]):
        """Create model instance based on type and config"""
        
        # Import models dynamically to avoid import issues
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor
        from sklearn.svm import SVC, SVR
        from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        from sklearn.naive_bayes import GaussianNB
        import lightgbm as lgb
        import xgboost as xgb
        
        # Filter config to only include valid parameters (exclude key_params and other invalid ones)
        invalid_params = {'key_params', 'model_type', 'name', 'complexity', 'description'}
        safe_config = {k: v for k, v in config.items() if k not in invalid_params and v is not None}
        
        # Add random state for reproducibility (exclude deterministic models)
        if 'random_state' not in safe_config and model_type not in ['naive_bayes', 'knn', 'linear_regression']:
            safe_config['random_state'] = 42
        
        try:
            if model_type == "random_forest":
                return RandomForestClassifier(**safe_config) if task_type == "classification" else RandomForestRegressor(**safe_config)
            elif model_type == "extra_trees":
                return ExtraTreesClassifier(**safe_config) if task_type == "classification" else ExtraTreesRegressor(**safe_config)
            elif model_type == "xgboost":
                return xgb.XGBClassifier(**safe_config) if task_type == "classification" else xgb.XGBRegressor(**safe_config)
            elif model_type == "lightgbm":
                return lgb.LGBMClassifier(**safe_config, verbose=-1) if task_type == "classification" else lgb.LGBMRegressor(**safe_config, verbose=-1)
            elif model_type == "neural_network":
                # Use sklearn MLPClassifier/MLPRegressor for neural networks
                from sklearn.neural_network import MLPClassifier, MLPRegressor
                return MLPClassifier(**safe_config) if task_type == "classification" else MLPRegressor(**safe_config)
            elif model_type == "svm":
                return SVC(**safe_config, probability=True) if task_type == "classification" else SVR(**safe_config)
            elif model_type == "logistic_regression":
                if task_type == "classification":
                    # Add robust defaults for better convergence
                    lr_defaults = {
                        'max_iter': 2000, 
                        'solver': 'saga',  # More robust solver
                        'tol': 1e-4
                    }
                    lr_config = {**lr_defaults, **safe_config}
                    return LogisticRegression(**lr_config)
                else:
                    return LinearRegression()
            elif model_type == "linear_regression":
                # LinearRegression doesn't accept random_state, filter it out
                lr_config = {k: v for k, v in safe_config.items() if k != 'random_state'}
                return LinearRegression(**lr_config)
            elif model_type == "ridge":
                if task_type == "classification":
                    # Use LogisticRegression with L2 penalty for classification
                    lr_defaults = {
                        'max_iter': 2000, 
                        'solver': 'saga', 
                        'penalty': 'l2',
                        'tol': 1e-4
                    }
                    lr_config = {**lr_defaults, **safe_config}
                    return LogisticRegression(**lr_config)
                else:
                    return Ridge(**safe_config)
            elif model_type == "lasso":
                if task_type == "classification":
                    # Use LogisticRegression with L1 penalty for classification  
                    lr_defaults = {
                        'max_iter': 2000, 
                        'solver': 'saga',  # saga supports L1 
                        'penalty': 'l1',
                        'tol': 1e-4
                    }
                    lr_config = {**lr_defaults, **safe_config}
                    return LogisticRegression(**lr_config)
                else:
                    return Lasso(**safe_config)
            elif model_type == "knn":
                return KNeighborsClassifier(**safe_config) if task_type == "classification" else KNeighborsRegressor(**safe_config)
            elif model_type == "naive_bayes":
                return GaussianNB(**safe_config)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
                
        except Exception as e:
            logger.error(f"Failed to create {model_type} with config {safe_config}", error=str(e))
            # Fallback to basic config
            if model_type == "random_forest":
                return RandomForestClassifier(random_state=42) if task_type == "classification" else RandomForestRegressor(random_state=42)
            else:
                raise e
    
    def _tune_architecture_params(self, trial: optuna.Trial, model_type: str, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Fine-tune architecture parameters with Optuna"""
        
        tuned_config = base_config.copy()
        
        if model_type == "neural_network":
            # Fine-tune neural network specific params
            if 'alpha' in base_config:
                tuned_config['alpha'] = trial.suggest_float('alpha', base_config['alpha'] * 0.1, base_config['alpha'] * 10, log=True)
            if 'learning_rate_init' in base_config:
                tuned_config['learning_rate_init'] = trial.suggest_float('learning_rate_init', base_config['learning_rate_init'] * 0.1, base_config['learning_rate_init'] * 10, log=True)
                
        elif model_type in ["random_forest", "extra_trees"]:
            # Fine-tune tree params
            if 'n_estimators' in base_config:
                tuned_config['n_estimators'] = trial.suggest_int('n_estimators', max(10, base_config['n_estimators'] - 50), base_config['n_estimators'] + 100)
            if 'max_depth' in base_config and base_config['max_depth'] is not None:
                tuned_config['max_depth'] = trial.suggest_int('max_depth', max(3, base_config['max_depth'] - 5), base_config['max_depth'] + 10)
                
        elif model_type in ["xgboost", "lightgbm"]:
            # Fine-tune boosting params
            if 'learning_rate' in base_config:
                tuned_config['learning_rate'] = trial.suggest_float('learning_rate', base_config['learning_rate'] * 0.5, base_config['learning_rate'] * 2)
            if 'n_estimators' in base_config:
                tuned_config['n_estimators'] = trial.suggest_int('n_estimators', max(50, base_config['n_estimators'] - 100), base_config['n_estimators'] + 200)
        
        return tuned_config
    
    def _prepare_data(self, df: pd.DataFrame, target_col: str, constraints: Dict[str, Any]):
        """Prepare features and target - reuse from original trainer"""
        exclude_cols = constraints.get("exclude_cols", [])
        feature_cols = [col for col in df.columns if col != target_col and col not in exclude_cols]
        
        X = df[feature_cols]
        y = df[target_col]
        
        # Encode string targets for classification
        if y.dtype == 'object':
            label_encoder = LabelEncoder()
            y = pd.Series(label_encoder.fit_transform(y), index=y.index)
            logger.info("🏷️ Target encoded", 
                       original_labels=label_encoder.classes_.tolist(),
                       encoded_mapping={label: i for i, label in enumerate(label_encoder.classes_)})
        
        return X, y
    
    def _create_preprocessor(self, X: pd.DataFrame, constraints: Dict[str, Any], task_type: str = "classification"):
        """Create preprocessing pipeline - reuse from original trainer"""
        # Get column indices instead of names to avoid DataFrame column selection issues
        numeric_features = []
        categorical_features = []
        
        for i, col in enumerate(X.columns):
            if X[col].dtype in ['int64', 'int32', 'float64', 'float32']:
                numeric_features.append(i)
            else:
                categorical_features.append(i)
        
        # Also create backup with column names for logging
        numeric_feature_names = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_feature_names = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        logger.debug("Creating preprocessor", 
                    numeric_features=len(numeric_features), 
                    categorical_features=len(categorical_features),
                    numeric_names=numeric_feature_names,
                    categorical_names=categorical_feature_names)
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Use column indices instead of names
        transformers = []
        if numeric_features:
            transformers.append(('num', numeric_transformer, numeric_features))
        if categorical_features:
            transformers.append(('cat', categorical_transformer, categorical_features))
        
        if not transformers:
            # If no features, create a passthrough transformer
            preprocessor = Pipeline([('passthrough', StandardScaler())])
        else:
            preprocessor = ColumnTransformer(transformers=transformers)
        
        # Add feature selection if requested
        feature_limit = constraints.get("feature_limit")
        if feature_limit and transformers:
            estimated_features = len(numeric_features)
            for cat_col_idx in categorical_features:
                cat_col_name = X.columns[cat_col_idx]
                estimated_features += min(X[cat_col_name].nunique(), 20)
            
            actual_limit = min(feature_limit, estimated_features, len(X.columns))
            
            if actual_limit > 0 and actual_limit < estimated_features:
                # Choose appropriate score function based on task type
                score_func = f_classif if task_type == "classification" else f_regression
                preprocessor = Pipeline([
                    ("transform", preprocessor),
                    ("select", SelectKBest(score_func=score_func, k=actual_limit))
                ])
        
        return preprocessor
    
    def _calculate_metric(self, y_true, y_pred, metric: str) -> float:
        """Calculate specified metric - reuse from original trainer"""
        
        # Debug logging
        unique_true = np.unique(y_true)
        unique_pred = np.unique(y_pred)
        logger.debug(f"Calculating {metric}", 
                    y_true_unique=unique_true.tolist(),
                    y_pred_unique=unique_pred.tolist(),
                    y_true_shape=y_true.shape,
                    y_pred_shape=y_pred.shape)
        
        try:
            if metric == "accuracy":
                score = float(accuracy_score(y_true, y_pred))
            elif metric == "precision":
                score = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
            elif metric == "recall":
                score = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
            elif metric == "f1":
                score = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
            elif metric == "rmse":
                score = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            elif metric == "mae":
                score = float(mean_absolute_error(y_true, y_pred))
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            logger.debug(f"Calculated {metric}", score=score)
            return score
            
        except Exception as e:
            logger.error(f"Error calculating {metric}", error=str(e))
            return 0.0
    
    async def _send_ensemble_strategy_update(self, ensemble_strategy: EnsembleStrategy):
        """Send ensemble strategy update via WebSocket"""
        await self.websocket_manager.broadcast_trial_update(
            self.session_id,
            {
                "event": "ensemble_strategy",
                "strategy": ensemble_strategy.ensemble_method,
                "models": [rec.model_type for rec in ensemble_strategy.recommended_models],
                "reasoning": ensemble_strategy.reasoning
            }
        )
    
    async def _send_architecture_trial_update(self, model_name: str, trial: int, best_score: float):
        """Send architecture trial update via WebSocket"""
        await self.websocket_manager.broadcast_trial_update(
            self.session_id,
            {
                "event": "architecture_trial",
                "family": model_name,
                "model": model_name,
                "trial": trial,
                "val_metric": best_score,
                "elapsed_s": int(time.time() - self.start_time)
            }
        )
    
    async def _send_architecture_completion(self, model_name: str, val_score: float):
        """Send architecture completion notification via WebSocket"""
        elapsed = time.time() - self.start_time
        await self.websocket_manager.broadcast_family_completion(
            self.session_id,
            {
                "event": "architecture_complete",
                "model": model_name,
                "val_metric": val_score,
                "elapsed_s": elapsed
            }
        )

    async def _send_training_status_update(self, status: str, message: Dict[str, Any]):
        """Send training status update via WebSocket"""
        await self.websocket_manager.broadcast_training_status(
            self.session_id,
            {
                "event": "training_status",
                "status": status,
                "message": message
            }
        ) 