"""
Neural Architecture Search (NAS) engine for PyTorch models
"""

import time
import optuna
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import structlog

from .pytorch_models import PyTorchModelFactory, BaseNeuralNet
from ..core.config import settings

logger = structlog.get_logger()


class PyTorchNAS:
    """Neural Architecture Search for PyTorch models"""
    
    def __init__(self, task_type: str, metric: str, max_time_minutes: int = 10):
        self.task_type = task_type
        self.metric = metric
        self.max_time_seconds = max_time_minutes * 60
        self.start_time = None
        
        # Architecture search spaces
        self.search_spaces = {
            "simple_mlp": self._get_simple_mlp_space,
            "resnet_mlp": self._get_resnet_mlp_space,
            "attention_mlp": self._get_attention_mlp_space
        }
        
        # Progressive search configuration
        self.progressive_stages = [
            {"max_layers": 2, "max_width": 128, "trials": 15},
            {"max_layers": 4, "max_width": 256, "trials": 20},
            {"max_layers": 6, "max_width": 512, "trials": 25}
        ]
        
    def search_architecture(self, X_train: torch.Tensor, y_train: torch.Tensor,
                          X_val: torch.Tensor, y_val: torch.Tensor,
                          architecture_type: str = "auto",
                          search_strategy: str = "progressive") -> Dict[str, Any]:
        """Search for optimal neural architecture"""
        
        self.start_time = time.time()
        logger.info("🔍 Starting Neural Architecture Search", 
                   architecture=architecture_type, strategy=search_strategy)
        
        if architecture_type == "auto":
            return self._search_all_architectures(X_train, y_train, X_val, y_val, search_strategy)
        else:
            return self._search_single_architecture(
                X_train, y_train, X_val, y_val, architecture_type, search_strategy
            )
    
    def _search_all_architectures(self, X_train: torch.Tensor, y_train: torch.Tensor,
                                X_val: torch.Tensor, y_val: torch.Tensor,
                                search_strategy: str) -> Dict[str, Any]:
        """Search across all architecture types"""
        
        best_results = []
        
        for arch_type in ["simple_mlp", "resnet_mlp", "attention_mlp"]:
            logger.info(f"🏗️ Searching {arch_type} architecture...")
            
            result = self._search_single_architecture(
                X_train, y_train, X_val, y_val, arch_type, search_strategy
            )
            
            result["architecture_type"] = arch_type
            best_results.append(result)
            
            # Check time budget
            if time.time() - self.start_time > self.max_time_seconds * 0.8:
                logger.info("⏰ Time budget almost exhausted, stopping architecture search")
                break
        
        # Return best overall result
        best_result = max(best_results, key=lambda x: x["best_score"] if self._higher_is_better() else -x["best_score"])
        logger.info("🎯 Best architecture found", 
                   type=best_result["architecture_type"],
                   score=f"{best_result['best_score']:.4f}")
        
        return best_result
    
    def _search_single_architecture(self, X_train: torch.Tensor, y_train: torch.Tensor,
                                  X_val: torch.Tensor, y_val: torch.Tensor,
                                  architecture_type: str, search_strategy: str) -> Dict[str, Any]:
        """Search within a single architecture type"""
        
        if search_strategy == "progressive":
            return self._progressive_search(X_train, y_train, X_val, y_val, architecture_type)
        elif search_strategy == "optuna":
            return self._optuna_search(X_train, y_train, X_val, y_val, architecture_type)
        else:  # random
            return self._random_search(X_train, y_train, X_val, y_val, architecture_type)
    
    def _progressive_search(self, X_train: torch.Tensor, y_train: torch.Tensor,
                          X_val: torch.Tensor, y_val: torch.Tensor,
                          architecture_type: str) -> Dict[str, Any]:
        """Progressive search: start small, gradually increase complexity"""
        
        logger.info("📈 Starting progressive search")
        best_config = None
        best_score = float('-inf') if self._higher_is_better() else float('inf')
        
        for stage_idx, stage in enumerate(self.progressive_stages):
            logger.info(f"🔄 Progressive stage {stage_idx + 1}/3", 
                       max_layers=stage["max_layers"],
                       max_width=stage["max_width"])
            
            # Configure Optuna study for this stage
            study = optuna.create_study(
                direction="maximize" if self._higher_is_better() else "minimize",
                sampler=optuna.samplers.TPESampler(seed=42)
            )
            
            # Suppress Optuna logging
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            
            # Define objective for this stage
            def objective(trial):
                return self._evaluate_architecture(
                    trial, X_train, y_train, X_val, y_val,
                    architecture_type, stage
                )
            
            # Run optimization for this stage
            study.optimize(
                objective, 
                n_trials=stage["trials"],
                timeout=self.max_time_seconds // len(self.progressive_stages)
            )
            
            # Check if this stage produced better results
            stage_best_score = study.best_value
            if self._is_better_score(stage_best_score, best_score):
                best_score = stage_best_score
                best_config = study.best_params.copy()
                best_config.update({"architecture_type": architecture_type})
                
                logger.info("✅ New best configuration found",
                           stage=stage_idx + 1, score=f"{best_score:.4f}")
            
            # Early stopping if time is running out
            if time.time() - self.start_time > self.max_time_seconds * 0.8:
                break
        
        return {
            "best_config": best_config,
            "best_score": best_score,
            "search_strategy": "progressive"
        }
    
    def _optuna_search(self, X_train: torch.Tensor, y_train: torch.Tensor,
                     X_val: torch.Tensor, y_val: torch.Tensor,
                     architecture_type: str) -> Dict[str, Any]:
        """Standard Optuna-based search"""
        
        logger.info("🎯 Starting Optuna search")
        
        study = optuna.create_study(
            direction="maximize" if self._higher_is_better() else "minimize",
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        def objective(trial):
            return self._evaluate_architecture(
                trial, X_train, y_train, X_val, y_val,
                architecture_type, {"max_layers": 8, "max_width": 1024}
            )
        
        # Run optimization
        study.optimize(
            objective,
            n_trials=60,
            timeout=self.max_time_seconds
        )
        
        best_config = study.best_params.copy()
        best_config.update({"architecture_type": architecture_type})
        
        return {
            "best_config": best_config,
            "best_score": study.best_value,
            "search_strategy": "optuna"
        }
    
    def _random_search(self, X_train: torch.Tensor, y_train: torch.Tensor,
                     X_val: torch.Tensor, y_val: torch.Tensor,
                     architecture_type: str) -> Dict[str, Any]:
        """Random search baseline"""
        
        logger.info("🎲 Starting random search")
        
        study = optuna.create_study(
            direction="maximize" if self._higher_is_better() else "minimize",
            sampler=optuna.samplers.RandomSampler(seed=42)
        )
        
        def objective(trial):
            return self._evaluate_architecture(
                trial, X_train, y_train, X_val, y_val,
                architecture_type, {"max_layers": 6, "max_width": 512}
            )
        
        study.optimize(
            objective,
            n_trials=50,
            timeout=self.max_time_seconds
        )
        
        best_config = study.best_params.copy()
        best_config.update({"architecture_type": architecture_type})
        
        return {
            "best_config": best_config,
            "best_score": study.best_value,
            "search_strategy": "random"
        }
    
    def _evaluate_architecture(self, trial: optuna.Trial, 
                             X_train: torch.Tensor, y_train: torch.Tensor,
                             X_val: torch.Tensor, y_val: torch.Tensor,
                             architecture_type: str, constraints: Dict[str, Any]) -> float:
        """Evaluate a single architecture configuration"""
        
        try:
            # Generate architecture config
            config = self.search_spaces[architecture_type](trial, constraints)
            
            # Create model
            input_dim = X_train.shape[1]
            output_dim = len(torch.unique(y_train)) if self.task_type == "classification" else 1
            
            model = PyTorchModelFactory.create_model(
                architecture_type, input_dim, output_dim, self.task_type, config
            )
            
            # Train and evaluate
            score = self._train_and_evaluate(model, config, X_train, y_train, X_val, y_val)
            
            return score
            
        except Exception as e:
            logger.warning("Architecture evaluation failed", error=str(e))
            return float('-inf') if self._higher_is_better() else float('inf')
    
    def _train_and_evaluate(self, model: BaseNeuralNet, config: Dict[str, Any],
                          X_train: torch.Tensor, y_train: torch.Tensor,
                          X_val: torch.Tensor, y_val: torch.Tensor) -> float:
        """Train model and return validation score"""
        
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
        
        # Training loop with early stopping
        model.train()
        epochs = config.get("epochs", 50)
        best_val_score = float('-inf') if self._higher_is_better() else float('inf')
        patience_counter = 0
        patience = 10
        
        for epoch in range(epochs):
            # Forward pass
            optimizer.zero_grad()
            outputs = model(X_train)
            
            if self.task_type == "classification":
                loss = loss_fn(outputs, y_train.long())
            else:
                loss = loss_fn(outputs.squeeze(), y_train.float())
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if scheduler:
                scheduler.step()
            
            # Early stopping check every 5 epochs
            if epoch % 5 == 0:
                val_score = self._evaluate_model(model, X_val, y_val)
                
                if self._is_better_score(val_score, best_val_score):
                    best_val_score = val_score
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    break
        
        # Return best validation score
        return best_val_score
    
    def _evaluate_model(self, model: BaseNeuralNet, X_val: torch.Tensor, y_val: torch.Tensor) -> float:
        """Evaluate model on validation set"""
        
        model.eval()
        with torch.no_grad():
            outputs = model(X_val)
            
            if self.task_type == "classification":
                predictions = torch.argmax(outputs, dim=1)
                score = accuracy_score(y_val.cpu().numpy(), predictions.cpu().numpy())
            else:
                predictions = outputs.squeeze()
                score = -mean_squared_error(y_val.cpu().numpy(), predictions.cpu().numpy())  # Negative MSE
        
        return score
    
    def _get_simple_mlp_space(self, trial: optuna.Trial, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Define search space for SimpleMLP"""
        
        max_layers = constraints.get("max_layers", 4)
        max_width = constraints.get("max_width", 512)
        
        # Number of layers
        n_layers = trial.suggest_int("n_layers", 1, max_layers)
        
        # Hidden dimensions
        hidden_dims = []
        for i in range(n_layers):
            dim = trial.suggest_int(f"hidden_dim_{i}", 32, max_width, step=32)
            hidden_dims.append(dim)
        
        return {
            "hidden_dims": hidden_dims,
            "activation": trial.suggest_categorical("activation", ["relu", "gelu", "swish", "tanh"]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "batch_norm": trial.suggest_categorical("batch_norm", [True, False]),
            "optimizer": trial.suggest_categorical("optimizer", ["adam", "adamw", "sgd"]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "epochs": trial.suggest_int("epochs", 20, 80),
            "loss_type": trial.suggest_categorical("loss_type", ["default", "focal", "label_smoothing"]) if self.task_type == "classification" else "default",
            "scheduler": trial.suggest_categorical("scheduler", ["none", "cosine", "step"])
        }
    
    def _get_resnet_mlp_space(self, trial: optuna.Trial, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Define search space for ResNetMLP"""
        
        max_layers = constraints.get("max_layers", 4)
        max_width = constraints.get("max_width", 512)
        
        # Number of layers
        n_layers = trial.suggest_int("n_layers", 2, max_layers)
        
        # Hidden dimensions (ResNet works better with consistent widths)
        base_width = trial.suggest_int("base_width", 64, max_width, step=32)
        hidden_dims = []
        for i in range(n_layers):
            # Allow some variation around base width
            width = trial.suggest_int(f"width_{i}", max(32, base_width - 64), base_width + 64, step=32)
            hidden_dims.append(width)
        
        return {
            "hidden_dims": hidden_dims,
            "activation": trial.suggest_categorical("activation", ["relu", "gelu", "swish"]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "batch_norm": trial.suggest_categorical("batch_norm", [True, False]),
            "optimizer": trial.suggest_categorical("optimizer", ["adam", "adamw"]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "epochs": trial.suggest_int("epochs", 30, 100),
            "loss_type": trial.suggest_categorical("loss_type", ["default", "focal"]) if self.task_type == "classification" else "default",
            "scheduler": trial.suggest_categorical("scheduler", ["none", "cosine"])
        }
    
    def _get_attention_mlp_space(self, trial: optuna.Trial, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Define search space for AttentionMLP"""
        
        max_layers = constraints.get("max_layers", 3)  # Attention is more complex
        max_width = constraints.get("max_width", 256)
        
        # Number of layers
        n_layers = trial.suggest_int("n_layers", 1, max_layers)
        
        # Hidden dimensions
        hidden_dims = []
        for i in range(n_layers):
            # Attention works better with dimensions divisible by num_heads
            dim = trial.suggest_int(f"hidden_dim_{i}", 64, max_width, step=64)
            hidden_dims.append(dim)
        
        num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
        
        # Ensure first dimension is divisible by num_heads
        if hidden_dims and hidden_dims[0] % num_heads != 0:
            hidden_dims[0] = ((hidden_dims[0] // num_heads) + 1) * num_heads
        
        return {
            "hidden_dims": hidden_dims,
            "num_heads": num_heads,
            "activation": trial.suggest_categorical("activation", ["gelu", "swish", "relu"]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.4),
            "optimizer": trial.suggest_categorical("optimizer", ["adamw", "adam"]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "epochs": trial.suggest_int("epochs", 40, 120),
            "loss_type": trial.suggest_categorical("loss_type", ["default", "label_smoothing"]) if self.task_type == "classification" else "default",
            "scheduler": trial.suggest_categorical("scheduler", ["cosine", "reduce_on_plateau"])
        }
    
    def _higher_is_better(self) -> bool:
        """Check if higher scores are better for the given metric"""
        return self.metric in ["accuracy", "precision", "recall", "f1"]
    
    def _is_better_score(self, new_score: float, current_best: float) -> bool:
        """Check if new score is better than current best"""
        if self._higher_is_better():
            return new_score > current_best
        else:
            return new_score < current_best 