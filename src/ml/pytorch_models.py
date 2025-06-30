"""
PyTorch neural network models with Neural Architecture Search support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from abc import ABC, abstractmethod


class BaseNeuralNet(nn.Module, ABC):
    """Base class for all neural network architectures"""
    
    def __init__(self, input_dim: int, output_dim: int, task_type: str):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.task_type = task_type  # 'classification' or 'regression'
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
    
    def get_loss_function(self, loss_type: str = "default"):
        """Get appropriate loss function"""
        if self.task_type == "classification":
            if loss_type == "focal":
                return FocalLoss()
            elif loss_type == "label_smoothing":
                return nn.CrossEntropyLoss(label_smoothing=0.1)
            else:  # default
                return nn.CrossEntropyLoss()
        else:  # regression
            if loss_type == "huber":
                return nn.HuberLoss()
            elif loss_type == "l1":
                return nn.L1Loss()
            else:  # default - mse
                return nn.MSELoss()


class SimpleMLP(BaseNeuralNet):
    """Simple Multi-Layer Perceptron"""
    
    def __init__(self, input_dim: int, output_dim: int, task_type: str, 
                 hidden_dims: List[int], activation: str = "relu", 
                 dropout: float = 0.1, batch_norm: bool = True):
        super().__init__(input_dim, output_dim, task_type)
        
        self.activation = self._get_activation(activation)
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self.activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def _get_activation(self, activation: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "swish": nn.SiLU(),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(0.1),
            "elu": nn.ELU()
        }
        return activations.get(activation, nn.ReLU())


class ResNetMLP(BaseNeuralNet):
    """MLP with ResNet-style skip connections"""
    
    def __init__(self, input_dim: int, output_dim: int, task_type: str,
                 hidden_dims: List[int], activation: str = "relu",
                 dropout: float = 0.1, batch_norm: bool = True):
        super().__init__(input_dim, output_dim, task_type)
        
        self.activation = self._get_activation(activation)
        
        # Input projection if needed
        self.input_proj = None
        if input_dim != hidden_dims[0]:
            self.input_proj = nn.Linear(input_dim, hidden_dims[0])
        
        # ResNet blocks
        self.blocks = nn.ModuleList()
        for i in range(len(hidden_dims)):
            if i == 0:
                in_dim = hidden_dims[0]
            else:
                in_dim = hidden_dims[i-1]
            out_dim = hidden_dims[i]
            
            self.blocks.append(ResNetBlock(
                in_dim, out_dim, activation, dropout, batch_norm
            ))
        
        # Output layer
        self.output = nn.Linear(hidden_dims[-1], output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input projection
        if self.input_proj:
            x = self.input_proj(x)
            x = self.activation(x)
        
        # ResNet blocks
        for block in self.blocks:
            x = block(x)
        
        # Output
        return self.output(x)
    
    def _get_activation(self, activation: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "swish": nn.SiLU(),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(0.1),
            "elu": nn.ELU()
        }
        return activations.get(activation, nn.ReLU())


class AttentionMLP(BaseNeuralNet):
    """MLP with self-attention mechanism"""
    
    def __init__(self, input_dim: int, output_dim: int, task_type: str,
                 hidden_dims: List[int], num_heads: int = 4,
                 activation: str = "relu", dropout: float = 0.1):
        super().__init__(input_dim, output_dim, task_type)
        
        self.activation = self._get_activation(activation)
        
        # Project input to first hidden dimension
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dims[0],
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward layers
        ff_layers = []
        for i in range(1, len(hidden_dims)):
            ff_layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            ff_layers.append(self.activation)
            if dropout > 0:
                ff_layers.append(nn.Dropout(dropout))
        
        self.feed_forward = nn.Sequential(*ff_layers)
        
        # Output layer
        self.output = nn.Linear(hidden_dims[-1], output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project input
        x = self.input_proj(x)
        x = self.activation(x)
        
        # Add sequence dimension for attention (batch_size, seq_len=1, features)
        x = x.unsqueeze(1)
        
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        attn_out = attn_out.squeeze(1)  # Remove sequence dimension
        
        # Residual connection
        x = x.squeeze(1) + attn_out
        
        # Feed-forward
        x = self.feed_forward(x)
        
        # Output
        return self.output(x)
    
    def _get_activation(self, activation: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "swish": nn.SiLU(),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(0.1),
            "elu": nn.ELU()
        }
        return activations.get(activation, nn.ReLU())


class ResNetBlock(nn.Module):
    """Residual block for ResNetMLP"""
    
    def __init__(self, in_dim: int, out_dim: int, activation: str = "relu",
                 dropout: float = 0.1, batch_norm: bool = True):
        super().__init__()
        
        self.activation = self._get_activation(activation)
        
        # Main path
        layers = [nn.Linear(in_dim, out_dim)]
        if batch_norm:
            layers.append(nn.BatchNorm1d(out_dim))
        layers.append(self.activation)
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(out_dim, out_dim))
        if batch_norm:
            layers.append(nn.BatchNorm1d(out_dim))
        
        self.main_path = nn.Sequential(*layers)
        
        # Skip connection
        self.skip_connection = None
        if in_dim != out_dim:
            self.skip_connection = nn.Linear(in_dim, out_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        # Main path
        out = self.main_path(x)
        
        # Skip connection
        if self.skip_connection:
            residual = self.skip_connection(residual)
        
        # Add residual and apply activation
        out = out + residual
        return self.activation(out)
    
    def _get_activation(self, activation: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "swish": nn.SiLU(),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(0.1),
            "elu": nn.ELU()
        }
        return activations.get(activation, nn.ReLU())


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class PyTorchModelFactory:
    """Factory for creating PyTorch models based on architecture specifications"""
    
    @staticmethod
    def create_model(architecture: str, input_dim: int, output_dim: int, 
                    task_type: str, config: Dict[str, Any]) -> BaseNeuralNet:
        """Create a model based on architecture type and configuration"""
        
        if architecture == "simple_mlp":
            return SimpleMLP(
                input_dim=input_dim,
                output_dim=output_dim,
                task_type=task_type,
                hidden_dims=config.get("hidden_dims", [128, 64]),
                activation=config.get("activation", "relu"),
                dropout=config.get("dropout", 0.1),
                batch_norm=config.get("batch_norm", True)
            )
        
        elif architecture == "resnet_mlp":
            return ResNetMLP(
                input_dim=input_dim,
                output_dim=output_dim,
                task_type=task_type,
                hidden_dims=config.get("hidden_dims", [128, 128, 64]),
                activation=config.get("activation", "relu"),
                dropout=config.get("dropout", 0.1),
                batch_norm=config.get("batch_norm", True)
            )
        
        elif architecture == "attention_mlp":
            return AttentionMLP(
                input_dim=input_dim,
                output_dim=output_dim,
                task_type=task_type,
                hidden_dims=config.get("hidden_dims", [128, 64]),
                num_heads=config.get("num_heads", 4),
                activation=config.get("activation", "gelu"),
                dropout=config.get("dropout", 0.1)
            )
        
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
    
    @staticmethod
    def get_optimizer(model: nn.Module, optimizer_type: str, 
                     learning_rate: float, **kwargs) -> torch.optim.Optimizer:
        """Get optimizer for the model"""
        
        if optimizer_type == "adam":
            return torch.optim.Adam(
                model.parameters(), 
                lr=learning_rate,
                weight_decay=kwargs.get("weight_decay", 1e-5)
            )
        elif optimizer_type == "adamw":
            return torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=kwargs.get("weight_decay", 1e-2)
            )
        elif optimizer_type == "sgd":
            return torch.optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=kwargs.get("momentum", 0.9),
                weight_decay=kwargs.get("weight_decay", 1e-4)
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    @staticmethod
    def get_scheduler(optimizer: torch.optim.Optimizer, scheduler_type: str, 
                     **kwargs) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Get learning rate scheduler"""
        
        if scheduler_type == "none":
            return None
        elif scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=kwargs.get("T_max", 100)
            )
        elif scheduler_type == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer, 
                step_size=kwargs.get("step_size", 30),
                gamma=kwargs.get("gamma", 0.1)
            )
        elif scheduler_type == "reduce_on_plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}") 