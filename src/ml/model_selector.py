"""
LLM-guided model selector with ensemble recommendations and neural architecture search
"""

import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

try:
    import requests
except ImportError:
    requests = None

import openai
import optuna
import structlog
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.linear_model import (
    LogisticRegression, LinearRegression, Ridge, Lasso,
    ElasticNet, SGDClassifier, SGDRegressor
)
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
import lightgbm as lgb
import xgboost as xgb

from ..core.config import settings
from ..models.schema import DataProfile

logger = structlog.get_logger()


@dataclass
class ModelRecommendation:
    """Model recommendation with complexity estimates"""
    model_type: str
    model_family: str  # 'tree_based', 'linear', 'neural', 'ensemble', 'instance_based'
    complexity_score: float  # 1-10 scale
    expected_training_time: str  # 'fast', 'medium', 'slow'
    parameter_estimates: Dict[str, Any]
    reasoning: str
    architectures: List[Dict[str, Any]]  # 3 different configurations
    pros: List[str] = None
    cons: List[str] = None
    best_for: str = ""
    training_time_estimate: str = ""
    memory_usage: str = "medium"  # 'low', 'medium', 'high'
    interpretability: str = "medium"  # 'high', 'medium', 'low'


@dataclass
class EnsembleStrategy:
    """Ensemble strategy recommendation"""
    recommended_models: List[ModelRecommendation]
    ensemble_method: str  # 'voting', 'stacking', 'blending'
    diversity_score: float
    reasoning: str


class NeuralArchitectureSearcher:
    """Neural Architecture Search (NAS) implementation"""
    
    def __init__(self, task_type: str, data_shape: Tuple[int, int]):
        self.task_type = task_type
        self.n_samples, self.n_features = data_shape
        
    def suggest_architectures(self, trial: optuna.Trial = None) -> List[Dict[str, Any]]:
        """Generate 3 diverse neural network architectures"""
        
        architectures = []
        
        # Architecture 1: Simple/Fast
        arch1 = self._suggest_simple_architecture()
        architectures.append({
            "name": "simple_mlp",
            "description": "Fast, simple architecture for quick results",
            "config": arch1,
            "complexity": "low"
        })
        
        # Architecture 2: Balanced
        arch2 = self._suggest_balanced_architecture()
        architectures.append({
            "name": "balanced_mlp",
            "description": "Balanced architecture with moderate complexity",
            "config": arch2,
            "complexity": "medium"
        })
        
        # Architecture 3: Complex/Deep
        arch3 = self._suggest_complex_architecture()
        architectures.append({
            "name": "deep_mlp",
            "description": "Deep architecture for complex patterns",
            "config": arch3,
            "complexity": "high"
        })
        
        return architectures
    
    def _suggest_simple_architecture(self) -> Dict[str, Any]:
        """Simple 1-2 layer architecture"""
        hidden_size = min(128, max(32, self.n_features * 2))
        
        return {
            "hidden_layer_sizes": (hidden_size,),
            "activation": "relu",
            "alpha": 0.001,
            "learning_rate_init": 0.001,
            "max_iter": 300,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "n_iter_no_change": 10
        }
    
    def _suggest_balanced_architecture(self) -> Dict[str, Any]:
        """Balanced 2-3 layer architecture"""
        layer1 = min(256, max(64, self.n_features * 3))
        layer2 = max(32, layer1 // 2)
        
        return {
            "hidden_layer_sizes": (layer1, layer2),
            "activation": "relu",
            "alpha": 0.0001,
            "learning_rate_init": 0.001,
            "max_iter": 500,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "n_iter_no_change": 15
        }
    
    def _suggest_complex_architecture(self) -> Dict[str, Any]:
        """Complex 3-4 layer architecture"""
        layer1 = min(512, max(128, self.n_features * 4))
        layer2 = max(64, layer1 // 2)
        layer3 = max(32, layer2 // 2)
        
        return {
            "hidden_layer_sizes": (layer1, layer2, layer3),
            "activation": "relu",
            "alpha": 0.00001,
            "learning_rate_init": 0.0001,
            "learning_rate": "adaptive",
            "max_iter": 1000,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "n_iter_no_change": 20
        }


class ModelSelector:
    """LLM-guided model selector with ensemble recommendations"""
    
    def __init__(self):
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not configured")
        
        self.client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model_catalog = self._build_model_catalog()
    
    async def recommend_ensemble(
        self,
        data_profile: DataProfile,
        task_type: str,
        target_column: str,
        constraints: Dict[str, Any]
    ) -> EnsembleStrategy:
        """Get LLM recommendations for ensemble model selection"""
        
        logger.info("Requesting model recommendations from LLM", 
                   task_type=task_type, 
                   n_rows=data_profile.n_rows,
                   n_cols=data_profile.n_cols)
        
        # Prepare context for LLM
        context = self._prepare_context(data_profile, task_type, target_column, constraints)
        
        try:
            # Check if requests module is available
            if requests is None:
                logger.warning("Requests module not available, falling back to default recommendations")
                return self._get_fallback_recommendations(data_profile, task_type)
            
            # Get LLM recommendations
            response = await self.client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": context}
                ],
                temperature=0.2,
                max_tokens=2000
            )
            
            response_text = response.choices[0].message.content.strip()
            logger.info("LLM response received", response_length=len(response_text))
            
            # Parse LLM response
            recommendations_data = self._extract_json(response_text)
            
            # Convert to structured recommendations
            model_recommendations = []
            for rec_data in recommendations_data["recommended_models"]:
                # Generate architectures for each model
                architectures = self._generate_model_architectures(
                    rec_data, task_type, (data_profile.n_rows, data_profile.n_cols)
                )
                
                model_rec = ModelRecommendation(
                    model_type=rec_data["model_type"],
                    model_family=rec_data["model_family"],
                    complexity_score=rec_data["complexity_score"],
                    expected_training_time=rec_data["expected_training_time"],
                    parameter_estimates=rec_data["parameter_estimates"],
                    reasoning=rec_data["reasoning"],
                    architectures=architectures,
                    pros=rec_data.get("pros", []),
                    cons=rec_data.get("cons", []),
                    best_for=rec_data.get("best_for", ""),
                    training_time_estimate=rec_data.get("training_time_estimate", ""),
                    memory_usage=rec_data.get("memory_usage", "medium"),
                    interpretability=rec_data.get("interpretability", "medium")
                )
                model_recommendations.append(model_rec)
            
            ensemble_strategy = EnsembleStrategy(
                recommended_models=model_recommendations,
                ensemble_method=recommendations_data["ensemble_strategy"]["method"],
                diversity_score=recommendations_data["ensemble_strategy"]["diversity_score"],
                reasoning=recommendations_data["ensemble_strategy"]["reasoning"]
            )
            
            return ensemble_strategy
            
        except Exception as e:
            logger.error("Failed to get model recommendations", error=str(e))
            # Fallback to default recommendations
            return self._get_fallback_recommendations(data_profile, task_type)
    
    def _get_system_prompt(self) -> str:
        """System prompt for model recommendation"""
        return """You are an expert ML engineer specializing in model selection and ensemble design.

Given a dataset profile and task description, recommend the best ensemble of 3-5 models.

Output ONLY valid JSON matching this schema:

{
  "recommended_models": [
    {
              "model_type": "random_forest" | "xgboost" | "lightgbm" | "neural_network" | "svm" | "logistic_regression" | "linear_regression" | "knn" | "naive_bayes" | "kmeans" | "dbscan" | "hierarchical" | "spectral" | "gaussian_mixture",
        "model_family": "tree_based" | "linear" | "neural" | "ensemble" | "instance_based" | "probabilistic" | "clustering",
      "complexity_score": 1-10,
      "expected_training_time": "fast" | "medium" | "slow",
      "parameter_estimates": {
        "key_params": "values with rough estimates"
      },
      "reasoning": "why this model fits the data/task",
      "pros": ["advantage 1", "advantage 2", "advantage 3"],
      "cons": ["limitation 1", "limitation 2"],
      "best_for": "specific use case description",
      "training_time_estimate": "X minutes",
      "memory_usage": "low" | "medium" | "high",
      "interpretability": "high" | "medium" | "low"
    }
  ],
  "ensemble_strategy": {
    "method": "voting" | "stacking" | "blending",
    "diversity_score": 0.0-1.0,
    "reasoning": "why this ensemble strategy"
  },
  "recommendations_summary": {
    "total_training_time": "estimated total time",
    "recommended_combination": "suggested subset for optimal balance"
  }
}

RULES:
1. Recommend 3-5 diverse models that complement each other
2. Consider data size, feature count, and complexity
3. Balance accuracy potential with training time
4. Ensure model diversity (different families/approaches)
5. For small datasets (<1000 rows): prefer simpler models
6. For large datasets (>2000 rows): can use complex models such as neural networks. In fact, neural networks are the best models for large datasets.
7. For high-dimensional data: consider regularization
8. Provide clear pros/cons for user decision-making
9. Estimate realistic training times
10. ONLY output JSON, no other text"""

    def _prepare_context(
        self, 
        data_profile: DataProfile, 
        task_type: str, 
        target_column: str,
        constraints: Dict[str, Any]
    ) -> str:
        """Prepare context for LLM"""
        
        # Analyze target column
        target_info = data_profile.columns.get(target_column, {})
        
        # Count feature types
        numeric_features = 0
        categorical_features = 0
        high_cardinality_features = 0
        
        for col, col_info in data_profile.columns.items():
            if col == target_column:
                continue
            if col_info["dtype"] in ["int64", "float64"]:
                numeric_features += 1
            elif col_info["dtype"] == "object":
                if col_info["n_unique"] > 100:
                    high_cardinality_features += 1
                else:
                    categorical_features += 1
        
        # Identify potential challenges
        challenges = []
        if data_profile.n_rows < 1000:
            challenges.append("small dataset")
        if data_profile.n_cols > data_profile.n_rows:
            challenges.append("high dimensional")
        if high_cardinality_features > 0:
            challenges.append("high cardinality categoricals")
        if len(data_profile.issues) > 0:
            challenges.extend(data_profile.issues[:3])
        
        context = f"""Dataset Profile:
- Task: {task_type}
- Target: {target_column} (type: {target_info.get('dtype', 'unknown')})
- Rows: {data_profile.n_rows:,}
- Features: {data_profile.n_cols} total ({numeric_features} numeric, {categorical_features} categorical, {high_cardinality_features} high-cardinality)
- Challenges: {', '.join(challenges) if challenges else 'none detected'}

Target Analysis:
- Unique values: {target_info.get('n_unique', 'unknown')}
- Null fraction: {target_info.get('fraction_null', 0):.2%}

Constraints:
- Feature limit: {constraints.get('feature_limit', 'none')}
- Time budget: {constraints.get('time_budget', 'medium')}
- Excluded columns: {constraints.get('exclude_cols', [])}

Recommend an ensemble of 3-5 diverse models optimized for this dataset."""
        
        return context
    
    def _generate_model_architectures(
        self, 
        rec_data: Dict[str, Any], 
        task_type: str,
        data_shape: Tuple[int, int]
    ) -> List[Dict[str, Any]]:
        """Generate 3 different architectures for a recommended model"""
        
        model_type = rec_data["model_type"]
        complexity = rec_data["complexity_score"]
        
        architectures = []
        
        if model_type == "neural_network":
            # Use NAS for neural networks
            nas = NeuralArchitectureSearcher(task_type, data_shape)
            architectures = nas.suggest_architectures()
            
        elif model_type in ["random_forest", "extra_trees"]:
            # Tree-based ensemble architectures
            architectures = [
                {
                    "name": "fast_forest",
                    "description": "Fast, shallow forest",
                    "config": {"n_estimators": 50, "max_depth": 8, "min_samples_split": 10},
                    "complexity": "low"
                },
                {
                    "name": "balanced_forest", 
                    "description": "Balanced forest configuration",
                    "config": {"n_estimators": 100, "max_depth": 15, "min_samples_split": 5},
                    "complexity": "medium"
                },
                {
                    "name": "deep_forest",
                    "description": "Deep, complex forest",
                    "config": {"n_estimators": 200, "max_depth": None, "min_samples_split": 2},
                    "complexity": "high"
                }
            ]
            
        elif model_type in ["xgboost", "lightgbm"]:
            # Gradient boosting architectures
            architectures = [
                {
                    "name": "fast_boost",
                    "description": "Fast boosting with early stopping",
                    "config": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 4},
                    "complexity": "low"
                },
                {
                    "name": "balanced_boost",
                    "description": "Balanced boosting configuration",
                    "config": {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 6},
                    "complexity": "medium"
                },
                {
                    "name": "precise_boost",
                    "description": "Slow but precise boosting",
                    "config": {"n_estimators": 500, "learning_rate": 0.01, "max_depth": 8},
                    "complexity": "high"
                }
            ]
            
        elif model_type in ["svm"]:
            # SVM architectures
            architectures = [
                {
                    "name": "linear_svm",
                    "description": "Linear SVM for speed",
                    "config": {"kernel": "linear", "C": 1.0},
                    "complexity": "low"
                },
                {
                    "name": "rbf_svm",
                    "description": "RBF kernel for non-linearity",
                    "config": {"kernel": "rbf", "C": 1.0, "gamma": "scale"},
                    "complexity": "medium"
                },
                {
                    "name": "poly_svm",
                    "description": "Polynomial kernel for complex patterns",
                    "config": {"kernel": "poly", "degree": 3, "C": 1.0},
                    "complexity": "high"
                }
            ]
            
        elif model_type == "kmeans":
            # K-means clustering architectures
            architectures = [
                {
                    "name": "simple_kmeans",
                    "description": "Simple K-means clustering",
                    "config": {"n_clusters": 3, "init": "k-means++", "n_init": 10},
                    "complexity": "low"
                },
                {
                    "name": "optimized_kmeans",
                    "description": "Optimized K-means with better initialization",
                    "config": {"n_clusters": 5, "init": "k-means++", "n_init": 20, "algorithm": "elkan"},
                    "complexity": "medium"
                },
                {
                    "name": "robust_kmeans",
                    "description": "Robust K-means with multiple restarts",
                    "config": {"n_clusters": 8, "init": "k-means++", "n_init": 50, "algorithm": "elkan"},
                    "complexity": "high"
                }
            ]
            
        elif model_type == "dbscan":
            # DBSCAN clustering architectures
            architectures = [
                {
                    "name": "loose_dbscan",
                    "description": "Loose clustering for broader groups",
                    "config": {"eps": 0.5, "min_samples": 5},
                    "complexity": "low"
                },
                {
                    "name": "balanced_dbscan",
                    "description": "Balanced density clustering",
                    "config": {"eps": 0.3, "min_samples": 10},
                    "complexity": "medium"
                },
                {
                    "name": "tight_dbscan",
                    "description": "Tight clustering for dense groups",
                    "config": {"eps": 0.1, "min_samples": 20},
                    "complexity": "high"
                }
            ]
            
        elif model_type == "hierarchical":
            # Hierarchical clustering architectures
            architectures = [
                {
                    "name": "ward_hierarchical",
                    "description": "Ward linkage for compact clusters",
                    "config": {"n_clusters": 3, "linkage": "ward"},
                    "complexity": "low"
                },
                {
                    "name": "complete_hierarchical",
                    "description": "Complete linkage for balanced clusters",
                    "config": {"n_clusters": 5, "linkage": "complete"},
                    "complexity": "medium"
                },
                {
                    "name": "average_hierarchical",
                    "description": "Average linkage for robust clustering",
                    "config": {"n_clusters": 8, "linkage": "average"},
                    "complexity": "high"
                }
            ]
            
        elif model_type == "gaussian_mixture":
            # Gaussian Mixture Model architectures
            architectures = [
                {
                    "name": "simple_gmm",
                    "description": "Simple Gaussian Mixture Model",
                    "config": {"n_components": 3, "covariance_type": "full"},
                    "complexity": "low"
                },
                {
                    "name": "diagonal_gmm",
                    "description": "Diagonal covariance GMM",
                    "config": {"n_components": 5, "covariance_type": "diag"},
                    "complexity": "medium"
                },
                {
                    "name": "tied_gmm",
                    "description": "Tied covariance GMM",
                    "config": {"n_components": 8, "covariance_type": "tied"},
                    "complexity": "high"
                }
            ]
            
        else:
            # Default simple architectures for other models
            architectures = [
                {
                    "name": f"simple_{model_type}",
                    "description": f"Simple {model_type} configuration",
                    "config": rec_data["parameter_estimates"],
                    "complexity": "low"
                },
                {
                    "name": f"tuned_{model_type}",
                    "description": f"Tuned {model_type} configuration", 
                    "config": rec_data["parameter_estimates"],
                    "complexity": "medium"
                },
                {
                    "name": f"complex_{model_type}",
                    "description": f"Complex {model_type} configuration",
                    "config": rec_data["parameter_estimates"],
                    "complexity": "high"
                }
            ]
        
        return architectures
    
    def _extract_json(self, response_text: str) -> Dict[str, Any]:
        """Extract JSON from LLM response"""
        try:
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            
            if start == -1 or end == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response_text[start:end]
            return json.loads(json_str)
            
        except json.JSONDecodeError as e:
            logger.error("Failed to decode JSON", response=response_text, error=str(e))
            raise ValueError(f"Invalid JSON in response: {str(e)}")
    
    def _get_fallback_recommendations(
        self, 
        data_profile: DataProfile, 
        task_type: str
    ) -> EnsembleStrategy:
        """Fallback recommendations if LLM fails"""
        
        fallback_models = []
        
        # Simple rule-based fallback
        if task_type == "clustering":
            # Clustering models
            if data_profile.n_rows < 1000:
                models = ["kmeans", "hierarchical", "dbscan"]
            else:
                models = ["kmeans", "dbscan", "gaussian_mixture", "hierarchical"]
        elif data_profile.n_rows < 1000:
            # Small dataset - use simple, robust models
            models = ["logistic_regression", "random_forest"] if task_type == "classification" else ["linear_regression", "random_forest"]
        elif data_profile.n_rows < 5000:
            # Medium dataset - balanced approach with proven models
            models = ["logistic_regression", "random_forest", "xgboost"] if task_type == "classification" else ["linear_regression", "random_forest", "xgboost"]
        else:
            # Large dataset - can use more complex models but keep it balanced
            models = ["random_forest", "xgboost", "lightgbm"] if task_type == "classification" else ["random_forest", "xgboost", "lightgbm"]
        
        for model_type in models:
            fallback_models.append(ModelRecommendation(
                model_type=model_type,
                model_family=self._get_model_family(model_type),
                complexity_score=5.0,
                expected_training_time="medium",
                parameter_estimates={},
                reasoning="Fallback recommendation",
                architectures=self._generate_model_architectures(
                    {"model_type": model_type, "complexity_score": 5.0, "parameter_estimates": {}},
                    task_type,
                    (data_profile.n_rows, data_profile.n_cols)
                )
            ))
        
        return EnsembleStrategy(
            recommended_models=fallback_models,
            ensemble_method="voting",
            diversity_score=0.7,
            reasoning="Fallback ensemble strategy"
        )
    
    def _get_model_family(self, model_type: str) -> str:
        """Get model family for a model type"""
        family_map = {
            "random_forest": "tree_based",
            "extra_trees": "tree_based", 
            "xgboost": "tree_based",
            "lightgbm": "tree_based",
            "neural_network": "neural",
            "svm": "instance_based",
            "logistic_regression": "linear",
            "linear_regression": "linear",
            "knn": "instance_based",
            "naive_bayes": "probabilistic",
            "kmeans": "clustering",
            "dbscan": "clustering",
            "hierarchical": "clustering",
            "spectral": "clustering",
            "gaussian_mixture": "clustering"
        }
        return family_map.get(model_type, "unknown")
    
    def _build_model_catalog(self) -> Dict[str, Any]:
        """Build catalog of available models"""
        return {
            "tree_based": {
                "random_forest": {"class": RandomForestClassifier, "regressor": RandomForestRegressor},
                "extra_trees": {"class": ExtraTreesClassifier, "regressor": ExtraTreesRegressor},
                "xgboost": {"class": xgb.XGBClassifier, "regressor": xgb.XGBRegressor},
                "lightgbm": {"class": lgb.LGBMClassifier, "regressor": lgb.LGBMRegressor}
            },
            "linear": {
                "logistic_regression": {"class": LogisticRegression, "regressor": LinearRegression},
                "ridge": {"class": LogisticRegression, "regressor": Ridge},
                "lasso": {"class": LogisticRegression, "regressor": Lasso}
            },
            "neural": {
                "neural_network": {"class": MLPClassifier, "regressor": MLPRegressor}
            },
            "instance_based": {
                "knn": {"class": KNeighborsClassifier, "regressor": KNeighborsRegressor},
                "svm": {"class": SVC, "regressor": SVR}
            },
            "probabilistic": {
                "naive_bayes": {"class": GaussianNB, "regressor": None}
            },
            "clustering": {
                "kmeans": {"class": KMeans, "regressor": None},
                "dbscan": {"class": DBSCAN, "regressor": None},
                "hierarchical": {"class": AgglomerativeClustering, "regressor": None},
                "spectral": {"class": SpectralClustering, "regressor": None},
                "gaussian_mixture": {"class": GaussianMixture, "regressor": None}
            }
        } 