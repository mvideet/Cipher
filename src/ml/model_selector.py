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
    ExtraTreesClassifier, ExtraTreesRegressor,
    IsolationForest, VotingClassifier, VotingRegressor,
    StackingClassifier, StackingRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor
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

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

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
                max_tokens=3000  # Increased for more comprehensive recommendations
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
        """Enhanced system prompt for comprehensive model recommendation including time series"""
        return """You are an expert ML engineer and data scientist with deep expertise in model selection, ensemble design, and practical ML deployment, including specialized knowledge in time series forecasting.

Your role is to analyze dataset characteristics and business requirements to recommend the optimal combination of 3-5 machine learning models that will work together in an ensemble.

Consider these factors when making recommendations:
1. **Dataset Size & Complexity**: Small (<1000), Medium (1K-10K), Large (>10K) datasets require different approaches
2. **Feature Characteristics**: Numeric vs categorical ratios, missing data patterns, feature interactions
3. **Business Context**: Time constraints, interpretability needs, deployment requirements
4. **Model Diversity**: Ensure complementary strengths (linear + non-linear, simple + complex, different biases)
5. **Training Efficiency**: Balance accuracy potential with computational costs
6. **Ensemble Synergy**: Models should have different error patterns to benefit from voting/stacking
7. **Time Series Specifics**: For temporal data, consider seasonality, trends, stationarity, and forecast horizon

OUTPUT FORMAT: Respond with ONLY valid JSON matching this schema:

{
  "recommended_models": [
    {
      "model_type": "random_forest" | "xgboost" | "lightgbm" | "catboost" | "hist_gradient_boosting" | "neural_network" | "svm" | "logistic_regression" | "linear_regression" | "knn" | "naive_bayes" | "isolation_forest" | "voting" | "stacking" | "kmeans" | "dbscan" | "hierarchical" | "spectral" | "gaussian_mixture" | "arima" | "prophet" | "exponential_smoothing" | "lstm_ts" | "seasonal_decompose",
      "model_family": "tree_based" | "linear" | "neural" | "ensemble" | "instance_based" | "probabilistic" | "anomaly_detection" | "clustering" | "time_series",
      "complexity_score": 1-10,
      "expected_training_time": "fast" | "medium" | "slow",
      "parameter_estimates": {
        "key_hyperparameters": "realistic estimates based on data size and complexity"
      },
      "reasoning": "Detailed explanation of why this model fits the specific dataset and task requirements",
      "pros": ["specific advantage 1", "specific advantage 2", "specific advantage 3"],
      "cons": ["specific limitation 1", "specific limitation 2"],
      "best_for": "Specific use case where this model excels given the data characteristics",
      "training_time_estimate": "X minutes/hours based on data size",
      "memory_usage": "low" | "medium" | "high",
      "interpretability": "high" | "medium" | "low",
      "handles_missing_data": true | false,
      "handles_categorical": true | false,
      "overfitting_risk": "low" | "medium" | "high"
    }
  ],
  "ensemble_strategy": {
    "method": "voting" | "stacking" | "blending" | "averaging",
    "diversity_score": 0.0-1.0,
    "reasoning": "Why this specific ensemble strategy works best for these models and data characteristics",
    "expected_improvement": "Quantitative estimate of ensemble benefit over individual models"
  },
  "recommendations_summary": {
    "total_training_time": "Realistic estimate including hyperparameter tuning",
    "recommended_combination": "Specific advice on which 2-3 models to prioritize if time/resources are limited",
    "deployment_considerations": "Practical notes about model serving and monitoring",
    "data_preprocessing_notes": "Specific preprocessing recommendations for optimal performance"
  }
}

ADVANCED GUIDELINES:

**For Small Datasets (<1000 rows):**
- Prefer regularized linear models, simple tree ensembles
- Avoid deep neural networks (high overfitting risk)
- Consider Naive Bayes for text/categorical features
- Use cross-validation aware estimates

**For Medium Datasets (1K-10K rows):**
- Balanced approach: tree methods + linear + possibly neural
- XGBoost/LightGBM with careful regularization
- SVM with appropriate kernels
- Consider feature engineering complexity

**For Large Datasets (>10K rows):**
- Neural networks become viable and recommended
- Complex tree ensembles (Random Forest + Gradient Boosting)
- Deep models for complex pattern recognition
- Scalability becomes important

**For Time Series Forecasting:**
- **CRITICAL: For forecasting tasks, ONLY recommend time series models from this list:**
  - arima, prophet, exponential_smoothing, lstm_ts, seasonal_decompose
  - **DO NOT recommend** tree-based, linear, neural, or other non-time-series models
  - **model_family MUST be "time_series" for ALL forecasting recommendations**

- **Data Length Considerations**:
  - <100 points: exponential_smoothing, arima (simple)
  - 100-1000 points: arima, prophet, exponential_smoothing
  - >1000 points: prophet, lstm_ts, seasonal_decompose, arima
  
- **Seasonality & Patterns**:
  - Strong seasonality: prophet, seasonal_decompose, arima (seasonal)
  - Trend-dominated: prophet, exponential_smoothing, arima
  - Irregular patterns: lstm_ts, prophet
  - Multiple seasonalities: prophet, seasonal_decompose
  
- **Forecast Horizon**:
  - Short-term (1-7 steps): arima, exponential_smoothing
  - Medium-term (1-4 weeks): prophet, arima
  - Long-term (months/years): prophet, lstm_ts
  
- **Data Frequency**:
  - Daily: prophet, arima, seasonal_decompose
  - Weekly/Monthly: arima, exponential_smoothing, prophet
  - Hourly: lstm_ts, prophet
  
- **Business Requirements**:
  - Interpretability needed: arima, exponential_smoothing, prophet
  - Automatic seasonality: prophet
  - Complex non-linear patterns: lstm_ts, prophet
  - Uncertainty quantification: prophet, arima

**Model Synergy Rules:**
- **For forecasting tasks**: ONLY use time series models (arima, prophet, exponential_smoothing, lstm_ts, seasonal_decompose)
- **For forecasting ensembles**: combine trend-following (prophet) + statistical (arima) + simple baseline (exponential_smoothing)
- **For non-forecasting tasks**: Always include one fast, interpretable baseline (linear/tree)
- **For non-forecasting tasks**: Mix model families for diversity (linear + tree + neural)
- Consider feature preprocessing differences (scaling sensitive vs not)
- Balance interpretable vs black-box models

**Business Context Integration:**
- **For forecasting tasks**: High interpretability needs → arima, exponential_smoothing, prophet
- **For forecasting tasks**: Real-time inference → exponential_smoothing, arima (simple)
- **For forecasting tasks**: Batch prediction → prophet, lstm_ts, arima (complex)
- **For forecasting tasks**: Missing data handling → prophet (handles gaps automatically)
- **For forecasting tasks**: Seasonal business patterns → prophet, seasonal_decompose, arima (seasonal)
- **For non-forecasting tasks**: High interpretability needs → linear, tree-based models
- **For non-forecasting tasks**: Real-time inference → linear, small trees
- **For non-forecasting tasks**: Missing data handling → tree-based over linear methods

Provide specific, actionable recommendations tailored to the exact dataset characteristics provided."""

    def _prepare_context(
        self, 
        data_profile: DataProfile, 
        task_type: str, 
        target_column: str,
        constraints: Dict[str, Any]
    ) -> str:
        """Enhanced context preparation with comprehensive dataset analysis"""
        
        # Analyze target column in detail
        target_info = data_profile.columns.get(target_column, {})
        
        # Count and categorize feature types
        numeric_features = 0
        categorical_features = 0
        high_cardinality_features = 0
        missing_data_features = 0
        skewed_features = 0
        
        feature_analysis = {
            "numeric": [],
            "categorical_low": [],
            "categorical_high": [],
            "high_missing": [],
            "skewed": []
        }
        
        for col, col_info in data_profile.columns.items():
            if col == target_column:
                continue
                
            dtype = col_info.get("dtype", "")
            n_unique = col_info.get("n_unique", 0)
            null_fraction = col_info.get("fraction_null", 0)
            
            if dtype in ["int64", "float64"]:
                numeric_features += 1
                feature_analysis["numeric"].append(col)
                # Check for skewness if available
                if null_fraction > 0.3:
                    feature_analysis["high_missing"].append(col)
            elif dtype == "object":
                if n_unique > 100:
                    high_cardinality_features += 1
                    feature_analysis["categorical_high"].append(col)
                else:
                    categorical_features += 1
                    feature_analysis["categorical_low"].append(col)
                
                if null_fraction > 0.3:
                    feature_analysis["high_missing"].append(col)
            
            if null_fraction > 0.2:
                missing_data_features += 1
        
        # Determine dataset size category
        if data_profile.n_rows < 1000:
            size_category = "Small"
            size_implications = "Prefer simpler models to avoid overfitting. Limited data for complex patterns."
        elif data_profile.n_rows < 10000:
            size_category = "Medium" 
            size_implications = "Good balance - can use moderate complexity models with proper validation."
        else:
            size_category = "Large"
            size_implications = "Can leverage complex models including neural networks. Computational efficiency matters."
        
        # Analyze potential challenges and opportunities
        challenges = []
        opportunities = []
        
        if missing_data_features > 0:
            challenges.append(f"Missing data in {missing_data_features} features ({missing_data_features/len(data_profile.columns)*100:.1f}%)")
        
        if high_cardinality_features > 0:
            challenges.append(f"High cardinality categoricals: {len(feature_analysis['categorical_high'])} features")
        
        if data_profile.n_cols > data_profile.n_rows:
            challenges.append("High dimensional data (more features than samples)")
        
        if len(data_profile.issues) > 0:
            challenges.extend(data_profile.issues[:3])
        
        # Identify opportunities
        if numeric_features > categorical_features:
            opportunities.append("Numeric-heavy dataset - good for linear models and neural networks")
        
        if categorical_features > 0:
            opportunities.append("Mixed data types - tree-based models will handle well")
        
        if data_profile.n_rows > 5000:
            opportunities.append("Sufficient data for ensemble methods and hyperparameter tuning")
        
        # Target analysis for task-specific insights
        target_analysis = ""
        if target_info:
            unique_targets = target_info.get('n_unique', 'unknown')
            null_fraction = target_info.get('fraction_null', 0)
            
            if task_type == "classification":
                if unique_targets == 2:
                    target_analysis = f"Binary classification task. Target balance should be considered for metric selection."
                elif unique_targets <= 10:
                    target_analysis = f"Multi-class classification with {unique_targets} classes. Class imbalance may be a factor."
                else:
                    target_analysis = f"Multi-class problem with {unique_targets} classes - consider hierarchical approaches."
            else:
                target_analysis = f"Regression task. Target has {unique_targets} unique values."
            
            if null_fraction > 0:
                target_analysis += f" Note: {null_fraction*100:.1f}% missing target values."
        
        # Time and resource constraints analysis
        constraint_analysis = ""
        time_budget = constraints.get('time_budget', 'medium')
        feature_limit = constraints.get('feature_limit')
        
        if time_budget == 'fast':
            constraint_analysis = "Fast turnaround required - prioritize quick-training models (linear, simple trees)."
        elif time_budget == 'slow':
            constraint_analysis = "Comprehensive analysis - can use complex models with extensive hyperparameter tuning."
        
        if feature_limit:
            constraint_analysis += f" Feature selection needed (limit: {feature_limit})."
        
        # Business domain context
        domain_context = ""
        if any(indicator in target_column.lower() for indicator in ["price", "revenue", "sales", "cost"]):
            domain_context = "Financial prediction - interpretability and robustness to outliers important."
        elif any(indicator in target_column.lower() for indicator in ["risk", "fraud", "default"]):
            domain_context = "Risk assessment - precision/recall balance critical, regulatory compliance may require interpretability."
        elif any(indicator in target_column.lower() for indicator in ["churn", "conversion", "retention"]):
            domain_context = "Customer behavior - understanding feature importance crucial for actionable insights."
        
        context = f"""DATASET ANALYSIS REPORT

**Basic Statistics:**
- Task Type: {task_type.title()}
- Target Variable: {target_column}
- Dataset Size: {data_profile.n_rows:,} rows × {data_profile.n_cols} columns ({size_category})
- Size Implications: {size_implications}

**Feature Composition:**
- Numeric Features: {numeric_features} ({numeric_features/data_profile.n_cols*100:.1f}%)
- Categorical Features: {categorical_features} ({categorical_features/data_profile.n_cols*100:.1f}%)
- High-Cardinality Categoricals: {high_cardinality_features}
- Features with >20% Missing Data: {missing_data_features}

**Target Variable Analysis:**
{target_analysis}

**Data Quality & Challenges:**
{chr(10).join(['- ' + challenge for challenge in challenges]) if challenges else '- No major data quality issues detected'}

**Opportunities:**
{chr(10).join(['- ' + opp for opp in opportunities]) if opportunities else '- Standard dataset suitable for most algorithms'}

**Business Context:**
{domain_context if domain_context else '- General ML task - standard model selection applies'}

**Resource Constraints:**
- Time Budget: {time_budget.title()}
- {constraint_analysis}
- Excluded Features: {constraints.get('exclude_cols', [])}

**Feature Details:**
- Numeric: {', '.join(feature_analysis['numeric'][:10])}{'...' if len(feature_analysis['numeric']) > 10 else ''}
- Low-Card Categorical: {', '.join(feature_analysis['categorical_low'][:5])}{'...' if len(feature_analysis['categorical_low']) > 5 else ''}
- High-Card Categorical: {', '.join(feature_analysis['categorical_high'][:3])}{'...' if len(feature_analysis['categorical_high']) > 3 else ''}

REQUIREMENTS:
Based on this comprehensive analysis, recommend an optimal ensemble of 3-5 diverse models that will:
1. Handle the specific data characteristics effectively
2. Provide complementary strengths and error patterns
3. Meet the time/resource constraints
4. Deliver actionable insights for the business context
5. Balance accuracy with interpretability needs

Consider the dataset size, feature types, missing data patterns, and business domain when selecting models and ensemble strategy."""
        
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
            
        elif model_type in ["xgboost", "lightgbm", "catboost"]:
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
            
        elif model_type == "hist_gradient_boosting":
            # Histogram-based gradient boosting architectures
            architectures = [
                {
                    "name": "fast_hist_boost",
                    "description": "Fast histogram-based boosting",
                    "config": {"max_iter": 100, "learning_rate": 0.1, "max_depth": 4},
                    "complexity": "low"
                },
                {
                    "name": "balanced_hist_boost",
                    "description": "Balanced histogram boosting",
                    "config": {"max_iter": 300, "learning_rate": 0.05, "max_depth": 8},
                    "complexity": "medium"
                },
                {
                    "name": "precise_hist_boost",
                    "description": "Precise histogram boosting",
                    "config": {"max_iter": 500, "learning_rate": 0.01, "max_depth": 12},
                    "complexity": "high"
                }
            ]
            
        elif model_type == "isolation_forest":
            # Isolation Forest architectures
            architectures = [
                {
                    "name": "fast_isolation",
                    "description": "Fast anomaly detection",
                    "config": {"n_estimators": 50, "max_samples": 0.5, "contamination": 0.1},
                    "complexity": "low"
                },
                {
                    "name": "balanced_isolation",
                    "description": "Balanced anomaly detection",
                    "config": {"n_estimators": 100, "max_samples": 0.8, "contamination": 0.05},
                    "complexity": "medium"
                },
                {
                    "name": "thorough_isolation",
                    "description": "Thorough anomaly detection",
                    "config": {"n_estimators": 200, "max_samples": 1.0, "contamination": 0.01},
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
            
        elif model_type == "arima":
            # ARIMA time series architectures
            architectures = [
                {
                    "name": "simple_arima",
                    "description": "Simple ARIMA model with auto-selection",
                    "config": {"seasonal": False, "max_p": 3, "max_d": 2, "max_q": 3},
                    "complexity": "low"
                },
                {
                    "name": "seasonal_arima",
                    "description": "Seasonal ARIMA with moderate complexity",
                    "config": {"seasonal": True, "max_p": 5, "max_d": 2, "max_q": 5},
                    "complexity": "medium"
                },
                {
                    "name": "comprehensive_arima",
                    "description": "Comprehensive ARIMA with extensive search",
                    "config": {"seasonal": True, "max_p": 8, "max_d": 3, "max_q": 8},
                    "complexity": "high"
                }
            ]
            
        elif model_type == "prophet":
            # Prophet time series architectures
            architectures = [
                {
                    "name": "basic_prophet",
                    "description": "Basic Prophet with default seasonality",
                    "config": {"yearly_seasonality": True, "weekly_seasonality": False, "daily_seasonality": False},
                    "complexity": "low"
                },
                {
                    "name": "seasonal_prophet",
                    "description": "Prophet with weekly and yearly seasonality",
                    "config": {"yearly_seasonality": True, "weekly_seasonality": True, "daily_seasonality": False},
                    "complexity": "medium"
                },
                {
                    "name": "full_prophet",
                    "description": "Prophet with all seasonality components",
                    "config": {"yearly_seasonality": True, "weekly_seasonality": True, "daily_seasonality": True},
                    "complexity": "high"
                }
            ]
            
        elif model_type == "exponential_smoothing":
            # Exponential Smoothing architectures
            architectures = [
                {
                    "name": "simple_exponential",
                    "description": "Simple exponential smoothing",
                    "config": {"trend": None, "seasonal": None},
                    "complexity": "low"
                },
                {
                    "name": "holt_linear",
                    "description": "Holt's linear trend method",
                    "config": {"trend": "add", "seasonal": None},
                    "complexity": "medium"
                },
                {
                    "name": "holt_winters",
                    "description": "Holt-Winters with seasonal component",
                    "config": {"trend": "add", "seasonal": "add"},
                    "complexity": "high"
                }
            ]
            
        elif model_type == "lstm_ts":
            # LSTM time series architectures
            architectures = [
                {
                    "name": "simple_lstm",
                    "description": "Simple LSTM for time series",
                    "config": {"hidden_size": 32, "num_layers": 1, "sequence_length": 10},
                    "complexity": "low"
                },
                {
                    "name": "deep_lstm",
                    "description": "Deep LSTM with multiple layers",
                    "config": {"hidden_size": 64, "num_layers": 2, "sequence_length": 20},
                    "complexity": "medium"
                },
                {
                    "name": "complex_lstm",
                    "description": "Complex LSTM with attention mechanisms",
                    "config": {"hidden_size": 128, "num_layers": 3, "sequence_length": 30},
                    "complexity": "high"
                }
            ]
            
        elif model_type == "seasonal_decompose":
            # Seasonal Decomposition architectures
            architectures = [
                {
                    "name": "additive_decompose",
                    "description": "Additive seasonal decomposition",
                    "config": {"model": "additive", "period": "auto"},
                    "complexity": "low"
                },
                {
                    "name": "multiplicative_decompose",
                    "description": "Multiplicative seasonal decomposition",
                    "config": {"model": "multiplicative", "period": "auto"},
                    "complexity": "medium"
                },
                {
                    "name": "stl_decompose",
                    "description": "STL (Seasonal and Trend decomposition using Loess)",
                    "config": {"model": "stl", "period": "auto", "robust": True},
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
        elif task_type == "forecasting":
            # Time series forecasting models
            if data_profile.n_rows < 100:
                # Small datasets - use simple forecasting methods
                models = ["exponential_smoothing", "arima"]
            elif data_profile.n_rows < 1000:
                # Medium datasets - balanced approach
                models = ["arima", "exponential_smoothing", "prophet"]
            else:
                # Large datasets - can use more complex models
                models = ["prophet", "arima", "lstm_ts"]
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
            # Enhanced forecasting model characteristics
            if task_type == "forecasting":
                if model_type == "prophet":
                    complexity = 6.0
                    training_time = "medium"
                    reasoning = "Excellent for seasonal patterns, handles holidays and missing data automatically"
                    pros = ["Automatic seasonality detection", "Handles holidays and events", "Robust to missing data", "Provides uncertainty intervals"]
                    cons = ["Requires pandas datetime index", "Can be slow on very large datasets", "Limited to univariate forecasting"]
                    best_for = "Business time series with strong seasonality and known holidays"
                    interpretability = "high"
                    memory_usage = "medium"
                elif model_type == "arima":
                    complexity = 7.0
                    training_time = "medium"
                    reasoning = "Classical statistical approach, excellent for trend and seasonality modeling"
                    pros = ["Well-established statistical foundation", "Good for trend modeling", "Provides confidence intervals", "Handles seasonality well"]
                    cons = ["Requires stationary data", "Manual parameter tuning", "Sensitive to outliers", "Univariate only"]
                    best_for = "Time series with clear trends and seasonal patterns, when statistical rigor is important"
                    interpretability = "high"
                    memory_usage = "low"
                elif model_type == "lstm_ts":
                    complexity = 8.0
                    training_time = "slow"
                    reasoning = "Deep learning approach capable of capturing complex non-linear patterns"
                    pros = ["Captures complex patterns", "Can handle multivariate data", "No stationarity requirements", "Good for long sequences"]
                    cons = ["Requires large datasets", "Black box model", "Computationally intensive", "Hyperparameter sensitive"]
                    best_for = "Large datasets with complex non-linear patterns and multiple input features"
                    interpretability = "low"
                    memory_usage = "high"
                elif model_type == "exponential_smoothing":
                    complexity = 4.0
                    training_time = "fast"
                    reasoning = "Simple and fast method, good baseline for trend and seasonal patterns"
                    pros = ["Very fast training", "Simple to understand", "Good baseline model", "Handles trend and seasonality"]
                    cons = ["Limited complexity", "No external features", "Simple assumptions", "May miss complex patterns"]
                    best_for = "Quick forecasting baseline, simple trend and seasonal patterns"
                    interpretability = "high"
                    memory_usage = "low"
                elif model_type == "seasonal_decompose":
                    complexity = 5.0
                    training_time = "fast"
                    reasoning = "Decomposes time series into trend, seasonal, and residual components"
                    pros = ["Clear interpretability", "Good for exploratory analysis", "Handles seasonality well", "Fast computation"]
                    cons = ["Simple assumptions", "Limited forecasting capability", "Requires regular patterns", "No external features"]
                    best_for = "Understanding seasonal patterns and time series decomposition"
                    interpretability = "high"
                    memory_usage = "low"
                else:
                    # Generic fallback for any other forecasting models
                    complexity = 5.0
                    training_time = "medium"
                    reasoning = f"Specialized {model_type} forecasting model for time series prediction"
                    pros = ["Specialized for time series", "Handles temporal dependencies"]
                    cons = ["Limited to temporal data", "May require parameter tuning"]
                    best_for = "Time series forecasting tasks"
                    interpretability = "medium"
                    memory_usage = "medium"
            else:
                complexity = 5.0
                training_time = "medium"
                reasoning = "Fallback recommendation"
                pros = []
                cons = []
                best_for = ""
                interpretability = "medium"
                memory_usage = "medium"

            fallback_models.append(ModelRecommendation(
                model_type=model_type,
                model_family=self._get_model_family(model_type),
                complexity_score=complexity,
                expected_training_time=training_time,
                parameter_estimates={},
                reasoning=reasoning,
                pros=pros,
                cons=cons,
                best_for=best_for,
                interpretability=interpretability,
                memory_usage=memory_usage,
                architectures=self._generate_model_architectures(
                    {"model_type": model_type, "complexity_score": complexity, "parameter_estimates": {}},
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
            "catboost": "tree_based",
            "hist_gradient_boosting": "tree_based",
            "neural_network": "neural",
            "svm": "instance_based",
            "logistic_regression": "linear",
            "linear_regression": "linear",
            "knn": "instance_based",
            "naive_bayes": "probabilistic",
            "isolation_forest": "anomaly_detection",
            "voting": "ensemble",
            "stacking": "ensemble",
            "kmeans": "clustering",
            "dbscan": "clustering",
            "hierarchical": "clustering",
            "spectral": "clustering",
            "gaussian_mixture": "clustering",
            "arima": "time_series",
            "prophet": "time_series",
            "exponential_smoothing": "time_series",
            "lstm_ts": "time_series",
            "seasonal_decompose": "time_series"
        }
        return family_map.get(model_type, "unknown")
    
    def _build_model_catalog(self) -> Dict[str, Any]:
        """Build catalog of available models"""
        catalog = {
            "tree_based": {
                "random_forest": {"class": RandomForestClassifier, "regressor": RandomForestRegressor},
                "extra_trees": {"class": ExtraTreesClassifier, "regressor": ExtraTreesRegressor},
                "xgboost": {"class": xgb.XGBClassifier, "regressor": xgb.XGBRegressor},
                "lightgbm": {"class": lgb.LGBMClassifier, "regressor": lgb.LGBMRegressor},
                "hist_gradient_boosting": {"class": HistGradientBoostingClassifier, "regressor": HistGradientBoostingRegressor}
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
            "anomaly_detection": {
                "isolation_forest": {"class": IsolationForest, "regressor": None}
            },
            "ensemble": {
                "voting": {"class": VotingClassifier, "regressor": VotingRegressor},
                "stacking": {"class": StackingClassifier, "regressor": StackingRegressor}
            },
            "clustering": {
                "kmeans": {"class": KMeans, "regressor": None},
                "dbscan": {"class": DBSCAN, "regressor": None},
                "hierarchical": {"class": AgglomerativeClustering, "regressor": None},
                "spectral": {"class": SpectralClustering, "regressor": None},
                "gaussian_mixture": {"class": GaussianMixture, "regressor": None}
            },
            "time_series": {
                "arima": {"class": "ARIMAForecaster", "regressor": None},
                "prophet": {"class": "ProphetForecaster", "regressor": None},
                "exponential_smoothing": {"class": "ExponentialSmoothing", "regressor": None},
                "lstm_ts": {"class": "LSTMTimeSeriesModel", "regressor": None},
                "seasonal_decompose": {"class": "SeasonalDecompose", "regressor": None}
            }
        }
        
        # Add CatBoost if available
        if CATBOOST_AVAILABLE:
            catalog["tree_based"]["catboost"] = {"class": cb.CatBoostClassifier, "regressor": cb.CatBoostRegressor}
        
        return catalog 