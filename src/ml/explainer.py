"""
SHAP-based model explainer for generating feature importance and explanations, including time series forecasting
"""

import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import structlog
import openai
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Time series analysis imports
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

from ..core.config import settings

logger = structlog.get_logger()


class Explainer:
    """SHAP-based model explainer with robust support for all model types including time series"""
    
    def __init__(self):
        self.client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY) if settings.OPENAI_API_KEY else None
    
    async def explain_model(
        self, 
        model_path: str, 
        dataset_path: str, 
        target_col: str,
        max_samples: int = 1000
    ) -> Dict[str, Any]:
        """Generate SHAP explanations for the model"""
        
        logger.info("Generating model explanations", model_path=model_path)
        
        try:
            # Load model and data
            with open(model_path, 'rb') as f:
                pipeline = pickle.load(f)
            
            df = pd.read_csv(dataset_path)
            
            # Prepare features (same as training)
            X = df.drop(columns=[target_col])
            y = df[target_col]
            
            # Sample data if too large
            if len(X) > max_samples:
                sample_idx = np.random.choice(len(X), max_samples, replace=False)
                X = X.iloc[sample_idx]
                y = y.iloc[sample_idx]
            
            logger.info(f"Explaining model with {len(X)} samples", model_type=type(pipeline).__name__)
                
            # Process the model and data for explanation
            explanation_result = await self._explain_with_fallback(pipeline, X, y, model_path, target_col)
            
            return explanation_result
                
        except Exception as e:
            logger.error("Model explanation failed", error=str(e), exc_info=True)
            # Return fallback explanation
            return self._create_fallback_explanation(model_path, target_col)
    
    async def explain_time_series_forecast(
        self,
        model_path: str,
        forecast_data: Dict[str, Any],
        time_series_data: pd.DataFrame,
        date_column: str,
        target_column: str
    ) -> Dict[str, Any]:
        """
        Generate explanations for time series forecasting models:
        - Feature importance for lag variables
        - Seasonal component breakdown
        - Trend contribution analysis
        - External factor impact (if applicable)
        - Forecast confidence explanation
        - Model assumption validation
        """
        
        logger.info("Generating time series forecast explanations", model_path=model_path)
        
        try:
            # Load the time series model
            with open(model_path, 'rb') as f:
                ts_model = pickle.load(f)
            
            # Detect model type
            model_type = self._detect_ts_model_type(ts_model)
            
            # Generate appropriate explanations based on model type
            if model_type == "arima":
                explanation = await self._explain_arima_model(ts_model, time_series_data, date_column, target_column)
            elif model_type == "prophet":
                explanation = await self._explain_prophet_model(ts_model, time_series_data, date_column, target_column)
            elif model_type == "exponential_smoothing":
                explanation = await self._explain_exponential_smoothing(ts_model, time_series_data, date_column, target_column)
            elif model_type == "lstm":
                explanation = await self._explain_lstm_model(ts_model, time_series_data, date_column, target_column)
            else:
                explanation = await self._explain_generic_ts_model(ts_model, time_series_data, date_column, target_column)
            
            # Add forecast confidence and uncertainty explanation
            explanation.update(await self._explain_forecast_uncertainty(
                forecast_data, time_series_data, target_column
            ))
            
            # Generate comprehensive text explanation
            explanation["text_explanation"] = await self._generate_ts_text_explanation(
                explanation, model_type, target_column
            )
            
            return explanation
            
        except Exception as e:
            logger.error("Time series explanation failed", error=str(e))
            return self._create_fallback_ts_explanation(model_path, target_column)
    
    def _detect_ts_model_type(self, model) -> str:
        """Detect the type of time series model"""
        
        model_name = type(model).__name__.lower()
        
        if "arima" in model_name or "sarimax" in model_name:
            return "arima"
        elif "prophet" in model_name:
            return "prophet"
        elif "exponentialsmoothing" in model_name or "holtwinters" in model_name:
            return "exponential_smoothing"
        elif "lstm" in model_name or hasattr(model, 'lstm'):
            return "lstm"
        else:
            return "unknown"
    
    async def _explain_arima_model(
        self, 
        model, 
        data: pd.DataFrame, 
        date_col: str, 
        target_col: str
    ) -> Dict[str, Any]:
        """Explain ARIMA model components and parameters"""
        
        explanation = {
            "model_type": "ARIMA",
            "model_components": {},
            "parameter_analysis": {},
            "statistical_tests": {},
            "residual_analysis": {}
        }
        
        try:
            # Extract ARIMA parameters
            if hasattr(model, 'order'):
                p, d, q = model.order
                explanation["parameter_analysis"]["order"] = {
                    "p": {"value": p, "meaning": f"Uses {p} lagged observations (AR terms)"},
                    "d": {"value": d, "meaning": f"Series was differenced {d} times for stationarity"},
                    "q": {"value": q, "meaning": f"Uses {q} lagged forecast errors (MA terms)"}
                }
            
            # Seasonal parameters if available
            if hasattr(model, 'seasonal_order') and model.seasonal_order:
                P, D, Q, s = model.seasonal_order
                explanation["parameter_analysis"]["seasonal_order"] = {
                    "P": {"value": P, "meaning": f"Seasonal AR terms: {P}"},
                    "D": {"value": D, "meaning": f"Seasonal differencing: {D}"},
                    "Q": {"value": Q, "meaning": f"Seasonal MA terms: {Q}"},
                    "s": {"value": s, "meaning": f"Seasonal period: {s}"}
                }
            
            # Model fit statistics
            if hasattr(model, 'aic'):
                explanation["parameter_analysis"]["fit_statistics"] = {
                    "aic": {"value": float(model.aic), "meaning": "Akaike Information Criterion (lower is better)"},
                    "bic": {"value": float(model.bic), "meaning": "Bayesian Information Criterion (lower is better)"}
                }
            
            # Residual analysis
            if hasattr(model, 'resid'):
                residuals = model.resid.dropna()
                explanation["residual_analysis"] = {
                    "mean": float(residuals.mean()),
                    "std": float(residuals.std()),
                    "ljung_box_test": "Check for autocorrelation in residuals",
                    "normality": "Residuals should be normally distributed for optimal forecasts"
                }
            
            # Coefficient interpretation
            if hasattr(model, 'params') and len(model.params) > 0:
                explanation["model_components"]["coefficients"] = {}
                for i, param in enumerate(model.params):
                    param_name = f"param_{i}"
                    explanation["model_components"]["coefficients"][param_name] = {
                        "value": float(param),
                        "type": "AR/MA coefficient"
                    }
        
        except Exception as e:
            logger.warning("ARIMA detailed analysis failed", error=str(e))
            explanation["error"] = f"Detailed analysis failed: {str(e)}"
        
        return explanation
    
    async def _explain_prophet_model(
        self, 
        model, 
        data: pd.DataFrame, 
        date_col: str, 
        target_col: str
    ) -> Dict[str, Any]:
        """Explain Prophet model components"""
        
        explanation = {
            "model_type": "Prophet",
            "model_components": {},
            "seasonality_analysis": {},
            "trend_analysis": {},
            "component_contributions": {}
        }
        
        try:
            # Seasonality components
            seasonalities = {}
            if hasattr(model, 'yearly_seasonality') and model.yearly_seasonality:
                seasonalities["yearly"] = "Captures annual patterns and cycles"
            if hasattr(model, 'weekly_seasonality') and model.weekly_seasonality:
                seasonalities["weekly"] = "Captures day-of-week patterns"
            if hasattr(model, 'daily_seasonality') and model.daily_seasonality:
                seasonalities["daily"] = "Captures hour-of-day patterns"
            
            explanation["seasonality_analysis"] = seasonalities
            
            # Trend analysis
            if hasattr(model, 'growth'):
                growth = model.growth
                explanation["trend_analysis"]["growth_type"] = {
                    "value": growth,
                    "meaning": "Linear growth" if growth == 'linear' else "Logistic growth with capacity constraints"
                }
            
            # Changepoints
            if hasattr(model, 'changepoints'):
                explanation["trend_analysis"]["changepoints"] = {
                    "count": len(model.changepoints) if model.changepoints is not None else 0,
                    "meaning": "Points where trend rate changes automatically detected"
                }
            
            # Holiday effects
            if hasattr(model, 'holidays') and model.holidays is not None:
                explanation["model_components"]["holidays"] = {
                    "count": len(model.holidays),
                    "meaning": "Special events that impact the forecast"
                }
            
            # Generate forecast decomposition if possible
            if hasattr(model, 'predict') and len(data) > 0:
                # Create future dataframe for analysis
                future = model.make_future_dataframe(periods=0)
                forecast = model.predict(future.tail(min(100, len(future))))
                
                # Component contributions
                components = ['trend', 'seasonal', 'yearly', 'weekly']
                for component in components:
                    if component in forecast.columns:
                        contribution = forecast[component].std() / forecast['yhat'].std() * 100
                        explanation["component_contributions"][component] = {
                            "variance_explained": float(contribution),
                            "meaning": f"Contributes {contribution:.1f}% to forecast variance"
                        }
        
        except Exception as e:
            logger.warning("Prophet detailed analysis failed", error=str(e))
            explanation["error"] = f"Detailed analysis failed: {str(e)}"
        
        return explanation
    
    async def _explain_exponential_smoothing(
        self, 
        model, 
        data: pd.DataFrame, 
        date_col: str, 
        target_col: str
    ) -> Dict[str, Any]:
        """Explain Exponential Smoothing model components"""
        
        explanation = {
            "model_type": "Exponential Smoothing",
            "model_components": {},
            "smoothing_parameters": {},
            "component_analysis": {}
        }
        
        try:
            # Smoothing parameters
            if hasattr(model, 'params'):
                params = model.params
                if 'smoothing_level' in params:
                    alpha = params['smoothing_level']
                    explanation["smoothing_parameters"]["alpha"] = {
                        "value": float(alpha),
                        "meaning": f"Level smoothing: {alpha:.3f} (higher = more responsive to recent changes)"
                    }
                
                if 'smoothing_trend' in params:
                    beta = params['smoothing_trend']
                    explanation["smoothing_parameters"]["beta"] = {
                        "value": float(beta),
                        "meaning": f"Trend smoothing: {beta:.3f} (controls trend adaptation)"
                    }
                
                if 'smoothing_seasonal' in params:
                    gamma = params['smoothing_seasonal']
                    explanation["smoothing_parameters"]["gamma"] = {
                        "value": float(gamma),
                        "meaning": f"Seasonal smoothing: {gamma:.3f} (controls seasonal pattern adaptation)"
                    }
            
            # Model type detection
            if hasattr(model, 'trend'):
                trend_type = model.trend
                explanation["model_components"]["trend"] = {
                    "type": trend_type,
                    "meaning": "No trend" if trend_type is None else f"{trend_type.title()} trend component"
                }
            
            if hasattr(model, 'seasonal'):
                seasonal_type = model.seasonal
                explanation["model_components"]["seasonal"] = {
                    "type": seasonal_type,
                    "meaning": "No seasonality" if seasonal_type is None else f"{seasonal_type.title()} seasonal component"
                }
            
            # Model fit information
            if hasattr(model, 'aic'):
                explanation["component_analysis"]["fit_statistics"] = {
                    "aic": float(model.aic),
                    "meaning": "Model fit quality (lower is better)"
                }
        
        except Exception as e:
            logger.warning("Exponential Smoothing analysis failed", error=str(e))
            explanation["error"] = f"Analysis failed: {str(e)}"
        
        return explanation
    
    async def _explain_lstm_model(
        self, 
        model, 
        data: pd.DataFrame, 
        date_col: str, 
        target_col: str
    ) -> Dict[str, Any]:
        """Explain LSTM model architecture and learned patterns"""
        
        explanation = {
            "model_type": "LSTM",
            "architecture": {},
            "learned_patterns": {},
            "feature_importance": {}
        }
        
        try:
            # Architecture details
            if hasattr(model, 'hidden_size'):
                explanation["architecture"]["hidden_size"] = {
                    "value": model.hidden_size,
                    "meaning": "Number of hidden units in LSTM layers"
                }
            
            if hasattr(model, 'num_layers'):
                explanation["architecture"]["num_layers"] = {
                    "value": model.num_layers,
                    "meaning": "Number of LSTM layers for pattern learning"
                }
            
            if hasattr(model, 'sequence_length'):
                explanation["architecture"]["sequence_length"] = {
                    "value": model.sequence_length,
                    "meaning": "Number of historical time steps used for prediction"
                }
            
            # Data scaling information
            if hasattr(model, 'scaler'):
                explanation["learned_patterns"]["data_scaling"] = {
                    "applied": True,
                    "meaning": "Data was normalized for optimal neural network training"
                }
            
            # Pattern learning capability
            explanation["learned_patterns"]["capabilities"] = {
                "non_linear": "Captures complex non-linear temporal relationships",
                "long_term": "Can learn long-term dependencies through LSTM memory",
                "adaptive": "Automatically discovers relevant patterns without manual specification"
            }
            
            # Feature importance (if lag features are used)
            sequence_len = getattr(model, 'sequence_length', 10)
            explanation["feature_importance"] = {
                f"lag_{i+1}": {
                    "importance": max(0, 1 - i/sequence_len),  # Recent lags more important
                    "meaning": f"Value from {i+1} time steps ago"
                } for i in range(min(sequence_len, 5))
            }
        
        except Exception as e:
            logger.warning("LSTM analysis failed", error=str(e))
            explanation["error"] = f"Analysis failed: {str(e)}"
        
        return explanation
    
    async def _explain_generic_ts_model(
        self, 
        model, 
        data: pd.DataFrame, 
        date_col: str, 
        target_col: str
    ) -> Dict[str, Any]:
        """Generic explanation for unknown time series models"""
        
        explanation = {
            "model_type": "Time Series Model",
            "general_analysis": {},
            "temporal_features": {}
        }
        
        # Analyze the time series data itself
        ts_data = data.set_index(date_col)[target_col].dropna()
        
        if STATSMODELS_AVAILABLE and len(ts_data) > 24:
            try:
                # Seasonal decomposition
                decomposition = seasonal_decompose(ts_data, period=min(12, len(ts_data)//2))
                
                trend_strength = decomposition.trend.dropna().std() / ts_data.std()
                seasonal_strength = decomposition.seasonal.dropna().std() / ts_data.std()
                
                explanation["temporal_features"]["trend_strength"] = {
                    "value": float(trend_strength),
                    "meaning": "Strength of underlying trend component"
                }
                
                explanation["temporal_features"]["seasonal_strength"] = {
                    "value": float(seasonal_strength),
                    "meaning": "Strength of seasonal patterns"
                }
                
            except Exception as e:
                logger.warning("Temporal decomposition failed", error=str(e))
        
        return explanation
    
    async def _explain_forecast_uncertainty(
        self,
        forecast_data: Dict[str, Any],
        historical_data: pd.DataFrame,
        target_column: str
    ) -> Dict[str, Any]:
        """Explain forecast confidence and uncertainty sources"""
        
        uncertainty_explanation = {
            "confidence_analysis": {},
            "uncertainty_sources": {},
            "reliability_assessment": {}
        }
        
        try:
            # Confidence interval analysis
            if "confidence_intervals" in forecast_data:
                ci_data = forecast_data["confidence_intervals"]
                uncertainty_explanation["confidence_analysis"] = {
                    "interpretation": "95% confidence intervals show the range where future values are likely to fall",
                    "width_meaning": "Wider intervals indicate higher uncertainty in predictions",
                    "business_use": "Use lower bound for conservative planning, upper bound for optimistic scenarios"
                }
            
            # Sources of uncertainty
            uncertainty_explanation["uncertainty_sources"] = {
                "model_uncertainty": "Uncertainty due to model parameter estimation",
                "data_uncertainty": "Uncertainty from historical data noise and measurement errors",
                "structural_uncertainty": "Uncertainty from potential changes in underlying patterns",
                "forecast_horizon": "Uncertainty increases with longer forecast horizons"
            }
            
            # Data quality assessment
            if len(historical_data) > 0:
                data_variance = historical_data[target_column].var()
                data_stability = 1 / (1 + data_variance / historical_data[target_column].mean()**2)
                
                uncertainty_explanation["reliability_assessment"] = {
                    "data_stability": {
                        "score": float(data_stability),
                        "meaning": "Higher scores indicate more stable historical patterns"
                    },
                    "sample_size": {
                        "value": len(historical_data),
                        "meaning": "More historical data generally improves forecast reliability"
                    }
                }
        
        except Exception as e:
            logger.warning("Uncertainty analysis failed", error=str(e))
            uncertainty_explanation["error"] = f"Analysis failed: {str(e)}"
        
        return uncertainty_explanation
    
    async def _generate_ts_text_explanation(
        self,
        explanation: Dict[str, Any],
        model_type: str,
        target_column: str
    ) -> str:
        """Generate comprehensive text explanation for time series model"""
        
        if not self.client:
            return self._generate_basic_ts_explanation(explanation, model_type, target_column)
        
        try:
            # Prepare explanation content
            explanation_content = []
            
            # Model type and overview
            explanation_content.append(f"Model Type: {model_type}")
            
            # Key components
            if "model_components" in explanation:
                components = explanation["model_components"]
                if components:
                    explanation_content.append("Key Components:")
                    for component, details in components.items():
                        if isinstance(details, dict) and "meaning" in details:
                            explanation_content.append(f"- {component}: {details['meaning']}")
            
            # Parameters and configuration
            if "parameter_analysis" in explanation:
                params = explanation["parameter_analysis"]
                if params:
                    explanation_content.append("Model Configuration:")
                    for param_group, details in params.items():
                        if isinstance(details, dict):
                            for param, info in details.items():
                                if isinstance(info, dict) and "meaning" in info:
                                    explanation_content.append(f"- {param}: {info['meaning']}")
            
            # Uncertainty information
            if "confidence_analysis" in explanation:
                conf_analysis = explanation["confidence_analysis"]
                if "interpretation" in conf_analysis:
                    explanation_content.append(f"Confidence: {conf_analysis['interpretation']}")
            
            content_text = "\n".join(explanation_content)
            
            prompt = f"""Based on the following time series forecasting model analysis for '{target_column}', write a clear, business-focused explanation (≤200 words) that covers:

1. What the model does and how it works
2. Key components and their business meaning
3. How to interpret forecast confidence
4. Practical implications for decision-making

Model Analysis:
{content_text}

Focus on:
- Practical interpretation for business users
- What drives the forecasts
- How to use the predictions effectively
- When to trust vs. be cautious about the forecasts

Write for business stakeholders who need to understand and act on these forecasts."""

            response = await self.client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a data science expert explaining time series forecasting to business stakeholders. Focus on practical insights and decision-making guidance."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error("Failed to generate time series text explanation", error=str(e))
            return self._generate_basic_ts_explanation(explanation, model_type, target_column)
    
    def _generate_basic_ts_explanation(
        self,
        explanation: Dict[str, Any],
        model_type: str,
        target_column: str
    ) -> str:
        """Generate basic text explanation for time series model"""
        
        basic_explanations = {
            "arima": f"ARIMA model uses historical values and patterns to forecast {target_column}. It analyzes trends, seasonality, and autocorrelations to make predictions.",
            "prophet": f"Prophet model decomposes {target_column} into trend, seasonal, and holiday components for robust forecasting with automatic pattern detection.",
            "exponential_smoothing": f"Exponential Smoothing gives more weight to recent observations when forecasting {target_column}, adapting to changes over time.",
            "lstm": f"LSTM neural network learns complex patterns in {target_column} history to make forecasts, capturing both short-term and long-term dependencies."
        }
        
        base_explanation = basic_explanations.get(model_type.lower(), 
            f"Time series model analyzes historical patterns in {target_column} to generate forecasts.")
        
        # Add uncertainty note
        uncertainty_note = " Forecast confidence decreases with longer time horizons. Use confidence intervals for risk assessment."
        
        return base_explanation + uncertainty_note
    
    def _create_fallback_ts_explanation(self, model_path: str, target_column: str) -> Dict[str, Any]:
        """Create fallback explanation for failed time series analysis"""
        
        return {
            "model_type": "Time Series",
            "text_explanation": f"Time series model trained to forecast {target_column}. The model learns from historical patterns to predict future values.",
            "explanation_method": "Fallback",
            "error": "Detailed analysis not available",
            "basic_interpretation": {
                "purpose": f"Forecasts future values of {target_column}",
                "input": "Historical time series data",
                "output": "Future predictions with uncertainty estimates",
                "confidence": "Forecast accuracy typically decreases with longer horizons"
            }
        }
    
    async def _explain_with_fallback(self, pipeline, X, y, model_path, target_col):
        """Try multiple explanation approaches with fallbacks"""
        
        # Try SHAP explanation first
        try:
            return await self._explain_with_shap(pipeline, X, y, model_path, target_col)
        except Exception as shap_error:
            logger.warning("SHAP explanation failed, trying coefficient-based", error=str(shap_error))
            
            # Try coefficient-based explanation for linear models
            try:
                return await self._explain_with_coefficients(pipeline, X, target_col)
            except Exception as coef_error:
                logger.warning("Coefficient explanation failed, using feature importance", error=str(coef_error))
                
                # Try built-in feature importance
                try:
                    return await self._explain_with_feature_importance(pipeline, X, target_col)
                except Exception as importance_error:
                    logger.warning("Feature importance failed, using fallback", error=str(importance_error))
                    
                    # Final fallback
                    return self._create_fallback_explanation(model_path, target_col)
    
    async def _explain_with_shap(self, pipeline, X, y, model_path, target_col):
        """Explain model using SHAP"""
        
        # Extract model and preprocess data
        model, X_processed, feature_names = self._extract_model_and_features(pipeline, X)
        
        logger.info(f"Model type: {type(model).__name__}, Processed data shape: {X_processed.shape}")
        
        # Get appropriate SHAP explainer
        explainer = self._get_robust_explainer(model, X_processed)
        
        # Calculate SHAP values with error handling
        logger.info("Calculating SHAP values...")
        shap_values = self._calculate_shap_values(explainer, X_processed)
        
        # Calculate feature importance
        feature_importance = self._calculate_feature_importance(shap_values, feature_names)
        
        # Generate plot
        plot_path = self._create_shap_plot(shap_values, feature_names, model_path)
        
        # Generate text explanation
        text_explanation = None
        if self.client:
            text_explanation = await self._generate_text_explanation(
                feature_importance, target_col
            )
        
        return {
            "feature_importance": feature_importance,
            "plot_path": plot_path,
            "text_explanation": text_explanation or f"The model's predictions are most influenced by: {', '.join(list(feature_importance.keys())[:5])}.",
            "n_samples_explained": len(X),
            "explanation_method": "SHAP"
        }
    
    def _extract_model_and_features(self, pipeline, X):
        """Extract the actual model and process features"""
        
        if isinstance(pipeline, Pipeline):
            # Extract the final model from pipeline
            model = pipeline.named_steps['model']
            
            # Process features through pipeline (excluding final model step)
            preprocessing_steps = Pipeline(pipeline.steps[:-1])
            X_processed = preprocessing_steps.transform(X) if len(pipeline.steps) > 1 else X.values
            
            # Get feature names after preprocessing
            feature_names = self._get_feature_names_from_pipeline(preprocessing_steps, X)
            
        else:
            # Direct model (ensemble models are often not wrapped in pipelines)
            model = pipeline
            
            # For ensemble models, assume data is already processed
            if hasattr(X, 'values'):
                X_processed = X.values
                feature_names = X.columns.tolist()
            else:
                X_processed = X
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Ensure X_processed is numpy array
        if hasattr(X_processed, 'toarray'):  # Sparse matrix
            X_processed = X_processed.toarray()
        elif not isinstance(X_processed, np.ndarray):
            X_processed = np.array(X_processed)
        
        # Ensure feature names match processed data dimensions
        if len(feature_names) != X_processed.shape[1]:
            logger.warning(f"Feature name mismatch: {len(feature_names)} names vs {X_processed.shape[1]} features")
            feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]
        
        return model, X_processed, feature_names
    
    def _get_robust_explainer(self, model, X_processed):
        """Get appropriate SHAP explainer with robust model type detection"""
        
        model_name = type(model).__name__.lower()
        logger.info(f"Selecting explainer for model: {model_name}")
        
        # Tree-based models - use TreeExplainer
        if any(tree_type in model_name for tree_type in [
            'lgbm', 'lightgbm', 'xgb', 'xgboost', 'randomforest', 'decisiontree', 
            'extratrees', 'gradientboosting'
        ]):
            logger.info("Using TreeExplainer for tree-based model")
            return shap.TreeExplainer(model)
        
        # Linear models - use LinearExplainer
        elif any(linear_type in model_name for linear_type in [
            'linear', 'logistic', 'ridge', 'lasso', 'elasticnet'
        ]):
            logger.info("Using LinearExplainer for linear model")
            return shap.LinearExplainer(model, X_processed)
        
        # SVM models - use appropriate explainer based on kernel
        elif 'svm' in model_name or 'svc' in model_name or 'svr' in model_name:
            logger.info("Detected SVM model")
            
            # Try to determine kernel type
            kernel = getattr(model, 'kernel', 'rbf')
            if kernel == 'linear':
                logger.info("Using LinearExplainer for linear SVM")
                return shap.LinearExplainer(model, X_processed)
            else:
                logger.info(f"Using KernelExplainer for {kernel} SVM")
                background_size = min(50, len(X_processed))  # Smaller background for SVM
                background = shap.sample(X_processed, background_size)
                return shap.KernelExplainer(model.predict, background)
        
        # Neural networks
        elif any(nn_type in model_name for nn_type in [
            'mlp', 'neural', 'perceptron'
        ]):
            logger.info("Using KernelExplainer for neural network")
            background_size = min(100, len(X_processed))
            background = shap.sample(X_processed, background_size)
            return shap.KernelExplainer(model.predict, background)
        
        # Ensemble models (Voting, Stacking, etc.)
        elif any(ensemble_type in model_name for ensemble_type in [
            'voting', 'stacking', 'bagging', 'ada'
        ]) or hasattr(model, 'estimators_'):
            logger.info("Using KernelExplainer for ensemble model")
            background_size = min(100, len(X_processed))
            background = shap.sample(X_processed, background_size)
            return shap.KernelExplainer(model.predict, background)
        
        # Default: KernelExplainer (slowest but most general)
        else:
            logger.info(f"Using KernelExplainer as fallback for {model_name}")
            background_size = min(100, len(X_processed))
            background = shap.sample(X_processed, background_size)
            return shap.KernelExplainer(model.predict, background)
    
    def _calculate_shap_values(self, explainer, X_processed):
        """Calculate SHAP values with error handling"""
        
        try:
            # Limit sample size for slow explainers
            if isinstance(explainer, shap.KernelExplainer):
                sample_size = min(100, len(X_processed))  # Limit for kernel explainer
                if len(X_processed) > sample_size:
                    indices = np.random.choice(len(X_processed), sample_size, replace=False)
                    X_sample = X_processed[indices]
                    logger.info(f"Using {sample_size} samples for KernelExplainer")
                else:
                    X_sample = X_processed
            else:
                X_sample = X_processed
            
            shap_values = explainer.shap_values(X_sample)
            
            # Handle multi-class classification (take first class or handle appropriately)
            if isinstance(shap_values, list):
                if len(shap_values) == 2:  # Binary classification
                    shap_values = shap_values[1]  # Take positive class
                else:  # Multi-class
                    shap_values = shap_values[0]  # Take first class
            
            logger.info(f"SHAP values calculated successfully, shape: {shap_values.shape}")
            return shap_values
            
        except Exception as e:
            logger.error(f"SHAP calculation failed: {str(e)}")
            raise e
    
    def _get_feature_names_from_pipeline(self, preprocessing_steps, X_original):
        """Robust feature name extraction from preprocessing pipeline"""
        
        try:
            # Try different methods to get feature names
            if hasattr(preprocessing_steps, 'get_feature_names_out'):
                try:
                    feature_names = preprocessing_steps.get_feature_names_out()
                    return [str(name) for name in feature_names]
                except:
                    pass
            
            if hasattr(preprocessing_steps, 'get_feature_names'):
                try:
                    feature_names = preprocessing_steps.get_feature_names()
                    return [str(name) for name in feature_names]
                except:
                    pass
            
            # Try to get from the last step
            if hasattr(preprocessing_steps, 'steps') and preprocessing_steps.steps:
                last_step = preprocessing_steps.steps[-1][1]
                if hasattr(last_step, 'get_feature_names_out'):
                    try:
                        feature_names = last_step.get_feature_names_out()
                        return [str(name) for name in feature_names]
                    except:
                        pass
            
            # Check for ColumnTransformer
            for step_name, step in preprocessing_steps.steps:
                if isinstance(step, ColumnTransformer):
                    try:
                        feature_names = step.get_feature_names_out()
                        return [str(name) for name in feature_names]
                    except:
                        pass
            
            # Fallback: use original column names
            return X_original.columns.tolist()
            
        except Exception as e:
            logger.warning(f"Could not extract feature names: {str(e)}")
            return X_original.columns.tolist() if hasattr(X_original, 'columns') else [f"feature_{i}" for i in range(X_original.shape[1])]
    
    async def _explain_with_coefficients(self, pipeline, X, target_col):
        """Enhanced explanation using model coefficients for linear models"""
        
        model, X_processed, feature_names = self._extract_model_and_features(pipeline, X)
        
        # Check if model has coefficients
        if not hasattr(model, 'coef_'):
            raise ValueError("Model does not have coefficients")
        
        coef = model.coef_
        if coef.ndim > 1:
            coef = coef[0]  # Take first class for multi-class
        
        # Get intercept if available
        intercept = getattr(model, 'intercept_', 0)
        
        # Create feature importance from coefficients
        feature_importance = {}
        coefficient_details = {}
        
        for i, name in enumerate(feature_names[:len(coef)]):
            coef_value = float(coef[i])
            abs_coef = abs(coef_value)
            
            feature_importance[name] = abs_coef
            coefficient_details[name] = {
                "coefficient": coef_value,
                "abs_coefficient": abs_coef,
                "direction": "positive" if coef_value > 0 else "negative",
                "magnitude": "high" if abs_coef > np.std(np.abs(coef)) else "medium" if abs_coef > np.std(np.abs(coef))/2 else "low"
            }
        
        # Sort by importance
        sorted_importance = dict(sorted(
            feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
        
        # Generate enhanced explanation for regression
        explanation_method = "Linear Coefficients"
        if hasattr(model, '__class__'):
            model_name = model.__class__.__name__.lower()
            if 'regression' in model_name:
                explanation_method = "Linear Regression Coefficients"
                enhanced_explanation = await self._generate_linear_regression_explanation(
                    coefficient_details, sorted_importance, target_col, intercept, model
                )
            else:
                enhanced_explanation = await self._generate_linear_classification_explanation(
                    coefficient_details, sorted_importance, target_col, model
                )
        else:
            enhanced_explanation = f"Linear model coefficients show that {', '.join(list(sorted_importance.keys())[:3])} are the most influential features for predicting {target_col}."
        
        # Create coefficient plot
        plot_path = self._create_coefficient_plot(coefficient_details, feature_names, pipeline, X.shape[0])
        
        return {
            "feature_importance": sorted_importance,
            "coefficient_details": coefficient_details,
            "intercept": float(intercept) if isinstance(intercept, (int, float, np.number)) else intercept,
            "plot_path": plot_path,
            "text_explanation": enhanced_explanation,
            "n_samples_explained": len(X),
            "explanation_method": explanation_method
        }
    
    async def _generate_linear_regression_explanation(
        self, 
        coefficient_details: Dict[str, Dict], 
        feature_importance: Dict[str, float], 
        target_col: str,
        intercept: float,
        model
    ) -> str:
        """Generate detailed explanation for linear regression models"""
        
        if not self.client:
            return self._generate_basic_regression_explanation(coefficient_details, feature_importance, target_col)
        
        try:
            # Get top features
            top_features = list(feature_importance.items())[:8]
            
            # Calculate R-squared if possible
            r_squared_info = ""
            try:
                if hasattr(model, 'score'):
                    # Note: We can't calculate R² here without test data, but we can mention it
                    r_squared_info = "\n\nNote: Model R-squared score indicates how well the model explains variance in the target variable."
            except:
                pass
            
            # Prepare detailed coefficient information
            coef_details = []
            for feature, importance in top_features:
                details = coefficient_details[feature]
                direction = details["direction"]
                magnitude = details["magnitude"]
                coef_val = details["coefficient"]
                
                coef_details.append(f"- {feature}: {coef_val:.4f} ({direction} impact, {magnitude} magnitude)")
            
            coef_text = "\n".join(coef_details)
            
            prompt = f"""Based on the following linear regression coefficients for predicting '{target_col}', write a comprehensive explanation (≤200 words) describing:

1. Which features have the strongest positive/negative effects
2. The practical meaning of these coefficients
3. How to interpret the magnitude of effects
4. Business insights and actionable recommendations

Model Details:
- Intercept: {intercept:.4f}
- Target Variable: {target_col}

Top Coefficients:
{coef_text}

Key Points to Address:
- A coefficient of X means: "holding all other features constant, a 1-unit increase in this feature leads to an X-unit change in {target_col}"
- Positive coefficients increase {target_col}, negative coefficients decrease it
- Larger absolute values indicate stronger influence
- Consider which features are most actionable for business decisions{r_squared_info}

Write for business stakeholders who need to understand what drives {target_col} and how to act on these insights."""

            response = await self.client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a data science expert explaining linear regression results to business stakeholders. Focus on practical interpretation and actionable insights."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            explanation = response.choices[0].message.content.strip()
            logger.info("Generated linear regression explanation", length=len(explanation))
            return explanation
            
        except Exception as e:
            logger.error("Failed to generate linear regression explanation", error=str(e))
            return self._generate_basic_regression_explanation(coefficient_details, feature_importance, target_col)
    
    async def _generate_linear_classification_explanation(
        self, 
        coefficient_details: Dict[str, Dict], 
        feature_importance: Dict[str, float], 
        target_col: str,
        model
    ) -> str:
        """Generate detailed explanation for linear classification models"""
        
        if not self.client:
            return self._generate_basic_classification_explanation(coefficient_details, feature_importance, target_col)
        
        try:
            # Get top features
            top_features = list(feature_importance.items())[:8]
            
            # Prepare detailed coefficient information
            coef_details = []
            for feature, importance in top_features:
                details = coefficient_details[feature]
                direction = details["direction"]
                coef_val = details["coefficient"]
                
                effect_desc = "increases" if direction == "positive" else "decreases"
                coef_details.append(f"- {feature}: {coef_val:.4f} ({effect_desc} probability of {target_col})")
            
            coef_text = "\n".join(coef_details)
            
            prompt = f"""Based on the following logistic regression coefficients for predicting '{target_col}', write a clear explanation (≤150 words) describing:

1. Which features most strongly influence the probability of {target_col}
2. How to interpret positive vs negative coefficients
3. Business implications and actionable insights

Top Coefficients:
{coef_text}

Key Points:
- Positive coefficients increase the odds/probability of {target_col}
- Negative coefficients decrease the odds/probability of {target_col}
- Larger absolute values indicate stronger influence on the prediction
- Focus on practical business meaning and what actions can be taken

Write for business users who need to understand what factors drive {target_col} outcomes."""

            response = await self.client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a data science expert explaining logistic regression results to business users. Focus on probability interpretation and business impact."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=250
            )
            
            explanation = response.choices[0].message.content.strip()
            logger.info("Generated linear classification explanation", length=len(explanation))
            return explanation
            
        except Exception as e:
            logger.error("Failed to generate linear classification explanation", error=str(e))
            return self._generate_basic_classification_explanation(coefficient_details, feature_importance, target_col)
    
    def _generate_basic_regression_explanation(
        self, 
        coefficient_details: Dict[str, Dict], 
        feature_importance: Dict[str, float], 
        target_col: str
    ) -> str:
        """Generate basic regression explanation without LLM"""
        
        top_features = list(feature_importance.items())[:5]
        positive_features = []
        negative_features = []
        
        for feature, _ in top_features:
            details = coefficient_details[feature]
            if details["direction"] == "positive":
                positive_features.append(feature)
            else:
                negative_features.append(feature)
        
        explanation = f"Linear regression analysis shows that "
        
        if positive_features:
            explanation += f"{', '.join(positive_features[:3])} have positive effects on {target_col}"
            if negative_features:
                explanation += f", while {', '.join(negative_features[:2])} have negative effects"
        elif negative_features:
            explanation += f"{', '.join(negative_features[:3])} have negative effects on {target_col}"
        
        explanation += f". The model coefficients indicate the magnitude of change in {target_col} for each unit increase in these features."
        
        return explanation
    
    def _generate_basic_classification_explanation(
        self, 
        coefficient_details: Dict[str, Dict], 
        feature_importance: Dict[str, float], 
        target_col: str
    ) -> str:
        """Generate basic classification explanation without LLM"""
        
        top_features = list(feature_importance.items())[:5]
        positive_features = []
        negative_features = []
        
        for feature, _ in top_features:
            details = coefficient_details[feature]
            if details["direction"] == "positive":
                positive_features.append(feature)
            else:
                negative_features.append(feature)
        
        explanation = f"Logistic regression analysis shows that "
        
        if positive_features:
            explanation += f"{', '.join(positive_features[:3])} increase the probability of {target_col}"
            if negative_features:
                explanation += f", while {', '.join(negative_features[:2])} decrease the probability"
        elif negative_features:
            explanation += f"{', '.join(negative_features[:3])} decrease the probability of {target_col}"
        
        explanation += ". The coefficients represent the log-odds change for each unit increase in these features."
        
        return explanation
    
    def _create_coefficient_plot(self, coefficient_details: Dict[str, Dict], feature_names: List[str], model_path, n_samples: int):
        """Create and save coefficient plot for linear models"""
        
        try:
            # Get top 15 features by absolute coefficient value
            sorted_features = sorted(
                coefficient_details.items(), 
                key=lambda x: x[1]["abs_coefficient"], 
                reverse=True
            )[:15]
            
            features = [item[0] for item in sorted_features]
            coefficients = [item[1]["coefficient"] for item in sorted_features]
            
            # Create horizontal bar plot
            plt.figure(figsize=(12, 8))
            
            # Color bars by positive/negative
            colors = ['green' if coef > 0 else 'red' for coef in coefficients]
            
            plt.barh(range(len(features)), coefficients, color=colors, alpha=0.7)
            plt.yticks(range(len(features)), features)
            plt.xlabel('Coefficient Value')
            plt.title('Linear Model Coefficients\n(Positive = Increases Target, Negative = Decreases Target)')
            plt.grid(axis='x', alpha=0.3)
            
            # Add value labels on bars
            for i, coef in enumerate(coefficients):
                plt.text(coef + (0.01 * max(abs(c) for c in coefficients)), i, f'{coef:.3f}', 
                        va='center', ha='left' if coef > 0 else 'right', fontsize=9)
            
            # Add vertical line at x=0
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
            
            plt.gca().invert_yaxis()  # Most important at top
            plt.tight_layout()
            
            # Save plot
            if hasattr(model_path, 'parent'):
                plot_dir = model_path.parent
            else:
                plot_dir = Path(str(model_path)).parent
            plot_path = plot_dir / "linear_coefficients.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info("Coefficient plot saved", path=str(plot_path))
            return str(plot_path)
            
        except Exception as e:
            logger.error("Failed to create coefficient plot", error=str(e))
            return None
    
    async def _explain_with_feature_importance(self, pipeline, X, target_col):
        """Explain using built-in feature importance"""
        
        model, X_processed, feature_names = self._extract_model_and_features(pipeline, X)
        
        # Check if model has feature importance
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            
            # Create feature importance dictionary
            feature_importance = {}
            for i, name in enumerate(feature_names[:len(importance)]):
                feature_importance[name] = float(importance[i])
            
            # Sort by importance
            sorted_importance = dict(sorted(
                feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            ))
            
            return {
                "feature_importance": sorted_importance,
                "plot_path": None,
                "text_explanation": f"Tree-based model feature importance shows that {', '.join(list(sorted_importance.keys())[:3])} are the most important features for predicting {target_col}.",
                "n_samples_explained": len(X),
                "explanation_method": "Feature Importance"
            }
        else:
            raise ValueError("Model does not have feature_importances_")
    
    def _create_fallback_explanation(self, model_path, target_col):
        """Create a basic fallback explanation when all methods fail"""
        
        logger.warning("Using fallback explanation - all explanation methods failed")
        
        return {
            "feature_importance": {"unknown_feature": 1.0},
            "plot_path": None,
            "text_explanation": f"Model explanation is not available for this model type. The model was trained to predict {target_col}.",
            "n_samples_explained": 0,
            "explanation_method": "Fallback"
        }
    
    def _calculate_feature_importance(self, shap_values, feature_names):
        """Calculate feature importance from SHAP values"""
        
        # Calculate mean absolute SHAP values
        importance_scores = np.abs(shap_values).mean(axis=0)
        
        # Create importance dictionary
        feature_importance = {}
        for i, name in enumerate(feature_names[:len(importance_scores)]):
            feature_importance[name] = float(importance_scores[i])
        
        # Sort by importance
        sorted_importance = dict(sorted(
            feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
        
        return sorted_importance
    
    def _create_shap_plot(self, shap_values, feature_names, model_path):
        """Create and save SHAP summary plot"""
        
        try:
            # Create summary plot
            plt.figure(figsize=(10, 6))
            
            # Get top 10 features
            importance_scores = np.abs(shap_values).mean(axis=0)
            top_indices = np.argsort(importance_scores)[-10:][::-1]
            
            top_shap_values = shap_values[:, top_indices]
            top_feature_names = [feature_names[i] for i in top_indices if i < len(feature_names)]
            
            # Create bar plot
            mean_importance = np.abs(top_shap_values).mean(axis=0)
            
            plt.barh(range(len(mean_importance)), mean_importance)
            plt.yticks(range(len(mean_importance)), top_feature_names[:len(mean_importance)])
            plt.xlabel('Mean |SHAP value|')
            plt.title('Feature Importance (SHAP)')
            plt.gca().invert_yaxis()  # Most important at top
            plt.tight_layout()
            
            # Save plot
            plot_dir = Path(model_path).parent
            plot_path = plot_dir / "shap_importance.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info("SHAP plot saved", path=str(plot_path))
            return str(plot_path)
            
        except Exception as e:
            logger.error("Failed to create SHAP plot", error=str(e))
            return None
    
    async def _generate_text_explanation(
        self, 
        feature_importance: Dict[str, float], 
        target_col: str
    ) -> str:
        """Generate text explanation using GPT-4"""
        
        try:
            # Get top 10 features
            top_features = list(feature_importance.items())[:10]
            
            # Prepare prompt
            features_text = "\n".join([
                f"- {feature}: {importance:.4f}"
                for feature, importance in top_features
            ])
            
            prompt = f"""Based on the following SHAP feature importance scores for predicting '{target_col}', write a concise explanation (≤120 words) describing which factors are most influential and how they might affect the target variable.

Top features by importance:
{features_text}

Focus on:
1. The most important features
2. What these features might represent
3. Their potential impact on {target_col}

Keep it accessible for business users."""

            response = await self.client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a data science expert explaining ML model insights to business users."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            explanation = response.choices[0].message.content.strip()
            logger.info("Generated text explanation", length=len(explanation))
            return explanation
            
        except Exception as e:
            logger.error("Failed to generate text explanation", error=str(e))
            return f"The model's predictions are most influenced by: {', '.join(list(feature_importance.keys())[:5])}." 