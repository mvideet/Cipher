"""
SHAP-based model explainer for generating feature importance and explanations
"""

import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import structlog
import openai
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from ..core.config import settings

logger = structlog.get_logger()


class Explainer:
    """SHAP-based model explainer with robust support for all model types"""
    
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