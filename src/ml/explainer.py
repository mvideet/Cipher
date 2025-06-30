"""
SHAP-based model explainer for generating feature importance and explanations
"""

import pickle
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import structlog
import openai
from sklearn.pipeline import Pipeline

from ..core.config import settings

logger = structlog.get_logger()


class Explainer:
    """SHAP-based model explainer"""
    
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
        
        # Get model from pipeline or use direct model
        if isinstance(pipeline, Pipeline):
            # This is a proper sklearn Pipeline
            model = pipeline.named_steps['model']
            X_processed = pipeline[:-1].transform(X)  # All steps except final model
        else:
            # This is a direct model (like ensemble) without pipeline wrapper
            model = pipeline
            X_processed = X  # Use raw features for ensemble models
        
        # For ensemble models, we can't easily explain individual components
        # so we'll use a kernel explainer
        if hasattr(model, 'estimators_') or 'Voting' in type(model).__name__ or 'Stacking' in type(model).__name__:
            logger.info("Detected ensemble model, using simplified explanation")
            # For ensembles, we'll just do basic feature importance
            if hasattr(model, 'feature_importances_'):
                # Some ensembles have feature importance
                feature_names = X.columns.tolist()
                feature_importance = dict(zip(feature_names, model.feature_importances_))
                sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
                return {
                    "feature_importance": sorted_importance,
                    "plot_path": None,
                    "text_explanation": f"This ensemble model uses multiple algorithms. The most important features are: {', '.join(list(sorted_importance.keys())[:5])}.",
                    "n_samples_explained": len(X)
                }
            else:
                # Fallback for ensembles without feature importance
                feature_names = X.columns.tolist()
                return {
                    "feature_importance": {name: 1.0/len(feature_names) for name in feature_names},
                    "plot_path": None,
                    "text_explanation": "This ensemble model combines multiple algorithms. Feature importance analysis is not available for this ensemble type.",
                    "n_samples_explained": len(X)
                }
        
        # Choose SHAP explainer based on model type
        explainer = self._get_explainer(model, X_processed)
        
        # Calculate SHAP values
        logger.info("Calculating SHAP values")
        shap_values = explainer.shap_values(X_processed)
        
        # For multi-class classification, use the first class
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        # Get feature names after preprocessing
        feature_names = self._get_feature_names(pipeline, X)
        
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
            "text_explanation": text_explanation,
            "n_samples_explained": len(X)
        }
    
    def _get_explainer(self, model, X_processed):
        """Get appropriate SHAP explainer for the model"""
        
        model_name = type(model).__name__.lower()
        
        # Use TreeExplainer for tree-based models
        if any(tree_type in model_name for tree_type in ['lgbm', 'lightgbm', 'xgb', 'randomforest', 'decisiontree']):
            logger.info("Using TreeExplainer")
            return shap.TreeExplainer(model)
        
        # Use LinearExplainer for linear models
        elif any(linear_type in model_name for linear_type in ['linear', 'logistic']):
            logger.info("Using LinearExplainer")
            return shap.LinearExplainer(model, X_processed)
        
        # Use KernelExplainer for other models (slower)
        else:
            logger.info("Using KernelExplainer")
            # Sample background data for kernel explainer
            background_size = min(100, len(X_processed))
            background = X_processed[:background_size]
            return shap.KernelExplainer(model.predict, background)
    
    def _get_feature_names(self, pipeline, X_original):
        """Get feature names after preprocessing"""
        
        try:
            # Check if this is a pipeline with preprocessing steps
            if isinstance(pipeline, Pipeline):
                # Try to get feature names from the preprocessor
                preprocessor = pipeline.named_steps.get('preprocessor') or pipeline.named_steps.get('transform')
                
                if preprocessor and hasattr(preprocessor, 'get_feature_names_out'):
                    return preprocessor.get_feature_names_out().tolist()
                elif preprocessor and hasattr(preprocessor, 'get_feature_names'):
                    return preprocessor.get_feature_names().tolist()
            
            # Fallback: use original column names
            return X_original.columns.tolist()
            
        except Exception as e:
            logger.warning("Could not extract feature names", error=str(e))
            # Generate generic names based on original data shape
            return X_original.columns.tolist() if hasattr(X_original, 'columns') else [f"feature_{i}" for i in range(X_original.shape[1])]
    
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