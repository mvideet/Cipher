"""
Data profiler for analyzing dataset characteristics and quality
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import structlog

from ..models.schema import DataProfile

logger = structlog.get_logger()


class DataProfiler:
    """Dataset profiling and quality analysis"""
    
    def profile_dataset(self, df: pd.DataFrame) -> DataProfile:
        """Profile a dataset and return analysis results"""
        
        logger.info("Profiling dataset", shape=df.shape)
        
        n_rows, n_cols = df.shape
        columns = {}
        issues = []
        recommendations = []
        
        for col in df.columns:
            column_profile = self._profile_column(df[col])
            columns[col] = column_profile
            
            # Check for issues
            if column_profile["fraction_null"] > 0.8:
                issues.append(f"Column '{col}' has >80% null values")
                recommendations.append(f"Consider dropping column '{col}'")
            
            elif column_profile["fraction_null"] > 0.4:
                issues.append(f"Column '{col}' has >40% null values")
                recommendations.append(f"Consider imputation strategy for '{col}'")
            
            # Check for unique ID columns
            if column_profile["is_unique_id"]:
                issues.append(f"Column '{col}' appears to be a unique identifier")
                recommendations.append(f"Consider excluding '{col}' from training")
            
            # Check for high cardinality categoricals
            if (column_profile["dtype"] == "object" and 
                column_profile["n_unique"] > 1000):
                issues.append(f"Column '{col}' has very high cardinality ({column_profile['n_unique']} unique values)")
                recommendations.append(f"Consider feature engineering or dropping '{col}'")
        
        # Overall dataset checks
        if n_rows < 500:
            issues.append("Dataset is very small (<500 rows)")
            recommendations.append("Consider cross-validation and reduce hyperparameter search space")
        
        if n_cols > n_rows:
            issues.append("More features than samples (curse of dimensionality)")
            recommendations.append("Consider feature selection or dimensionality reduction")
        
        return DataProfile(
            n_rows=n_rows,
            n_cols=n_cols,
            columns=columns,
            issues=issues,
            recommendations=recommendations
        )
    
    def _profile_column(self, series: pd.Series) -> Dict[str, Any]:
        """Profile a single column"""
        
        n_total = len(series)
        n_null = series.isnull().sum()
        n_unique = series.nunique()
        
        profile = {
            "dtype": str(series.dtype),
            "n_null": int(n_null),
            "fraction_null": float(n_null / n_total) if n_total > 0 else 0.0,
            "n_unique": int(n_unique),
            "is_unique_id": bool(n_unique == n_total and n_total > 1)
        }
        
        # Numeric column analysis
        if pd.api.types.is_numeric_dtype(series):
            non_null_series = series.dropna()
            if len(non_null_series) > 0:
                profile.update({
                    "min": float(non_null_series.min()),
                    "max": float(non_null_series.max()),
                    "mean": float(non_null_series.mean()),
                    "std": float(non_null_series.std()) if len(non_null_series) > 1 else 0.0,
                    "skew": float(non_null_series.skew()) if len(non_null_series) > 1 else 0.0,
                    "kurtosis": float(non_null_series.kurtosis()) if len(non_null_series) > 1 else 0.0
                })
        
        # Categorical column analysis
        elif series.dtype == 'object':
            if n_unique <= 20:  # Show value counts for low cardinality
                value_counts = series.value_counts().head(10).to_dict()
                profile["top_values"] = {str(k): int(v) for k, v in value_counts.items()}
            
            # Check if it looks like a category vs free text
            avg_length = series.dropna().astype(str).str.len().mean()
            profile["avg_string_length"] = float(avg_length) if not pd.isna(avg_length) else 0.0
            profile["likely_categorical"] = bool(n_unique < n_total * 0.1 and avg_length < 50)
        
        # DateTime analysis
        elif pd.api.types.is_datetime64_any_dtype(series):
            non_null_series = series.dropna()
            if len(non_null_series) > 0:
                profile.update({
                    "min_date": non_null_series.min().isoformat(),
                    "max_date": non_null_series.max().isoformat(),
                    "date_range_days": int((non_null_series.max() - non_null_series.min()).days)
                })
        
        return profile
    
    def suggest_preprocessing(self, profile: DataProfile, target_col: str) -> Dict[str, Any]:
        """Suggest preprocessing steps based on profile"""
        
        suggestions = {
            "drop_columns": [],
            "impute_columns": [],
            "encode_columns": [],
            "scale_columns": [],
            "feature_selection": None
        }
        
        for col, col_profile in profile.columns.items():
            if col == target_col:
                continue
            
            # Suggest dropping problematic columns
            if (col_profile["fraction_null"] > 0.8 or 
                col_profile["is_unique_id"] or
                (col_profile["dtype"] == "object" and col_profile["n_unique"] > 1000)):
                suggestions["drop_columns"].append(col)
            
            # Suggest imputation
            elif col_profile["fraction_null"] > 0.1:
                suggestions["impute_columns"].append(col)
            
            # Suggest encoding for categoricals
            elif col_profile["dtype"] == "object" and col_profile.get("likely_categorical", True):
                suggestions["encode_columns"].append(col)
            
            # Suggest scaling for numeric
            elif pd.api.types.is_numeric_dtype(col_profile["dtype"]):
                suggestions["scale_columns"].append(col)
        
        # Feature selection suggestion
        if profile.n_cols > 50:
            suggestions["feature_selection"] = "Consider using SelectKBest or recursive feature elimination"
        
        return suggestions 