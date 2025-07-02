"""
Data profiler for analyzing dataset characteristics and quality
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import structlog

from ..models.schema import DataProfile

logger = structlog.get_logger()


def convert_numpy_types(obj):
    """Convert numpy data types to native Python types for JSON serialization"""
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


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
            n_rows=int(n_rows),
            n_cols=int(n_cols),
            columns=convert_numpy_types(columns),
            issues=issues,
            recommendations=recommendations
        )
    
    def _profile_column(self, series: pd.Series) -> Dict[str, Any]:
        """Profile a single column"""
        
        n_total = len(series)
        
        # Enhanced null detection - handle different null representations
        n_null = 0
        for value in series:
            if pd.isna(value) or value is None:
                n_null += 1
            elif isinstance(value, str) and value.strip() in ['', 'null', 'NULL', 'None', 'NaN', 'nan', 'NA', 'n/a', 'N/A']:
                n_null += 1
        
        # Get unique count excluding nulls for better analysis
        n_unique = series.nunique()
        n_unique_including_null = series.nunique(dropna=False)
        
        profile = {
            "dtype": str(series.dtype),
            "n_null": int(n_null),
            "fraction_null": float(n_null / n_total) if n_total > 0 else 0.0,
            "n_unique": int(n_unique),
            "n_unique_including_null": int(n_unique_including_null),
            "is_unique_id": bool(n_unique == n_total and n_total > 1 and n_null == 0)
        }
        
        # Add data quality flags
        profile["data_quality"] = {
            "has_nulls": bool(n_null > 0),
            "high_null_rate": bool((n_null / n_total) > 0.5) if n_total > 0 else False,
            "is_constant": bool(n_unique <= 1),
            "is_binary": bool(n_unique == 2),
            "is_categorical": bool(n_unique < min(50, n_total * 0.1)) if n_total > 0 else False
        }
        
        # Numeric column analysis
        if pd.api.types.is_numeric_dtype(series):
            non_null_series = series.dropna()
            if len(non_null_series) > 0:
                # Check for potential encoding issues (e.g., categorical encoded as numeric)
                unique_values = sorted(non_null_series.unique())
                is_likely_categorical = (
                    len(unique_values) <= 20 and 
                    all(val == int(val) for val in unique_values if not pd.isna(val)) and
                    min(unique_values) >= 0 and
                    max(unique_values) < 100
                )
                
                profile.update({
                    "min": float(non_null_series.min()),
                    "max": float(non_null_series.max()),
                    "mean": float(non_null_series.mean()),
                    "median": float(non_null_series.median()),
                    "std": float(non_null_series.std()) if len(non_null_series) > 1 else 0.0,
                    "skew": float(non_null_series.skew()) if len(non_null_series) > 1 else 0.0,
                    "kurtosis": float(non_null_series.kurtosis()) if len(non_null_series) > 1 else 0.0,
                    "unique_values": [float(x) if pd.api.types.is_numeric_dtype(type(x)) else x for x in unique_values[:20]],  # Show first 20 unique values
                    "likely_categorical_numeric": bool(is_likely_categorical),
                    "has_zero_values": bool((non_null_series == 0).sum() > 0),
                    "zero_count": int((non_null_series == 0).sum()),
                    "negative_count": int((non_null_series < 0).sum())
                })
                
                # Add outlier detection
                if len(non_null_series) > 4:
                    Q1 = non_null_series.quantile(0.25)
                    Q3 = non_null_series.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = non_null_series[(non_null_series < lower_bound) | (non_null_series > upper_bound)]
                    profile["outlier_count"] = len(outliers)
                    profile["outlier_fraction"] = len(outliers) / len(non_null_series)
        
        # Categorical column analysis
        elif series.dtype == 'object':
            non_null_series = series.dropna()
            if len(non_null_series) > 0:
                # Enhanced categorical analysis
                value_counts = non_null_series.value_counts().head(20)
                profile["top_values"] = {str(k): int(v) for k, v in value_counts.items()}
                
                # Check if it looks like a category vs free text
                str_lengths = non_null_series.astype(str).str.len()
                avg_length = str_lengths.mean()
                max_length = str_lengths.max()
                min_length = str_lengths.min()
                
                profile.update({
                    "avg_string_length": float(avg_length) if not pd.isna(avg_length) else 0.0,
                    "max_string_length": int(max_length) if not pd.isna(max_length) else 0,
                    "min_string_length": int(min_length) if not pd.isna(min_length) else 0,
                    "likely_categorical": bool(n_unique < n_total * 0.1 and avg_length < 50),
                    "likely_free_text": bool(avg_length > 50 or max_length > 200),
                    "has_mixed_case": bool(non_null_series.str.contains(r'[a-z]').any() and non_null_series.str.contains(r'[A-Z]').any()),
                    "has_numbers": bool(non_null_series.str.contains(r'\d').any()),
                    "has_special_chars": bool(non_null_series.str.contains(r'[^a-zA-Z0-9\s]').any())
                })
                
                # Check for potential ID columns
                if n_unique == n_total and not profile["data_quality"]["has_nulls"]:
                    profile["likely_id_column"] = True
                else:
                    profile["likely_id_column"] = False
        
        # DateTime analysis
        elif pd.api.types.is_datetime64_any_dtype(series):
            non_null_series = series.dropna()
            if len(non_null_series) > 0:
                profile.update({
                    "min_date": non_null_series.min().isoformat(),
                    "max_date": non_null_series.max().isoformat(),
                    "date_range_days": int((non_null_series.max() - non_null_series.min()).days),
                    "has_future_dates": bool((non_null_series > pd.Timestamp.now()).any()),
                    "date_frequency": self._detect_date_frequency(non_null_series)
                })
        
        # Boolean analysis
        elif series.dtype == 'bool':
            non_null_series = series.dropna()
            if len(non_null_series) > 0:
                true_count = int((non_null_series == True).sum())
                false_count = int((non_null_series == False).sum())
                profile.update({
                    "true_count": true_count,
                    "false_count": false_count,
                    "true_fraction": float(true_count / len(non_null_series)) if len(non_null_series) > 0 else 0.0
                })
        
        return convert_numpy_types(profile)
    
    def _detect_date_frequency(self, date_series: pd.Series) -> str:
        """Detect the frequency of a datetime series"""
        try:
            if len(date_series) < 2:
                return "unknown"
            
            # Sort dates and calculate differences
            sorted_dates = date_series.sort_values()
            diffs = sorted_dates.diff().dropna()
            
            if len(diffs) == 0:
                return "unknown"
            
            # Get the most common difference
            mode_diff = diffs.mode().iloc[0] if len(diffs.mode()) > 0 else diffs.iloc[0]
            
            # Classify frequency
            if mode_diff.days == 1:
                return "daily"
            elif 6 <= mode_diff.days <= 8:
                return "weekly"
            elif 28 <= mode_diff.days <= 32:
                return "monthly"
            elif 89 <= mode_diff.days <= 92:
                return "quarterly"
            elif 360 <= mode_diff.days <= 370:
                return "yearly"
            else:
                return f"custom_{mode_diff.days}_days"
                
        except Exception:
            return "unknown"
    
    def suggest_preprocessing(self, profile: DataProfile, target_col: str) -> Dict[str, Any]:
        """Suggest preprocessing steps based on profile"""
        
        suggestions = {
            "drop_columns": [],
            "impute_columns": [],
            "encode_columns": [],
            "scale_columns": [],
            "feature_selection": None,
            "outlier_treatment": [],
            "data_quality_issues": [],
            "recommendations": []
        }
        
        for col, col_profile in profile.columns.items():
            if col == target_col:
                continue
            
            data_quality = col_profile.get("data_quality", {})
            
            # Suggest dropping problematic columns
            should_drop = False
            drop_reason = []
            
            if col_profile["fraction_null"] > 0.8:
                should_drop = True
                drop_reason.append(f"high null rate ({col_profile['fraction_null']*100:.1f}%)")
            
            if col_profile["is_unique_id"] or col_profile.get("likely_id_column", False):
                should_drop = True
                drop_reason.append("appears to be unique identifier")
            
            if data_quality.get("is_constant", False):
                should_drop = True
                drop_reason.append("constant values (no variance)")
            
            if (col_profile["dtype"] == "object" and 
                col_profile["n_unique"] > 1000 and 
                not col_profile.get("likely_categorical", False)):
                should_drop = True
                drop_reason.append("very high cardinality text field")
            
            if col_profile.get("likely_free_text", False):
                should_drop = True
                drop_reason.append("appears to be free text")
            
            if should_drop:
                suggestions["drop_columns"].append({
                    "column": col,
                    "reasons": drop_reason
                })
                continue
            
            # Suggest imputation for columns with moderate missing values
            if 0.05 < col_profile["fraction_null"] <= 0.4:
                impute_strategy = "median" if pd.api.types.is_numeric_dtype(col_profile["dtype"]) else "mode"
                suggestions["impute_columns"].append({
                    "column": col,
                    "strategy": impute_strategy,
                    "null_fraction": col_profile["fraction_null"]
                })
            
            # Suggest encoding for categoricals
            if col_profile["dtype"] == "object" and col_profile.get("likely_categorical", True):
                if col_profile["n_unique"] <= 10:
                    encoding = "onehot"
                elif col_profile["n_unique"] <= 50:
                    encoding = "target_encoding"  # For high-cardinality categoricals
                else:
                    encoding = "frequency_encoding"
                
                suggestions["encode_columns"].append({
                    "column": col,
                    "encoding": encoding,
                    "cardinality": col_profile["n_unique"]
                })
            
            # Handle numeric categorical columns (encoded as numbers)
            elif (pd.api.types.is_numeric_dtype(col_profile["dtype"]) and 
                  col_profile.get("likely_categorical_numeric", False)):
                suggestions["encode_columns"].append({
                    "column": col,
                    "encoding": "onehot",
                    "note": "numeric column that appears categorical",
                    "unique_values": col_profile.get("unique_values", [])
                })
            
            # Suggest scaling for numeric features
            elif pd.api.types.is_numeric_dtype(col_profile["dtype"]) and not col_profile.get("likely_categorical_numeric", False):
                # Determine scaling type based on distribution
                skew = abs(col_profile.get("skew", 0))
                has_outliers = col_profile.get("outlier_fraction", 0) > 0.1
                
                if skew > 2 or has_outliers:
                    scaling = "robust"  # Less sensitive to outliers
                elif col_profile.get("min", 0) >= 0:
                    scaling = "minmax"  # For non-negative values
                else:
                    scaling = "standard"  # Default standardization
                
                suggestions["scale_columns"].append({
                    "column": col,
                    "scaling": scaling,
                    "skew": skew,
                    "outlier_fraction": col_profile.get("outlier_fraction", 0)
                })
            
            # Flag outlier treatment needs
            if (pd.api.types.is_numeric_dtype(col_profile["dtype"]) and 
                col_profile.get("outlier_fraction", 0) > 0.2):
                suggestions["outlier_treatment"].append({
                    "column": col,
                    "outlier_fraction": col_profile["outlier_fraction"],
                    "outlier_count": col_profile.get("outlier_count", 0)
                })
            
            # Data quality warnings
            if data_quality.get("high_null_rate", False):
                suggestions["data_quality_issues"].append(f"Column '{col}' has high missing value rate")
            
            if col_profile.get("has_zero_values", False) and col != target_col:
                zero_fraction = col_profile.get("zero_count", 0) / profile.n_rows
                if zero_fraction > 0.5:
                    suggestions["data_quality_issues"].append(
                        f"Column '{col}' has many zero values ({zero_fraction*100:.1f}%) - verify if these are meaningful"
                    )
        
        # Feature selection suggestions
        if profile.n_cols > 50:
            suggestions["feature_selection"] = "Consider using SelectKBest or recursive feature elimination due to high dimensionality"
        elif profile.n_cols > profile.n_rows:
            suggestions["feature_selection"] = "More features than samples - strongly recommend feature selection or regularization"
        
        # Overall recommendations
        if len(suggestions["drop_columns"]) > profile.n_cols * 0.5:
            suggestions["recommendations"].append("Many columns flagged for dropping - consider domain expert review")
        
        if len(suggestions["impute_columns"]) > 5:
            suggestions["recommendations"].append("Multiple columns need imputation - consider advanced imputation methods")
        
        if len(suggestions["outlier_treatment"]) > 0:
            suggestions["recommendations"].append("Outliers detected - consider robust preprocessing methods")
        
        return suggestions
    
    def validate_for_ml(self, df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """Validate dataset for ML training and return actionable insights"""
        
        validation_results = {
            "is_ready": True,
            "errors": [],
            "warnings": [],
            "suggestions": [],
            "target_analysis": {},
            "feature_analysis": {}
        }
        
        # Check basic requirements
        if df.empty:
            validation_results["errors"].append("Dataset is empty")
            validation_results["is_ready"] = False
            return validation_results
        
        if target_col not in df.columns:
            validation_results["errors"].append(f"Target column '{target_col}' not found in dataset")
            validation_results["is_ready"] = False
            return validation_results
        
        # Analyze target column
        target_series = df[target_col]
        target_null_count = target_series.isnull().sum()
        
        if target_null_count > 0:
            validation_results["errors"].append(f"Target column has {target_null_count} missing values")
            validation_results["is_ready"] = False
        
        # Determine task type from target
        target_unique = target_series.nunique()
        if target_unique <= 20 and target_series.dtype in ['object', 'bool'] or all(target_series.dropna().astype(str).str.isdigit()):
            task_type = "classification"
            validation_results["target_analysis"]["task_type"] = "classification"
            validation_results["target_analysis"]["n_classes"] = target_unique
            
            # Check class balance
            class_counts = target_series.value_counts()
            min_class_size = class_counts.min()
            max_class_size = class_counts.max()
            imbalance_ratio = max_class_size / min_class_size if min_class_size > 0 else float('inf')
            
            if imbalance_ratio > 10:
                validation_results["warnings"].append(f"Severe class imbalance detected (ratio: {imbalance_ratio:.1f})")
                validation_results["suggestions"].append("Consider using class_weight='balanced' or resampling techniques")
            
            if min_class_size < 5:
                validation_results["warnings"].append(f"Some classes have very few samples (min: {min_class_size})")
                validation_results["suggestions"].append("Consider combining rare classes or collecting more data")
        
        else:
            task_type = "regression"
            validation_results["target_analysis"]["task_type"] = "regression"
            
            # Check target distribution
            target_clean = target_series.dropna()
            if len(target_clean) > 0:
                skewness = target_clean.skew()
                if abs(skewness) > 2:
                    validation_results["warnings"].append(f"Target variable is highly skewed (skew: {skewness:.2f})")
                    validation_results["suggestions"].append("Consider log transformation for target variable")
        
        # Analyze features
        feature_cols = [col for col in df.columns if col != target_col]
        
        if len(feature_cols) == 0:
            validation_results["errors"].append("No feature columns available")
            validation_results["is_ready"] = False
            return validation_results
        
        # Check feature quality
        problematic_features = []
        for col in feature_cols:
            series = df[col]
            null_fraction = series.isnull().sum() / len(series)
            
            if null_fraction == 1.0:
                problematic_features.append(f"{col} (all missing)")
            elif series.nunique() <= 1:
                problematic_features.append(f"{col} (constant)")
            elif null_fraction > 0.9:
                problematic_features.append(f"{col} (>90% missing)")
        
        if problematic_features:
            validation_results["warnings"].append(f"Problematic features detected: {', '.join(problematic_features)}")
            validation_results["suggestions"].append("Consider removing or investigating these features")
        
        # Check dataset size appropriateness
        n_samples, n_features = df.shape
        
        if n_samples < 100:
            validation_results["warnings"].append(f"Very small dataset ({n_samples} samples)")
            validation_results["suggestions"].append("Consider cross-validation and simpler models")
        
        if n_features > n_samples:
            validation_results["warnings"].append(f"More features ({n_features}) than samples ({n_samples})")
            validation_results["suggestions"].append("Feature selection or regularization strongly recommended")
        
        # Check for potential data leakage
        potential_leakage = []
        for col in feature_cols:
            # Check for perfect correlation with target (potential leakage)
            if pd.api.types.is_numeric_dtype(df[col]) and pd.api.types.is_numeric_dtype(target_series):
                try:
                    correlation = abs(df[col].corr(target_series))
                    if correlation > 0.95:
                        potential_leakage.append(col)
                except:
                    pass
        
        if potential_leakage:
            validation_results["warnings"].append(f"Potential data leakage in columns: {', '.join(potential_leakage)}")
            validation_results["suggestions"].append("Investigate high correlations - may indicate data leakage")
        
        validation_results["feature_analysis"]["n_features"] = len(feature_cols)
        validation_results["feature_analysis"]["problematic_count"] = len(problematic_features)
        
        return validation_results 