"""
Data profiler for analyzing dataset characteristics and quality, including time series analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import structlog
from datetime import datetime, timedelta

# Time series analysis imports
try:
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.tsa.seasonal import seasonal_decompose
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    import warnings
    warnings.warn("Statsmodels not available for advanced time series analysis")
    import warnings
    warnings.warn("Statsmodels not available for advanced time series analysis")

# Define DataProfile class locally since we can't import it
class DataProfile:
    def __init__(self, n_rows: int, n_cols: int, columns: Dict[str, Any], issues: List[str], recommendations: List[str]):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.columns = columns
        self.issues = issues
        self.recommendations = recommendations

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
    """Dataset profiling and quality analysis with time series support"""
    
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
    
    def profile_time_series(self, df: pd.DataFrame, date_column: str, target_column: str) -> Dict[str, Any]:
        """
        Analyze temporal characteristics for time series forecasting:
        - Date range and frequency detection
        - Missing value patterns in time
        - Trend analysis (increasing/decreasing/stable)
        - Seasonality detection (FFT, autocorrelation)
        - Stationarity tests (ADF, KPSS)
        - Outlier detection in temporal context
        - Data frequency consistency
        """
        
        logger.info("Profiling time series data", 
                   date_col=date_column, 
                   target_col=target_column, 
                   n_rows=len(df))
        
        ts_profile = {
            "is_time_series": True,
            "date_column": date_column,
            "target_column": target_column,
            "temporal_analysis": {},
            "recommendations": [],
            "issues": []
        }
        
        try:
            # Convert and validate date column
            df_ts = df.copy()
            df_ts[date_column] = pd.to_datetime(df_ts[date_column])
            
            # Sort by date
            df_ts = df_ts.sort_values(date_column).reset_index(drop=True)
            
            # Basic temporal characteristics
            date_range = df_ts[date_column].max() - df_ts[date_column].min()
            
            ts_profile["temporal_analysis"]["date_range"] = {
                "start_date": df_ts[date_column].min().isoformat(),
                "end_date": df_ts[date_column].max().isoformat(),
                "total_days": date_range.days,
                "total_periods": len(df_ts)
            }
            
            # Frequency detection
            frequency_info = self._detect_frequency(df_ts[date_column])
            ts_profile["temporal_analysis"]["frequency"] = frequency_info
            
            # Gap analysis
            gap_analysis = self._analyze_temporal_gaps(df_ts[date_column], frequency_info["inferred_freq"])
            ts_profile["temporal_analysis"]["gaps"] = gap_analysis
            
            # Target variable temporal analysis
            if target_column in df_ts.columns:
                target_analysis = self._analyze_target_temporal_patterns(
                    df_ts, date_column, target_column, frequency_info["inferred_freq"]
                )
                ts_profile["temporal_analysis"]["target_patterns"] = target_analysis
            
            # Generate time series specific recommendations
            ts_recommendations = self._generate_ts_recommendations(ts_profile["temporal_analysis"])
            ts_profile["recommendations"] = ts_recommendations
            
        except Exception as e:
            logger.error("Time series profiling failed", error=str(e))
            ts_profile["issues"].append(f"Time series analysis failed: {str(e)}")
            ts_profile["is_time_series"] = False
        
        return convert_numpy_types(ts_profile)
    
    def _detect_frequency(self, date_series: pd.Series) -> Dict[str, Any]:
        """Detect the frequency of time series data"""
        
        if len(date_series) < 2:
            return {
                "inferred_freq": "unknown",
                "frequency_confidence": 0.0,
                "irregular_periods": True,
                "common_intervals": []
            }
        
        # Calculate differences between consecutive dates
        diffs = date_series.diff().dropna()
        
        # Convert to hours for analysis
        diff_hours = diffs.dt.total_seconds() / 3600
        
        # Find most common intervals
        from collections import Counter
        diff_counts = Counter(diff_hours.round(2))
        
        if not diff_counts:
            return {
                "inferred_freq": "unknown",
                "frequency_confidence": 0.0,
                "irregular_periods": True,
                "common_intervals": []
            }
        
        most_common = diff_counts.most_common(5)
        dominant_interval = most_common[0][0]  # hours
        dominant_count = most_common[0][1]
        
        # Calculate frequency confidence
        confidence = dominant_count / len(diff_hours)
        
        # Classify frequency
        freq_classification = "unknown"
        if abs(dominant_interval - 1) < 0.1:  # ~1 hour
            freq_classification = "H"
        elif abs(dominant_interval - 24) < 1:  # ~24 hours (daily)
            freq_classification = "D"
        elif 160 <= dominant_interval <= 192:  # ~168 hours (weekly)
            freq_classification = "W"
        elif 720 <= dominant_interval <= 780:  # ~744 hours (monthly)
            freq_classification = "M"
        elif 2160 <= dominant_interval <= 2208:  # ~2190 hours (quarterly)
            freq_classification = "Q"
        elif 8640 <= dominant_interval <= 8928:  # ~8760 hours (yearly)
            freq_classification = "Y"
        
        return {
            "inferred_freq": freq_classification,
            "frequency_confidence": float(confidence),
            "irregular_periods": confidence < 0.8,
            "dominant_interval_hours": float(dominant_interval),
            "common_intervals": [(float(interval), count) for interval, count in most_common]
        }
    
    def _analyze_temporal_gaps(self, date_series: pd.Series, frequency: str) -> Dict[str, Any]:
        """Analyze missing periods and temporal gaps"""
        
        if frequency == "unknown":
            return {"analysis_possible": False, "reason": "Unknown frequency"}
        
        # Expected interval mapping
        freq_to_hours = {
            "H": 1, "D": 24, "W": 168, "M": 720, "Q": 2190, "Y": 8760
        }
        
        expected_interval = freq_to_hours.get(frequency, 24)
        
        # Calculate actual intervals
        diffs = date_series.diff().dropna()
        diff_hours = diffs.dt.total_seconds() / 3600
        
        # Find gaps (intervals significantly larger than expected)
        gap_threshold = expected_interval * 1.5
        gaps = diff_hours[diff_hours > gap_threshold]
        
        return {
            "analysis_possible": True,
            "total_gaps": len(gaps),
            "gap_percentage": float(len(gaps) / len(diff_hours) * 100),
            "largest_gap_hours": float(gaps.max()) if len(gaps) > 0 else 0,
            "average_gap_hours": float(gaps.mean()) if len(gaps) > 0 else 0,
            "regular_intervals": float((diff_hours <= gap_threshold).mean() * 100)
        }
    
    def _analyze_target_temporal_patterns(
        self, 
        df: pd.DataFrame, 
        date_col: str, 
        target_col: str, 
        frequency: str
    ) -> Dict[str, Any]:
        """Analyze temporal patterns in the target variable"""
        
        # Set up time series
        ts_data = df.set_index(date_col)[target_col].dropna()
        
        if len(ts_data) < 10:
            return {"analysis_possible": False, "reason": "Insufficient data"}
        
        analysis = {"analysis_possible": True}
        
        # Basic statistics
        analysis["basic_stats"] = {
            "mean": float(ts_data.mean()),
            "std": float(ts_data.std()),
            "min": float(ts_data.min()),
            "max": float(ts_data.max()),
            "cv": float(ts_data.std() / ts_data.mean()) if ts_data.mean() != 0 else float('inf')
        }
        
        # Trend analysis
        try:
            trend_analysis = self._detect_trend(ts_data)
            analysis["trend"] = trend_analysis
        except Exception as e:
            logger.warning("Trend analysis failed", error=str(e))
            analysis["trend"] = {"detected": False, "error": str(e)}
        
        # Seasonality detection
        try:
            seasonality_analysis = self._detect_seasonality(ts_data, frequency)
            analysis["seasonality"] = seasonality_analysis
        except Exception as e:
            logger.warning("Seasonality analysis failed", error=str(e))
            analysis["seasonality"] = {"detected": False, "error": str(e)}
        
        # Stationarity tests
        if STATSMODELS_AVAILABLE:
            try:
                stationarity_analysis = self._test_stationarity(ts_data)
                analysis["stationarity"] = stationarity_analysis
            except Exception as e:
                logger.warning("Stationarity test failed", error=str(e))
                analysis["stationarity"] = {"test_possible": False, "error": str(e)}
        
        # Outlier detection in temporal context
        try:
            outlier_analysis = self._detect_temporal_outliers(ts_data)
            analysis["outliers"] = outlier_analysis
        except Exception as e:
            logger.warning("Outlier detection failed", error=str(e))
            analysis["outliers"] = {"analysis_possible": False, "error": str(e)}
        
        return analysis
    
    def _detect_trend(self, series: pd.Series) -> Dict[str, Any]:
        """Detect trend using linear regression"""
        try:
            x = np.arange(len(series))
            y = series.values
            A = np.vstack([x, np.ones(len(x))]).T
            slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
            
            # Calculate R-squared
            y_pred = slope * x + intercept
            ss_tot = np.sum((y - y.mean()) ** 2)
            ss_res = np.sum((y - y_pred) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            return {
                "detected": r_squared > 0.25,  # Simple threshold
                "direction": "increasing" if slope > 0 else "decreasing",
                "slope": float(slope),
                "r_squared": float(r_squared)
            }
        except Exception as e:
            logger.warning("Trend detection failed", error=str(e))
            return {"detected": False}
    
    def _detect_seasonality(self, series: pd.Series, frequency: str) -> Dict[str, Any]:
        """Detect seasonality using FFT and autocorrelation"""
        
        if len(series) < 20:
            return {"detected": False, "reason": "Insufficient data for seasonality analysis"}
        
        # FFT-based seasonality detection
        try:
            # Remove trend first
            detrended = series - series.rolling(window=min(len(series)//4, 12), center=True).mean()
            detrended = detrended.dropna()
            
            if len(detrended) < 10:
                return {"detected": False, "reason": "Insufficient data after detrending"}
            
            # FFT analysis
            fft_values = np.fft.fft(detrended.values)
            frequencies = np.fft.fftfreq(len(detrended))
            
            # Find dominant frequencies (excluding DC component)
            magnitudes = np.abs(fft_values)[1:len(fft_values)//2]
            frequencies = frequencies[1:len(frequencies)//2]
            
            if len(magnitudes) == 0:
                return {"detected": False, "reason": "No frequencies to analyze"}
            
            # Find peaks
            peak_indices = []
            for i in range(1, len(magnitudes)-1):
                if magnitudes[i] > magnitudes[i-1] and magnitudes[i] > magnitudes[i+1]:
                    peak_indices.append(i)
            
            if not peak_indices:
                return {"detected": False, "reason": "No clear peaks found"}
            
            # Get top 3 peaks
            top_peaks = sorted(peak_indices, key=lambda x: magnitudes[x], reverse=True)[:3]
            
            seasonal_periods = []
            for peak_idx in top_peaks:
                if frequencies[peak_idx] > 0:
                    period = 1 / frequencies[peak_idx]
                    if 2 <= period <= len(detrended) / 3:  # Reasonable period range
                        seasonal_periods.append({
                            "period": float(period),
                            "strength": float(magnitudes[peak_idx]),
                            "frequency": float(frequencies[peak_idx])
                        })
            
            seasonality_detected = len(seasonal_periods) > 0
            
            # Classify seasonality type based on frequency and period
            seasonality_type = "unknown"
            if seasonal_periods:
                dominant_period = seasonal_periods[0]["period"]
                
                if frequency == "D":
                    if 6 <= dominant_period <= 8:
                        seasonality_type = "weekly"
                    elif 28 <= dominant_period <= 32:
                        seasonality_type = "monthly"
                    elif 90 <= dominant_period <= 100:
                        seasonality_type = "quarterly"
                    elif 360 <= dominant_period <= 370:
                        seasonality_type = "yearly"
                elif frequency == "W":
                    if 4 <= dominant_period <= 5:
                        seasonality_type = "monthly"
                    elif 12 <= dominant_period <= 14:
                        seasonality_type = "quarterly"
                    elif 50 <= dominant_period <= 54:
                        seasonality_type = "yearly"
                elif frequency == "M":
                    if 11 <= dominant_period <= 13:
                        seasonality_type = "yearly"
            
            return {
                "detected": seasonality_detected,
                "type": seasonality_type,
                "periods": seasonal_periods,
                "dominant_period": seasonal_periods[0]["period"] if seasonal_periods else None,
                "strength_score": seasonal_periods[0]["strength"] if seasonal_periods else 0.0
            }
            
        except Exception as e:
            return {"detected": False, "error": f"FFT analysis failed: {str(e)}"}
    
    def _test_stationarity(self, series: pd.Series) -> Dict[str, Any]:
        """Test stationarity using ADF and KPSS tests"""
        
        if len(series) < 10:
            return {"test_possible": False, "reason": "Insufficient data"}
        
        stationarity_results = {"test_possible": True}
        
        # Augmented Dickey-Fuller test
        try:
            adf_stat, adf_pvalue, adf_usedlag, adf_nobs, adf_critical, adf_icbest = adfuller(series)
            
            stationarity_results["adf_test"] = {
                "statistic": float(adf_stat),
                "p_value": float(adf_pvalue),
                "critical_values": {k: float(v) for k, v in adf_critical.items()},
                "is_stationary": adf_pvalue < 0.05,
                "confidence_level": "95%" if adf_pvalue < 0.05 else "low"
            }
        except Exception as e:
            stationarity_results["adf_test"] = {"error": str(e)}
        
        # KPSS test
        try:
            kpss_stat, kpss_pvalue, kpss_lags, kpss_critical = kpss(series, regression='c')
            
            stationarity_results["kpss_test"] = {
                "statistic": float(kpss_stat),
                "p_value": float(kpss_pvalue),
                "critical_values": {k: float(v) for k, v in kpss_critical.items()},
                "is_stationary": kpss_pvalue > 0.05,
                "confidence_level": "95%" if kpss_pvalue > 0.05 else "low"
            }
        except Exception as e:
            stationarity_results["kpss_test"] = {"error": str(e)}
        
        # Combined interpretation
        if "adf_test" in stationarity_results and "kpss_test" in stationarity_results:
            adf_stationary = stationarity_results["adf_test"].get("is_stationary", False)
            kpss_stationary = stationarity_results["kpss_test"].get("is_stationary", False)
            
            if adf_stationary and kpss_stationary:
                conclusion = "stationary"
            elif not adf_stationary and not kpss_stationary:
                conclusion = "non_stationary"
            else:
                conclusion = "inconclusive"
            
            stationarity_results["conclusion"] = conclusion
        
        return stationarity_results
    
    def _detect_temporal_outliers(self, series: pd.Series) -> Dict[str, Any]:
        """Detect outliers using IQR method"""
        try:
            values = series.values
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = np.where((values < lower_bound) | (values > upper_bound))[0]
            outlier_values = values[outliers]
            
            return {
                "detected": len(outliers) > 0,
                "n_outliers": len(outliers),
                "outlier_indices": outliers.tolist(),
                "outlier_values": outlier_values.tolist(),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound)
            }
        except Exception as e:
            logger.warning("Outlier detection failed", error=str(e))
            return {"detected": False}
    
    def _generate_ts_recommendations(self, temporal_analysis: Dict[str, Any]) -> List[str]:
        """Generate time series specific recommendations"""
        
        recommendations = []
        
        # Frequency recommendations
        freq_info = temporal_analysis.get("frequency", {})
        if freq_info.get("irregular_periods", False):
            recommendations.append("Data has irregular time intervals - consider resampling or interpolation")
        
        if freq_info.get("frequency_confidence", 0) < 0.7:
            recommendations.append("Time frequency detection has low confidence - verify data quality")
        
        # Gap recommendations
        gap_info = temporal_analysis.get("gaps", {})
        if gap_info.get("gap_percentage", 0) > 10:
            recommendations.append("Significant temporal gaps detected - consider gap filling strategies")
        
        # Target pattern recommendations
        target_patterns = temporal_analysis.get("target_patterns", {})
        
        if target_patterns.get("analysis_possible", False):
            # Trend recommendations
            trend_info = target_patterns.get("trend", {})
            if trend_info.get("detected", False):
                direction = trend_info.get("direction", "unknown")
                recommendations.append(f"Strong {direction} trend detected - consider trend-aware models (Prophet, ARIMA)")
            
            # Seasonality recommendations
            seasonality_info = target_patterns.get("seasonality", {})
            if seasonality_info.get("detected", False):
                season_type = seasonality_info.get("type", "unknown")
                recommendations.append(f"Seasonality detected ({season_type}) - use seasonal models (Prophet, seasonal ARIMA)")
            
            # Stationarity recommendations
            stationarity_info = target_patterns.get("stationarity", {})
            if stationarity_info.get("test_possible", False):
                conclusion = stationarity_info.get("conclusion", "unknown")
                if conclusion == "non_stationary":
                    recommendations.append("Series is non-stationary - consider differencing or detrending")
                elif conclusion == "inconclusive":
                    recommendations.append("Stationarity tests are inconclusive - manual inspection recommended")
            
            # Outlier recommendations
            outlier_info = target_patterns.get("outliers", {})
            if outlier_info.get("analysis_possible", False):
                total_outliers = outlier_info.get("n_outliers", 0)
                total_points = len(temporal_analysis.get("target_patterns", {}).get("basic_stats", {}))
                if total_outliers > total_points * 0.05:  # More than 5% outliers
                    recommendations.append("High number of outliers detected - consider robust forecasting methods")
        
        # Model recommendations based on data characteristics
        data_length = temporal_analysis.get("date_range", {}).get("total_periods", 0)
        
        if data_length < 100:
            recommendations.append("Limited data points - prefer simple models (exponential smoothing, linear trend)")
        elif data_length < 500:
            recommendations.append("Moderate data size - suitable for ARIMA, Prophet, and Holt-Winters models")
        else:
            recommendations.append("Large dataset - can leverage complex models including LSTM and ensemble methods")
        
        return recommendations
    
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
    
    def validate_for_ml(self, df: pd.DataFrame, target_col: str = None) -> Dict[str, Any]:
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

        # If no target column provided, suggest clustering
        if target_col is None or target_col == "":
            task_type = "clustering"
            validation_results["target_analysis"]["task_type"] = "clustering"
            validation_results["target_analysis"]["note"] = "No target variable specified - suitable for clustering analysis"
            
            # Add clustering-specific suggestions
            validation_results["suggestions"].append("Consider K-means clustering for customer segmentation")
            validation_results["suggestions"].append("Try DBSCAN for density-based pattern discovery")
            validation_results["suggestions"].append("Use hierarchical clustering for taxonomy discovery")
            
        elif target_col not in df.columns:
            validation_results["errors"].append(f"Target column '{target_col}' not found in dataset")
            validation_results["is_ready"] = False
            return validation_results
        else:
            # Analyze target column for supervised learning
            target_series = df[target_col]
            target_null_count = target_series.isnull().sum()
            
            if target_null_count > 0:
                validation_results["errors"].append(f"Target column has {target_null_count} missing values")
                validation_results["is_ready"] = False
            
            # Determine task type from target
            target_unique = target_series.nunique()
            
            # Classification if:
            # 1. Non-numeric (object/bool) with reasonable cardinality, OR
            # 2. Numeric but with very low cardinality (likely categorical integers like 0/1, 1/2/3, etc.)
            is_categorical_string = target_series.dtype in ['object', 'bool'] and target_unique <= 20
            is_categorical_numeric = (
                pd.api.types.is_numeric_dtype(target_series) and 
                target_unique <= 10 and  # Very low cardinality for numeric
                all(target_series.dropna() == target_series.dropna().astype(int))  # All integers
            )
            
            if is_categorical_string or is_categorical_numeric:
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
        feature_cols = [col for col in df.columns if col != target_col] if target_col else df.columns.tolist()
        
        if len(feature_cols) == 0 and task_type != "clustering":
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
            if task_type == "clustering":
                validation_results["suggestions"].append("Small datasets work well with hierarchical clustering")
            else:
                validation_results["suggestions"].append("Consider cross-validation and simpler models")
        
        if n_features > n_samples:
            validation_results["warnings"].append(f"More features ({n_features}) than samples ({n_samples})")
            if task_type == "clustering":
                validation_results["suggestions"].append("Consider PCA dimensionality reduction before clustering")
            else:
                validation_results["suggestions"].append("Feature selection or regularization strongly recommended")
        
        # Check for potential data leakage (only for supervised learning)
        if task_type != "clustering" and target_col:
            potential_leakage = []
            target_series = df[target_col]
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
        
        # Add clustering-specific analysis
        if task_type == "clustering":
            # Analyze feature types for clustering
            numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            validation_results["feature_analysis"]["numeric_features"] = len(numeric_features)
            validation_results["feature_analysis"]["categorical_features"] = len(categorical_features)
            
            if len(numeric_features) >= 2:
                validation_results["suggestions"].append("Good for K-means clustering with numeric features")
            if len(categorical_features) > 0:
                validation_results["suggestions"].append("Consider K-modes or mixed-type clustering for categorical data")
        
        validation_results["feature_analysis"]["n_features"] = len(feature_cols)
        validation_results["feature_analysis"]["problematic_count"] = len(problematic_features)
        
        return validation_results 