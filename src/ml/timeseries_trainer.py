"""
Time series forecasting trainer with LLM-guided model selection and comprehensive evaluation
"""

import asyncio
import pickle
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import optuna
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import structlog

# Time series libraries
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss

# Prophet (with error handling)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    warnings.warn("Prophet not available. Install with: pip install prophet")

# Auto-ARIMA - Disabled for Python 3.13 compatibility
PMDARIMA_AVAILABLE = False
# Note: pmdarima has compatibility issues with Python 3.13
# We'll use manual ARIMA parameter selection instead

# PyTorch for LSTM
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Install with: pip install torch")

from ..core.config import settings
from ..models.schema import ModelArtifact, DataProfile
from .model_selector import ModelSelector, EnsembleStrategy, ModelRecommendation

logger = structlog.get_logger()


class TimeSeriesDataPipeline:
    """Handle time series specific data transformations"""
    
    def __init__(self, date_column: str, target_column: str):
        self.date_column = date_column
        self.target_column = target_column
        self.scaler = None
        self.frequency = None
        
    def prepare_time_series_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Prepare time series data with temporal validation and feature engineering"""
        
        logger.info("Preparing time series data", 
                   shape=df.shape, 
                   date_col=self.date_column, 
                   target_col=self.target_column)
        
        # Create a copy to avoid modifying original
        ts_df = df.copy()
        
        # Convert date column to datetime
        ts_df[self.date_column] = pd.to_datetime(ts_df[self.date_column])
        
        # Sort by date
        ts_df = ts_df.sort_values(self.date_column).reset_index(drop=True)
        
        # Remove duplicates and handle missing dates
        ts_df = ts_df.drop_duplicates(subset=[self.date_column])
        
        # Detect frequency
        self.frequency = self._detect_frequency(ts_df[self.date_column])
        
        # Handle missing values in target
        ts_df[self.target_column] = ts_df[self.target_column].interpolate(method='linear')
        
        # Create lag features
        ts_df = self._create_lag_features(ts_df)
        
        # Create rolling statistics features
        ts_df = self._create_rolling_features(ts_df)
        
        # Create seasonal features
        ts_df = self._create_seasonal_features(ts_df)
        
        # Statistical characteristics
        stats_info = self._analyze_time_series_characteristics(ts_df[self.target_column])
        
        return ts_df, stats_info
    
    def _detect_frequency(self, date_series: pd.Series) -> str:
        """Detect the frequency of time series data"""
        if len(date_series) < 2:
            return "unknown"
        
        diffs = date_series.diff().dropna()
        mode_diff = diffs.mode().iloc[0] if len(diffs.mode()) > 0 else diffs.iloc[0]
        
        if mode_diff.days == 1:
            return "D"  # Daily
        elif 6 <= mode_diff.days <= 8:
            return "W"  # Weekly
        elif 28 <= mode_diff.days <= 32:
            return "M"  # Monthly
        elif 89 <= mode_diff.days <= 92:
            return "Q"  # Quarterly
        elif 360 <= mode_diff.days <= 370:
            return "Y"  # Yearly
        else:
            return f"{mode_diff.days}D"  # Custom
    
    def _create_lag_features(self, df: pd.DataFrame, lags: List[int] = None) -> pd.DataFrame:
        """Create lagged versions of target variable"""
        if lags is None:
            # Adaptive lag selection based on frequency
            if self.frequency == "D":
                lags = [1, 7, 30]  # 1 day, 1 week, 1 month
            elif self.frequency == "W":
                lags = [1, 4, 12]  # 1 week, 1 month, 3 months
            elif self.frequency == "M":
                lags = [1, 3, 12]  # 1, 3, 12 months
            else:
                lags = [1, 2, 3]  # Default
        
        for lag in lags:
            df[f"{self.target_column}_lag_{lag}"] = df[self.target_column].shift(lag)
        
        return df
    
    def _create_rolling_features(self, df: pd.DataFrame, windows: List[int] = None) -> pd.DataFrame:
        """Create rolling statistics features"""
        if windows is None:
            if self.frequency == "D":
                windows = [7, 30, 90]  # Weekly, monthly, quarterly
            elif self.frequency == "W":
                windows = [4, 12, 26]  # Monthly, quarterly, semi-annual
            elif self.frequency == "M":
                windows = [3, 6, 12]  # Quarterly, semi-annual, annual
            else:
                windows = [3, 6, 12]  # Default
        
        for window in windows:
            if window < len(df):
                df[f"{self.target_column}_rolling_mean_{window}"] = df[self.target_column].rolling(window=window).mean()
                df[f"{self.target_column}_rolling_std_{window}"] = df[self.target_column].rolling(window=window).std()
                df[f"{self.target_column}_rolling_min_{window}"] = df[self.target_column].rolling(window=window).min()
                df[f"{self.target_column}_rolling_max_{window}"] = df[self.target_column].rolling(window=window).max()
        
        return df
    
    def _create_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create seasonal and temporal features"""
        df["year"] = df[self.date_column].dt.year
        df["month"] = df[self.date_column].dt.month
        df["day_of_week"] = df[self.date_column].dt.dayofweek
        df["day_of_month"] = df[self.date_column].dt.day
        df["quarter"] = df[self.date_column].dt.quarter
        
        # Cyclical encoding for seasonal features
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        
        # Is weekend
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        
        return df
    
    def _analyze_time_series_characteristics(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze time series statistical characteristics"""
        clean_series = series.dropna()
        
        if len(clean_series) < 10:
            return {"error": "Insufficient data for analysis"}
        
        characteristics = {
            "length": len(clean_series),
            "mean": float(clean_series.mean()),
            "std": float(clean_series.std()),
            "min": float(clean_series.min()),
            "max": float(clean_series.max()),
            "frequency": self.frequency
        }
        
        # Stationarity tests
        try:
            adf_stat, adf_pvalue, _, _, _, _ = adfuller(clean_series)
            characteristics["adf_statistic"] = float(adf_stat)
            characteristics["adf_pvalue"] = float(adf_pvalue)
            characteristics["is_stationary_adf"] = adf_pvalue < 0.05
        except Exception as e:
            logger.warning("ADF test failed", error=str(e))
            characteristics["is_stationary_adf"] = None
        
        # Simple seasonality detection using autocorrelation
        try:
            characteristics["seasonality"] = self._detect_seasonality_simple(clean_series)
        except Exception as e:
            logger.warning("Seasonality detection failed", error=str(e))
            characteristics["seasonality"] = {"detected": False}
        
        # Trend analysis using numpy
        try:
            x = np.arange(len(clean_series))
            y = clean_series.values
            A = np.vstack([x, np.ones(len(x))]).T
            slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
            
            # Calculate R-squared
            y_pred = slope * x + intercept
            ss_tot = np.sum((y - y.mean()) ** 2)
            ss_res = np.sum((y - y_pred) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            characteristics["trend_slope"] = float(slope)
            characteristics["trend_r_squared"] = float(r_squared)
            characteristics["trend_significant"] = r_squared > 0.25  # Simple threshold
        except Exception as e:
            logger.warning("Trend analysis failed", error=str(e))
        
        return characteristics
    
    def _detect_seasonality_simple(self, series: pd.Series) -> Dict[str, Any]:
        """Simple seasonality detection using autocorrelation"""
        n = len(series)
        
        # Calculate autocorrelation for different lags
        autocorrs = []
        max_lag = min(n // 4, 50)  # Don't go beyond 1/4 of series length or 50
        
        for lag in range(1, max_lag):
            if lag < n:
                corr = series.autocorr(lag=lag)
                if not np.isnan(corr):
                    autocorrs.append((lag, abs(corr)))
        
        if not autocorrs:
            return {"detected": False}
        
        # Find peaks in autocorrelation
        autocorrs.sort(key=lambda x: x[1], reverse=True)
        
        seasonality_info = {
            "detected": False,
            "periods": [],
            "strengths": []
        }
        
        # Check if we have significant autocorrelation
        if autocorrs and autocorrs[0][1] > 0.3:  # Threshold for seasonality
            seasonality_info["detected"] = True
            seasonality_info["dominant_period"] = float(autocorrs[0][0])
            seasonality_info["dominant_strength"] = float(autocorrs[0][1])
            
            # Add top periods
            for lag, strength in autocorrs[:3]:
                if strength > 0.2:
                    seasonality_info["periods"].append(float(lag))
                    seasonality_info["strengths"].append(float(strength))
        
        return seasonality_info
    
    def temporal_train_test_split(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split time series data maintaining temporal order"""
        
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        return train_df, test_df


class TimeSeriesMetrics:
    """Time series specific evaluation metrics"""
    
    @staticmethod
    def calculate_mape(actual: np.ndarray, forecast: np.ndarray) -> float:
        """Mean Absolute Percentage Error"""
        actual, forecast = np.array(actual), np.array(forecast)
        mask = actual != 0
        return np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])) * 100
    
    @staticmethod
    def calculate_smape(actual: np.ndarray, forecast: np.ndarray) -> float:
        """Symmetric Mean Absolute Percentage Error"""
        actual, forecast = np.array(actual), np.array(forecast)
        denominator = (np.abs(actual) + np.abs(forecast)) / 2
        mask = denominator != 0
        return np.mean(np.abs(forecast[mask] - actual[mask]) / denominator[mask]) * 100
    
    @staticmethod
    def calculate_mase(actual: np.ndarray, forecast: np.ndarray, seasonal_period: int = 1) -> float:
        """Mean Absolute Scaled Error"""
        actual, forecast = np.array(actual), np.array(forecast)
        
        # Calculate MAE of the forecast
        mae = np.mean(np.abs(actual - forecast))
        
        if len(actual) <= seasonal_period:
            return mae
        
        # Calculate MAE of the naive seasonal forecast
        seasonal_naive_errors = []
        for i in range(seasonal_period, len(actual)):
            naive_forecast = actual[i - seasonal_period]
            error = abs(actual[i] - naive_forecast)
            seasonal_naive_errors.append(error)
        
        seasonal_mae = np.mean(seasonal_naive_errors) if seasonal_naive_errors else 1
        
        return mae / (seasonal_mae if seasonal_mae != 0 else 1)
    
    @staticmethod
    def directional_accuracy(actual: np.ndarray, forecast: np.ndarray) -> float:
        """Percentage of times the direction of change is correctly predicted"""
        actual, forecast = np.array(actual), np.array(forecast)
        
        actual_diff = np.diff(actual)
        forecast_diff = np.diff(forecast)
        
        correct_direction = (actual_diff * forecast_diff) > 0
        return np.mean(correct_direction) * 100
    
    @staticmethod
    def comprehensive_evaluation(actual: np.ndarray, forecast: np.ndarray, seasonal_period: int = 1) -> Dict[str, float]:
        """Calculate multiple evaluation metrics"""
        actual, forecast = np.array(actual), np.array(forecast)
        
        metrics = {
            "mae": float(np.mean(np.abs(actual - forecast))),
            "rmse": float(np.sqrt(np.mean((actual - forecast) ** 2))),
            "mape": float(TimeSeriesMetrics.calculate_mape(actual, forecast)),
            "smape": float(TimeSeriesMetrics.calculate_smape(actual, forecast)),
            "mase": float(TimeSeriesMetrics.calculate_mase(actual, forecast, seasonal_period)),
            "directional_accuracy": float(TimeSeriesMetrics.directional_accuracy(actual, forecast))
        }
        
        return metrics


class SimpleMovingAverageForecaster:
    """Simple Moving Average forecaster"""
    
    def __init__(self, window: int = 3):
        self.window = window
        self.series = None
        
    def fit(self, series: pd.Series) -> Dict[str, Any]:
        """Fit the moving average model"""
        self.series = series
        return {"window": self.window, "model_type": "simple_moving_average"}
    
    def forecast(self, steps: int) -> Dict[str, np.ndarray]:
        """Generate forecasts"""
        if self.series is None:
            raise ValueError("Model must be fitted first")
        
        # Use last 'window' values to forecast
        last_values = self.series.tail(self.window).values
        forecast_value = np.mean(last_values)
        
        # Simple forecast: repeat the average
        forecast = np.full(steps, forecast_value)
        
        return {
            "forecast": forecast,
            "confidence_intervals": None
        }


class ARIMAForecaster:
    """ARIMA forecaster with auto-parameter selection"""
    
    def __init__(self):
        self.model = None
        self.fitted_model = None
        
    def auto_fit(self, series: pd.Series, seasonal: bool = True, max_p: int = 5, max_d: int = 2, max_q: int = 5) -> Dict[str, Any]:
        """Automatically fit ARIMA model with optimal parameters"""
        
        # Store training data for fallback forecasting
        self.train_data = series.copy()
        
        try:
            # Manual ARIMA fitting (pmdarima not available in Python 3.13)
            order = self._find_best_arima_order(series, max_p, max_d, max_q)
            seasonal_order = None
            self.fitted_model = ARIMA(series, order=order).fit()
            
            return {
                "order": order,
                "seasonal_order": seasonal_order,
                "aic": float(self.fitted_model.aic),
                "bic": float(self.fitted_model.bic) if hasattr(self.fitted_model, 'bic') else None,
                "model_type": "arima"
            }
            
        except Exception as e:
            logger.warning("ARIMA fitting failed", error=str(e))
            # Fallback to simple ARIMA(1,1,1)
            try:
                self.fitted_model = ARIMA(series, order=(1, 1, 1)).fit()
                return {
                    "order": (1, 1, 1),
                    "seasonal_order": None,
                    "aic": float(self.fitted_model.aic),
                    "model_type": "arima_fallback"
                }
            except Exception as e2:
                logger.error("ARIMA fallback failed", error=str(e2))
                raise e2
    
    def _find_best_arima_order(self, series: pd.Series, max_p: int, max_d: int, max_q: int) -> Tuple[int, int, int]:
        """Find best ARIMA order using common patterns for Python 3.13 compatibility"""
        best_aic = float('inf')
        best_order = (1, 1, 1)  # Default fallback
        
        # Test common ARIMA orders that work well for most time series
        common_orders = [
            (0, 1, 1),   # Simple moving average
            (1, 1, 0),   # Simple trend
            (1, 1, 1),   # Classic ARIMA
            (2, 1, 2),   # More complex
            (0, 1, 2),   # Moving average with trend
            (2, 1, 0),   # Autoregressive with trend
            (1, 0, 1),   # ARMA (no differencing)
            (0, 0, 1),   # MA only
            (1, 0, 0),   # AR only
        ]
        
        for p, d, q in common_orders:
            if p <= max_p and d <= max_d and q <= max_q:
                try:
                    model = ARIMA(series, order=(p, d, q))
                    fitted = model.fit()
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                except Exception as e:
                    logger.debug(f"ARIMA({p},{d},{q}) failed: {str(e)}")
                    continue
        
        logger.info(f"Selected ARIMA order {best_order} with AIC {best_aic:.2f}")
        return best_order
    
    def forecast(self, steps: int, confidence_interval: float = 0.95) -> Dict[str, np.ndarray]:
        """Generate forecasts with confidence intervals"""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
        
        try:
            forecast_result = self.fitted_model.get_forecast(steps=steps)
            forecast_values = forecast_result.predicted_mean.values
            
            # Get confidence intervals
            conf_int = forecast_result.conf_int()
            confidence_intervals = [(conf_int.iloc[i, 0], conf_int.iloc[i, 1]) for i in range(len(conf_int))]
            
            return {
                "forecast": forecast_values,
                "confidence_intervals": confidence_intervals
            }
        except Exception as e:
            logger.warning("ARIMA forecast failed", error=str(e))
            # Fallback: use last value from training data
            if hasattr(self, 'train_data') and len(self.train_data) > 0:
                last_value = self.train_data.iloc[-1]
                fallback_forecast = np.full(steps, last_value)
                return {
                    "forecast": fallback_forecast,
                    "confidence_intervals": [(last_value * 0.9, last_value * 1.1) for _ in range(steps)]
                }
            else:
                raise e


class ExponentialSmoothingForecaster:
    """Exponential Smoothing forecaster with trend and seasonal components"""
    
    def __init__(self):
        self.model = None
        self.fitted_model = None
        self.train_data = None
        
    def fit(self, series: pd.Series, trend: str = None, seasonal: str = None, seasonal_periods: int = None) -> Dict[str, Any]:
        """Fit Exponential Smoothing model with trend and seasonal options"""
        
        # Store training data for fallback forecasting
        self.train_data = series.copy()
        
        try:
            # Auto-detect seasonal periods if not provided
            if seasonal_periods is None:
                seasonal_periods = min(12, len(series) // 4) if len(series) > 24 else None
            
            # Try different configurations in order of complexity
            configs_to_try = [
                {"trend": trend, "seasonal": seasonal, "seasonal_periods": seasonal_periods},
                {"trend": "add", "seasonal": "add", "seasonal_periods": seasonal_periods},
                {"trend": "add", "seasonal": None, "seasonal_periods": None},
                {"trend": None, "seasonal": None, "seasonal_periods": None},
            ]
            
            best_aic = float('inf')
            best_config = None
            best_model = None
            
            for config in configs_to_try:
                try:
                    if config["seasonal_periods"] and len(series) < 2 * config["seasonal_periods"]:
                        # Not enough data for seasonal component
                        config["seasonal"] = None
                        config["seasonal_periods"] = None
                    
                    model = ExponentialSmoothing(
                        series,
                        trend=config["trend"],
                        seasonal=config["seasonal"],
                        seasonal_periods=config["seasonal_periods"]
                    )
                    fitted = model.fit()
                    
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_config = config
                        best_model = fitted
                        
                except Exception as e:
                    logger.debug(f"Exponential Smoothing config {config} failed: {str(e)}")
                    continue
            
            if best_model is None:
                raise ValueError("All Exponential Smoothing configurations failed")
            
            self.fitted_model = best_model
            
            return {
                "trend": best_config["trend"],
                "seasonal": best_config["seasonal"],
                "seasonal_periods": best_config["seasonal_periods"],
                "aic": float(best_aic),
                "model_type": "exponential_smoothing"
            }
            
        except Exception as e:
            logger.error("Exponential Smoothing fitting failed", error=str(e))
            raise e
    
    def forecast(self, steps: int) -> Dict[str, np.ndarray]:
        """Generate forecasts"""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
        
        try:
            forecast_values = self.fitted_model.forecast(steps=steps)
            
            # Simple confidence intervals (±10% of forecast value)
            confidence_intervals = [(val * 0.9, val * 1.1) for val in forecast_values]
            
            return {
                "forecast": forecast_values,
                "confidence_intervals": confidence_intervals
            }
        except Exception as e:
            logger.warning("Exponential Smoothing forecast failed", error=str(e))
            # Fallback: use last value from training data
            if self.train_data is not None and len(self.train_data) > 0:
                last_value = self.train_data.iloc[-1]
                fallback_forecast = np.full(steps, last_value)
                return {
                    "forecast": fallback_forecast,
                    "confidence_intervals": [(last_value * 0.9, last_value * 1.1) for _ in range(steps)]
                }
            else:
                raise e


class ProphetForecaster:
    """Prophet forecaster (if available)"""
    
    def __init__(self):
        self.model = None
        self.fitted_data = None
        
    def fit(self, series: pd.Series) -> Dict[str, Any]:
        """Fit Prophet model with simple interface (for consistency)"""
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet not available. Install with: pip install prophet")
        
        # Create a simple dataframe for Prophet
        df = pd.DataFrame({
            'ds': pd.date_range(start='2020-01-01', periods=len(series), freq='D'),
            'y': series.values
        })
        
        try:
            self.model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False
            )
            self.model.fit(df)
            self.fitted_data = df
            
            return {
                "yearly_seasonality": True,
                "weekly_seasonality": True,
                "daily_seasonality": False,
                "model_type": "prophet"
            }
            
        except Exception as e:
            logger.error("Prophet fitting failed", error=str(e))
            raise e
        
    def fit_with_seasonality(self, df: pd.DataFrame, date_col: str, target_col: str, 
                           yearly: bool = True, weekly: bool = True, daily: bool = False) -> Dict[str, Any]:
        """Fit Prophet model with seasonality options"""
        
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet not available. Install with: pip install prophet")
        
        # Prepare data for Prophet
        prophet_df = df[[date_col, target_col]].copy()
        prophet_df.columns = ['ds', 'y']
        
        try:
            self.model = Prophet(
                yearly_seasonality=yearly,
                weekly_seasonality=weekly,
                daily_seasonality=daily
            )
            self.model.fit(prophet_df)
            self.fitted_data = prophet_df
            
            return {
                "yearly_seasonality": yearly,
                "weekly_seasonality": weekly,
                "daily_seasonality": daily,
                "model_type": "prophet"
            }
            
        except Exception as e:
            logger.error("Prophet fitting failed", error=str(e))
            raise e

    def forecast(self, steps: int) -> Dict[str, np.ndarray]:
        """Generate forecasts (consistent interface with other forecasters)"""
        if self.model is None:
            raise ValueError("Model must be fitted first")
        
        # Determine frequency from fitted data
        if self.fitted_data is not None and len(self.fitted_data) > 1:
            freq = pd.infer_freq(self.fitted_data['ds'])
            if freq is None:
                freq = 'D'  # Default to daily
        else:
            freq = 'D'
        
        future = self.model.make_future_dataframe(periods=steps, freq=freq)
        forecast = self.model.predict(future)
        
        # Get only the forecasted values (last 'steps' values)
        forecast_values = forecast['yhat'].tail(steps).values
        
        return {
            "forecast": forecast_values
        }
    
    def forecast_with_uncertainty(self, periods: int, freq: str = 'D') -> pd.DataFrame:
        """Generate forecasts with uncertainty intervals"""
        if self.model is None:
            raise ValueError("Model must be fitted first")
        
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        forecast = self.model.predict(future)
        
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)


class TimeSeriesTrainer:
    """Main time series trainer class"""
    
    def __init__(self, run_id: str, session_id: str, websocket_manager):
        self.run_id = run_id
        self.session_id = session_id
        self.websocket_manager = websocket_manager
        self.model_selector = ModelSelector()
        self.start_time = time.time()  # Track training start time
        
    async def train_forecast_models(
        self,
        dataset_path: str,
        date_column: str,
        target_column: str,
        forecast_horizon: int,
        data_profile: DataProfile,
        constraints: Dict[str, Any]
    ) -> ModelArtifact:
        """Train time series forecasting models"""
        
        logger.info("Starting time series model training", 
                   dataset_path=dataset_path, 
                   forecast_horizon=forecast_horizon)
        
        # Load dataset
        df = pd.read_csv(dataset_path)
        
        # Initialize data pipeline
        data_pipeline = TimeSeriesDataPipeline(date_column, target_column)
        
        # Prepare time series data
        prepared_df, ts_characteristics = data_pipeline.prepare_time_series_data(df)
        
        # Split data temporally
        train_df, test_df = data_pipeline.temporal_train_test_split(prepared_df, test_size=0.2)
        
        # Convert characteristics to JSON-serializable format
        serializable_characteristics = self._make_json_serializable(ts_characteristics)
        
        await self._send_training_status_update("data_prepared", {
            "train_size": len(train_df),
            "test_size": len(test_df),
            "characteristics": serializable_characteristics
        })
        
        # Get model recommendations using the ensemble recommender (supports time series)
        ensemble_strategy = await self.model_selector.recommend_ensemble(
            data_profile=data_profile,
            task_type="forecasting",
            target_column=target_column,
            constraints=constraints
        )
        model_recommendations = ensemble_strategy.recommended_models
        
        # Filter recommendations based on user selection if provided
        selected_models = constraints.get("selected_models", [])
        if selected_models:
            logger.info("👤 User selected specific time series models", selected_models=selected_models)
            
            # Filter to only selected model types
            filtered_recommendations = []
            for model_rec in model_recommendations:
                if model_rec.model_type in selected_models:
                    filtered_recommendations.append(model_rec)
                    logger.info("✅ Including user-selected model", model_type=model_rec.model_type)
                else:
                    logger.info("❌ Excluding non-selected model", model_type=model_rec.model_type)
            
            if filtered_recommendations:
                model_recommendations = filtered_recommendations
            else:
                logger.warning("🚨 No valid models found in user selection, using all recommendations")
        
        await self._send_training_status_update("models_selected", {
            "recommended_models": [rec.model_type for rec in model_recommendations],
            "user_selected": bool(len(selected_models) > 0)
        })
        
        # Train models in parallel with progress tracking
        trained_models = []
        training_tasks = []
        all_model_performances = []  # Initialize here to collect all model results
        
        # Send initial training start update
        await self._send_training_status_update("parallel_training_started", {
            "total_models": len(model_recommendations),
            "model_types": [rec.model_type for rec in model_recommendations],
            "status": "starting_parallel_training"
        })
        
        # Create training tasks for parallel execution
        for model_rec in model_recommendations:
            task = asyncio.create_task(
                self._train_ts_model_with_progress(
                    model_rec, train_df, test_df, data_pipeline, 
                    forecast_horizon, ts_characteristics
                )
            )
            training_tasks.append((model_rec.model_type, task))
            logger.info(f"Created training task for {model_rec.model_type}")
        
        # Wait for all models to complete using asyncio.gather for true parallelism
        logger.info(f"Starting parallel training of {len(training_tasks)} models")
        results = await asyncio.gather(*[task for _, task in training_tasks], return_exceptions=True)
        
        # Process results
        for i, ((model_type, task), result) in enumerate(zip(training_tasks, results)):
            try:
                if isinstance(result, Exception):
                    logger.error(f"Model {model_type} failed with exception", error=str(result))
                    await self._send_training_status_update("model_failed", {
                        "model_type": model_type,
                        "status": "failed",
                        "error": str(result)
                    })
                    # Add failed model to collection for UI display
                    all_model_performances.append({
                        "model_type": model_type,
                        "rmse": float('inf'),  # High RMSE to indicate failure
                        "status": "failed",
                        "error": str(result)
                    })
                elif result is not None:
                    trained_models.append(result)
                    logger.info(f"Model {model_type} completed successfully with RMSE: {result.val_score}")
                    await self._send_training_status_update("model_completed", {
                        "model_type": model_type,
                        "rmse": result.val_score,
                        "status": "completed"
                    })
                    # Add successful model to collection for UI display
                    all_model_performances.append({
                        "model_type": model_type,
                        "rmse": result.val_score,
                        "status": "completed"
                    })
                else:
                    logger.warning(f"Model {model_type} returned None result")
                    await self._send_training_status_update("model_failed", {
                        "model_type": model_type,
                        "status": "failed",
                        "error": "Model returned None result"
                    })
                    # Add failed model to collection for UI display
                    all_model_performances.append({
                        "model_type": model_type,
                        "rmse": float('inf'),  # High RMSE to indicate failure
                        "status": "failed",
                        "error": "Model returned None result"
                    })
                    
            except Exception as e:
                logger.error("Error processing model result", 
                           model_type=model_type, 
                           error=str(e))
                await self._send_training_status_update("model_failed", {
                    "model_type": model_type,
                    "status": "failed",
                    "error": str(e)
                })
                continue
        
        # Send completion summary
        await self._send_training_status_update("parallel_training_completed", {
            "total_models": len(model_recommendations),
            "successful_models": len(trained_models),
            "failed_models": len(model_recommendations) - len(trained_models),
            "models_completed": [model.family for model in trained_models]
        })
        
        # Only create fallback if no models trained and no specific user selection
        if not trained_models:
            if not selected_models:
                # No user selection - create fallback
                logger.info("No models trained successfully, creating fallback model")
                fallback_model = await self._create_fallback_model(
                    train_df, test_df, data_pipeline, forecast_horizon, ts_characteristics
                )
                trained_models.append(fallback_model)
            else:
                # User selected specific models but they all failed
                logger.error("All user-selected models failed to train")
                raise ValueError(f"All selected models ({selected_models}) failed to train. Please try different models or check your data.")
        
        # Select best model
        best_model = self._select_best_ts_model(trained_models)
        
        # Generate forecast data for visualization
        forecast_data = await self._generate_forecast_visualization_data(
            best_model, train_df, test_df, data_pipeline, forecast_horizon
        )
        
        # Store forecast data for later retrieval
        self.forecast_data = forecast_data
        
        # Store all model performances for later retrieval (collected during training loop)
        self.all_model_performances = all_model_performances
        
        await self._send_training_status_update("training_complete", {
            "best_model": best_model.family,
            "performance": {"rmse": best_model.val_score},
            "forecast_data": forecast_data,
            "all_models": all_model_performances
        })
        
        return best_model
    
    async def _train_ts_model_with_progress(
        self,
        model_rec: ModelRecommendation,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        data_pipeline: TimeSeriesDataPipeline,
        forecast_horizon: int,
        ts_characteristics: Dict[str, Any]
    ) -> Optional[ModelArtifact]:
        """Train a specific time series model with detailed progress tracking"""
        
        model_type = model_rec.model_type
        
        # Send initial training start update
        await self._send_training_status_update("model_training_started", {
            "model_type": model_type,
            "status": "training",
            "progress": 0
        })
        
        try:
            # Phase 1: Data preparation (10% progress)
            await self._send_training_status_update("model_training_progress", {
                "model_type": model_type,
                "status": "training",
                "progress": 10,
                "phase": "data_preparation"
            })
            
            target_col = data_pipeline.target_column
            train_series = train_df[target_col].dropna()
            test_series = test_df[target_col].dropna()
            
            # Phase 2: Model fitting (20-70% progress)
            await self._send_training_status_update("model_training_progress", {
                "model_type": model_type,
                "status": "training",
                "progress": 20,
                "phase": "model_fitting"
            })
            
            # Add realistic delay for UI visibility
            await asyncio.sleep(1)
            
            forecaster = None
            fit_info = None
            
            if model_type == "arima":
                await self._send_training_status_update("model_training_progress", {
                    "model_type": model_type,
                    "status": "training",
                    "progress": 30,
                    "phase": "arima_parameter_search"
                })
                forecaster = ARIMAForecaster()
                fit_info = forecaster.auto_fit(train_series)
                
                # Validate that ARIMA model was fitted successfully
                if forecaster.fitted_model is None:
                    raise ValueError("ARIMA model failed to fit")
                
                await self._send_training_status_update("model_training_progress", {
                    "model_type": model_type,
                    "status": "training",
                    "progress": 60,
                    "phase": "arima_fitting_complete"
                })
                
            elif model_type == "simple_moving_average":
                await self._send_training_status_update("model_training_progress", {
                    "model_type": model_type,
                    "status": "training",
                    "progress": 40,
                    "phase": "moving_average_calculation"
                })
                window = min(7, len(train_series) // 4)  # Adaptive window
                forecaster = SimpleMovingAverageForecaster(window=window)
                fit_info = forecaster.fit(train_series)
                
            elif model_type == "prophet" and PROPHET_AVAILABLE:
                await self._send_training_status_update("model_training_progress", {
                    "model_type": model_type,
                    "status": "training",
                    "progress": 35,
                    "phase": "prophet_seasonality_detection"
                })
                forecaster = ProphetForecaster()
                fit_info = forecaster.fit_with_seasonality(
                    train_df, data_pipeline.date_column, target_col
                )
                await self._send_training_status_update("model_training_progress", {
                    "model_type": model_type,
                    "status": "training",
                    "progress": 55,
                    "phase": "prophet_fitting_complete"
                })
                
            elif model_type == "exponential_smoothing":
                await self._send_training_status_update("model_training_progress", {
                    "model_type": model_type,
                    "status": "training",
                    "progress": 35,
                    "phase": "exponential_smoothing_config"
                })
                forecaster = ExponentialSmoothingForecaster()
                fit_info = forecaster.fit(train_series)
                await self._send_training_status_update("model_training_progress", {
                    "model_type": model_type,
                    "status": "training",
                    "progress": 55,
                    "phase": "exponential_smoothing_complete"
                })
                
            else:
                logger.warning("Unsupported model type", model_type=model_type)
                return None
            
            # Add delay after fitting
            await asyncio.sleep(0.5)
            
            # Phase 3: Forecasting (70-80% progress)
            await self._send_training_status_update("model_training_progress", {
                "model_type": model_type,
                "status": "training",
                "progress": 70,
                "phase": "forecasting"
            })
            
            forecast_result = forecaster.forecast(len(test_series))
            forecast = forecast_result["forecast"]
            
            # Validate forecast
            if forecast is None or len(forecast) == 0:
                raise ValueError(f"{model_type} forecast returned empty results")
            
            # Ensure forecast length matches test series length
            if len(forecast) != len(test_series):
                logger.warning(f"{model_type} forecast length mismatch: expected {len(test_series)}, got {len(forecast)}")
                # Truncate or pad forecast to match test series length
                if len(forecast) > len(test_series):
                    forecast = forecast[:len(test_series)]
                else:
                    # Pad with last value
                    last_val = forecast[-1] if len(forecast) > 0 else 0.0
                    forecast = np.concatenate([forecast, np.full(len(test_series) - len(forecast), last_val)])
            
            # Phase 4: Evaluation (80-90% progress)
            await self._send_training_status_update("model_training_progress", {
                "model_type": model_type,
                "status": "training",
                "progress": 80,
                "phase": "evaluation"
            })
            
            seasonal_period = self._get_seasonal_period(ts_characteristics)
            metrics = TimeSeriesMetrics.comprehensive_evaluation(
                test_series.values, forecast, seasonal_period
            )
            
            # Phase 5: Model saving (90-100% progress)
            await self._send_training_status_update("model_training_progress", {
                "model_type": model_type,
                "status": "training",
                "progress": 90,
                "phase": "saving"
            })
            
            model_path = self._save_ts_model(forecaster, f"{model_type}_{self.run_id}")
            
            # Final completion (100% progress)
            await self._send_training_status_update("model_training_progress", {
                "model_type": model_type,
                "status": "training",
                "progress": 100,
                "phase": "completed"
            })
            
            return ModelArtifact(
                run_id=self.run_id,
                family=model_type,
                model_path=model_path,
                val_score=metrics.get("rmse", 0.0),
                train_score=0.0,  # Placeholder - we don't have train score for time series
                feature_importance=None
            )
            
        except Exception as e:
            logger.error("Time series model training failed", 
                        model_type=model_type, 
                        error=str(e))
            
            # Send failure update
            await self._send_training_status_update("model_training_failed", {
                "model_type": model_type,
                "status": "failed",
                "error": str(e)
            })
            return None
    
    async def _train_ts_model(
        self,
        model_rec: ModelRecommendation,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        data_pipeline: TimeSeriesDataPipeline,
        forecast_horizon: int,
        ts_characteristics: Dict[str, Any]
    ) -> Optional[ModelArtifact]:
        """Train a specific time series model (legacy method for compatibility)"""
        return await self._train_ts_model_with_progress(
            model_rec, train_df, test_df, data_pipeline, forecast_horizon, ts_characteristics
        )
    
    async def _create_fallback_model(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        data_pipeline: TimeSeriesDataPipeline,
        forecast_horizon: int,
        ts_characteristics: Dict[str, Any]
    ) -> ModelArtifact:
        """Create a simple fallback model"""
        
        target_col = data_pipeline.target_column
        train_series = train_df[target_col].dropna()
        test_series = test_df[target_col].dropna()
        
        # Simple moving average fallback
        forecaster = SimpleMovingAverageForecaster(window=3)
        fit_info = forecaster.fit(train_series)
        forecast_result = forecaster.forecast(len(test_series))
        forecast = forecast_result["forecast"]
        
        # Evaluate
        metrics = TimeSeriesMetrics.comprehensive_evaluation(
            test_series.values, forecast, 1
        )
        
        # Save
        model_path = self._save_ts_model(forecaster, f"fallback_{self.run_id}")
        
        return ModelArtifact(
            run_id=self.run_id,
            family="simple_moving_average",
            model_path=model_path,
            val_score=metrics.get("rmse", 0.0),
            train_score=0.0,  # Placeholder - we don't have train score for time series
            feature_importance=None
        )
    
    def _get_seasonal_period(self, ts_characteristics: Dict[str, Any]) -> int:
        """Get seasonal period from characteristics"""
        try:
            seasonality = ts_characteristics.get("seasonality", {})
            if seasonality.get("detected", False):
                return int(seasonality.get("dominant_period", 1))
            else:
                frequency = ts_characteristics.get("frequency", "D")
                if frequency == "D":
                    return 7  # Weekly seasonality for daily data
                elif frequency == "W":
                    return 4  # Monthly seasonality for weekly data
                elif frequency == "M":
                    return 12  # Yearly seasonality for monthly data
                else:
                    return 1
        except:
            return 1
    
    def _save_ts_model(self, model: Any, filename: str) -> str:
        """Save time series model to disk"""
        models_dir = Path("models/timeseries")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = models_dir / f"{filename}.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        return str(model_path)
    
    def _select_best_ts_model(self, trained_models: List[ModelArtifact]) -> ModelArtifact:
        """Select the best performing model"""
        if not trained_models:
            raise ValueError("No trained models available")
        
        # Sort by val_score (RMSE - lower is better)
        best_model = min(trained_models, key=lambda m: m.val_score)
        return best_model
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert numpy types and other non-JSON-serializable objects to JSON-serializable types"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif obj is None:
            return None
        else:
            return obj

    async def _send_training_status_update(self, status: str, data: Dict[str, Any]):
        """Send training status update via WebSocket"""
        if self.websocket_manager:
            # Make sure data is JSON serializable
            serializable_data = self._make_json_serializable(data)
            
            # Format for UI compatibility - different formats for different status types
            if "model_training_progress" in status or "model_training_started" in status:
                # Send progress updates for individual models
                model_type = data.get("model_type", "unknown")
                progress = data.get("progress", 0)
                phase = data.get("phase", "training")
                
                # Send as trial update for UI compatibility
                await self.websocket_manager.broadcast_to_session(
                    self.session_id,
                    {
                        "type": "trial_update",
                        "family": model_type,
                        "trial": max(1, progress // 20),  # Convert progress to trial number
                        "val_metric": None,
                        "elapsed_s": time.time() - getattr(self, 'start_time', time.time()),
                        "message": f"{model_type}: {phase} ({progress}%)",
                        "status": "running"
                    }
                )
                
            elif "model_completed" in status:
                # Send completion update for individual models
                model_type = data.get("model_type", "unknown")
                rmse = data.get("rmse", 0)
                
                await self.websocket_manager.broadcast_to_session(
                    self.session_id,
                    {
                        "type": "family_complete",
                        "family": model_type,
                        "val_metric": rmse,
                        "status": "completed"
                    }
                )
                
            elif "model_failed" in status:
                # Send failure update for individual models
                model_type = data.get("model_type", "unknown")
                error = data.get("error", "Unknown error")
                
                await self.websocket_manager.broadcast_to_session(
                    self.session_id,
                    {
                        "type": "trial_update",
                        "family": model_type,
                        "trial": 1,
                        "val_metric": None,
                        "elapsed_s": time.time() - getattr(self, 'start_time', time.time()),
                        "message": f"{model_type}: Failed - {error}",
                        "status": "failed"
                    }
                )
            
            # Also send the original format for debugging
            await self.websocket_manager.broadcast_training_status(
                self.session_id, 
                {
                    "type": "timeseries_training_update",
                    "run_id": self.run_id,
                    "status": status,
                    "data": serializable_data,
                    "timestamp": time.time()
                }
            )
    
    async def _generate_forecast_visualization_data(
        self,
        best_model: ModelArtifact,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        data_pipeline: TimeSeriesDataPipeline,
        forecast_horizon: int
    ) -> Dict[str, Any]:
        """Generate forecast data for visualization"""
        
        try:
            # Load the trained model
            with open(best_model.model_path, 'rb') as f:
                forecaster = pickle.load(f)
            
            target_col = data_pipeline.target_column
            date_col = data_pipeline.date_column
            
            # Prepare historical data
            historical_data = []
            train_series = train_df[target_col].dropna()
            test_series = test_df[target_col].dropna()
            
            # Get dates for historical data
            train_dates = train_df[date_col].dropna()
            test_dates = test_df[date_col].dropna()
            
            # Historical training data (last 30 points for visualization)
            train_limit = min(30, len(train_series))
            for i in range(len(train_series) - train_limit, len(train_series)):
                if i < len(train_dates):
                    date = train_dates.iloc[i]
                    value = train_series.iloc[i]
                    historical_data.append({
                        "date": date.isoformat() if hasattr(date, 'isoformat') else str(date),
                        "actual": float(value),
                        "type": "historical"
                    })
            
            # Generate forecast for test period (validation)
            forecast_result = forecaster.forecast(len(test_series))
            forecast_values = forecast_result["forecast"]
            
            # Test data with predictions
            forecast_data = []
            for i, (date, actual, predicted) in enumerate(zip(test_dates, test_series, forecast_values)):
                forecast_data.append({
                    "date": date.isoformat() if hasattr(date, 'isoformat') else str(date),
                    "actual": float(actual),
                    "predicted": float(predicted),
                    "type": "validation"
                })
            
            # Generate future forecast
            future_forecast = forecaster.forecast(forecast_horizon)
            future_values = future_forecast["forecast"]
            
            # Generate future dates
            last_date = test_dates.iloc[-1] if len(test_dates) > 0 else train_dates.iloc[-1]
            future_dates = pd.date_range(start=last_date, periods=forecast_horizon + 1, freq=data_pipeline.frequency)[1:]
            
            # Future predictions
            future_data = []
            for i, (date, predicted) in enumerate(zip(future_dates, future_values)):
                future_data.append({
                    "date": date.isoformat() if hasattr(date, 'isoformat') else str(date),
                    "predicted": float(predicted),
                    "type": "future"
                })
            
            # Add confidence intervals if available
            if future_forecast.get("confidence_intervals") is not None:
                conf_int = future_forecast["confidence_intervals"]
                for i, data_point in enumerate(future_data):
                    if i < len(conf_int):
                        data_point["confidence_lower"] = float(conf_int[i][0])
                        data_point["confidence_upper"] = float(conf_int[i][1])
            
            return {
                "historical_data": historical_data,
                "forecast_data": forecast_data,
                "future_data": future_data,
                "model_type": best_model.family,
                "rmse": best_model.val_score,
                "forecast_horizon": forecast_horizon,
                "target_column": target_col,
                "date_column": date_col
            }
            
        except Exception as e:
            logger.warning("Failed to generate forecast visualization data", error=str(e))
            return {
                "error": f"Failed to generate forecast visualization: {str(e)}",
                "model_type": best_model.family,
                "rmse": best_model.val_score
            } 