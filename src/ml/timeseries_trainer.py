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

# Auto-ARIMA
try:
    from pmdarima import auto_arima
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False
    warnings.warn("pmdarima not available. Install with: pip install pmdarima")

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
        
        try:
            # Try to use auto_arima if available
            if PMDARIMA_AVAILABLE:
                self.fitted_model = auto_arima(
                    series,
                    seasonal=seasonal,
                    max_p=max_p,
                    max_d=max_d,
                    max_q=max_q,
                    suppress_warnings=True,
                    error_action='ignore'
                )
                order = self.fitted_model.order
                seasonal_order = self.fitted_model.seasonal_order if seasonal else None
            else:
                # Manual ARIMA fitting
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
        """Find best ARIMA order using grid search"""
        best_aic = float('inf')
        best_order = (1, 1, 1)
        
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(series, order=(p, d, q)).fit()
                        if model.aic < best_aic:
                            best_aic = model.aic
                            best_order = (p, d, q)
                    except:
                        continue
        
        return best_order
    
    def forecast(self, steps: int, confidence_interval: float = 0.95) -> Dict[str, np.ndarray]:
        """Generate forecasts with confidence intervals"""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
        
        try:
            forecast_result = self.fitted_model.forecast(steps=steps)
            
            if hasattr(forecast_result, 'predicted_mean'):
                forecast = forecast_result.predicted_mean.values
                conf_int = forecast_result.conf_int().values if hasattr(forecast_result, 'conf_int') else None
            else:
                forecast = forecast_result if isinstance(forecast_result, np.ndarray) else np.array([forecast_result])
                conf_int = None
            
            return {
                "forecast": forecast,
                "confidence_intervals": conf_int
            }
            
        except Exception as e:
            logger.warning("ARIMA forecast failed", error=str(e))
            # Fallback to last value
            last_value = self.fitted_model.fittedvalues.iloc[-1] if hasattr(self.fitted_model, 'fittedvalues') else 0
            return {
                "forecast": np.full(steps, last_value),
                "confidence_intervals": None
            }


class ProphetForecaster:
    """Prophet forecaster (if available)"""
    
    def __init__(self):
        self.model = None
        
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
            
            return {
                "yearly_seasonality": yearly,
                "weekly_seasonality": weekly,
                "daily_seasonality": daily,
                "model_type": "prophet"
            }
            
        except Exception as e:
            logger.error("Prophet fitting failed", error=str(e))
            raise e
    
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
        
        await self._send_training_status_update("data_prepared", {
            "train_size": len(train_df),
            "test_size": len(test_df),
            "characteristics": ts_characteristics
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
            "user_selected": len(selected_models) > 0
        })
        
        # Train models
        trained_models = []
        for i, model_rec in enumerate(model_recommendations):
            try:
                await self._send_training_status_update("training_model", {
                    "model_type": model_rec.model_type,
                    "progress": f"{i+1}/{len(model_recommendations)}"
                })
                
                model_artifact = await self._train_ts_model(
                    model_rec, train_df, test_df, data_pipeline, 
                    forecast_horizon, ts_characteristics
                )
                
                if model_artifact:
                    trained_models.append(model_artifact)
                    
            except Exception as e:
                logger.warning("Model training failed", 
                             model_type=model_rec.model_type, 
                             error=str(e))
                continue
        
        if not trained_models:
            # Fallback strategy
            fallback_model = await self._create_fallback_model(
                train_df, test_df, data_pipeline, forecast_horizon, ts_characteristics
            )
            trained_models.append(fallback_model)
        
        # Select best model
        best_model = self._select_best_ts_model(trained_models)
        
        await self._send_training_status_update("training_complete", {
            "best_model": best_model.model_type,
            "performance": best_model.performance_metrics
        })
        
        return best_model
    
    async def _train_ts_model(
        self,
        model_rec: ModelRecommendation,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        data_pipeline: TimeSeriesDataPipeline,
        forecast_horizon: int,
        ts_characteristics: Dict[str, Any]
    ) -> Optional[ModelArtifact]:
        """Train a specific time series model"""
        
        target_col = data_pipeline.target_column
        train_series = train_df[target_col].dropna()
        test_series = test_df[target_col].dropna()
        
        try:
            if model_rec.model_type == "arima":
                forecaster = ARIMAForecaster()
                fit_info = forecaster.auto_fit(train_series)
                forecast_result = forecaster.forecast(len(test_series))
                forecast = forecast_result["forecast"]
                
            elif model_rec.model_type == "simple_moving_average":
                window = min(7, len(train_series) // 4)  # Adaptive window
                forecaster = SimpleMovingAverageForecaster(window=window)
                fit_info = forecaster.fit(train_series)
                forecast_result = forecaster.forecast(len(test_series))
                forecast = forecast_result["forecast"]
                
            elif model_rec.model_type == "prophet" and PROPHET_AVAILABLE:
                forecaster = ProphetForecaster()
                fit_info = forecaster.fit_with_seasonality(
                    train_df, data_pipeline.date_column, target_col
                )
                prophet_forecast = forecaster.forecast_with_uncertainty(len(test_series))
                forecast = prophet_forecast['yhat'].values
                
            else:
                logger.warning("Unsupported model type", model_type=model_rec.model_type)
                return None
            
            # Evaluate model
            seasonal_period = self._get_seasonal_period(ts_characteristics)
            metrics = TimeSeriesMetrics.comprehensive_evaluation(
                test_series.values, forecast, seasonal_period
            )
            
            # Save model
            model_path = self._save_ts_model(forecaster, f"{model_rec.model_type}_{self.run_id}")
            
            return ModelArtifact(
                model_type=model_rec.model_type,
                model_path=model_path,
                performance_metrics=metrics,
                training_config=fit_info,
                feature_importance=None,
                model_size_mb=0.1,  # Placeholder
                training_time_seconds=1.0,  # Placeholder
                cross_validation_scores=None
            )
            
        except Exception as e:
            logger.error("Time series model training failed", 
                        model_type=model_rec.model_type, 
                        error=str(e))
            return None
    
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
            model_type="simple_moving_average",
            model_path=model_path,
            performance_metrics=metrics,
            training_config=fit_info,
            feature_importance=None,
            model_size_mb=0.01,
            training_time_seconds=0.1,
            cross_validation_scores=None
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
        
        # Sort by RMSE (lower is better)
        best_model = min(trained_models, key=lambda m: m.performance_metrics.get("rmse", float('inf')))
        return best_model
    
    async def _send_training_status_update(self, status: str, data: Dict[str, Any]):
        """Send training status update via WebSocket"""
        if self.websocket_manager:
            await self.websocket_manager.send_progress_update(
                self.session_id, 
                {
                    "type": "timeseries_training_update",
                    "run_id": self.run_id,
                    "status": status,
                    "data": data,
                    "timestamp": time.time()
                }
            ) 