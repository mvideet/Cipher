"""
Example: Time Series Forecasting with Cipher Platform
This example shows how to use the new time series capabilities
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create sample time series data
def create_sample_data():
    """Create sample sales data with trend and seasonality"""
    
    # Generate dates for 2 years of daily data
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    n_points = len(dates)
    
    # Create realistic sales pattern
    # Base trend (growing business)
    trend = np.linspace(1000, 1500, n_points)
    
    # Yearly seasonality (higher sales in winter holidays)
    yearly_seasonal = 200 * np.sin(2 * np.pi * np.arange(n_points) / 365.25 + np.pi)
    
    # Weekly seasonality (higher sales on weekends)
    weekly_seasonal = 100 * np.sin(2 * np.pi * np.arange(n_points) / 7 + np.pi/2)
    
    # Random noise
    noise = np.random.normal(0, 50, n_points)
    
    # Combine components
    sales = trend + yearly_seasonal + weekly_seasonal + noise
    
    # Ensure positive values
    sales = np.maximum(sales, 100)
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'sales': sales,
        'marketing_spend': np.random.normal(5000, 1000, n_points),
        'temperature': 20 + 15 * np.sin(2 * np.pi * np.arange(n_points) / 365.25)
    })
    
    return df

# Example 1: Data Profiling
def example_data_profiling():
    """Example of time series data profiling"""
    
    print("🔍 Time Series Data Profiling Example")
    print("=" * 50)
    
    # Create sample data
    df = create_sample_data()
    print(f"Created sample sales data: {len(df)} days from {df['date'].min()} to {df['date'].max()}")
    
    # The data profiler would work like this:
    # (This is pseudocode since we'd need the full environment)
    """
    from src.ml.data_profiler import DataProfiler
    
    profiler = DataProfiler()
    
    # Basic profiling
    profile = profiler.profile_dataset(df)
    print(f"Dataset: {profile.n_rows} rows × {profile.n_cols} columns")
    
    # Time series specific profiling
    ts_profile = profiler.profile_time_series(df, 'date', 'sales')
    
    frequency_info = ts_profile['temporal_analysis']['frequency']
    print(f"Detected frequency: {frequency_info['inferred_freq']} (confidence: {frequency_info['frequency_confidence']:.2f})")
    
    target_patterns = ts_profile['temporal_analysis']['target_patterns']
    if target_patterns['seasonality']['detected']:
        print(f"Seasonality detected: {target_patterns['seasonality']['type']}")
        print(f"Dominant period: {target_patterns['seasonality']['dominant_period']:.1f}")
    
    if target_patterns['trend']['detected']:
        print(f"Trend: {target_patterns['trend']['direction']} (R²: {target_patterns['trend']['r_squared']:.3f})")
    
    # Show recommendations
    print("\n💡 Recommendations:")
    for rec in ts_profile['recommendations']:
        print(f"  • {rec}")
    """
    
    print("\n✅ Data profiling would detect:")
    print("  • Daily frequency with high confidence")
    print("  • Yearly and weekly seasonality patterns")
    print("  • Increasing trend with strong significance")
    print("  • Recommendations for seasonal ARIMA and Prophet models")

# Example 2: Model Training
def example_model_training():
    """Example of time series model training"""
    
    print("\n🤖 Time Series Model Training Example")
    print("=" * 50)
    
    # This shows what the training would look like:
    """
    from src.ml.timeseries_trainer import TimeSeriesTrainer
    
    # Initialize trainer
    trainer = TimeSeriesTrainer(
        run_id="sales_forecast_001", 
        session_id="user_session_123", 
        websocket_manager=None
    )
    
    # Train models with automatic selection
    best_model = await trainer.train_forecast_models(
        dataset_path='sales_data.csv',
        date_column='date',
        target_column='sales',
        forecast_horizon=30,  # 30 days ahead
        data_profile=profile,
        constraints={
            'time_budget': 'medium',
            'selected_models': ['arima', 'prophet', 'exponential_smoothing']
        }
    )
    
    print(f"Best model: {best_model.family}")
    print(f"Validation RMSE: {best_model.val_score:.2f}")
    print(f"Training time: {best_model.training_time:.1f} minutes")
    """
    
    print("✅ Training would:")
    print("  • Test ARIMA, Prophet, and Exponential Smoothing models")
    print("  • Automatically tune hyperparameters for each model")
    print("  • Select best model based on time series metrics (RMSE, MAPE, etc.)")
    print("  • Generate 30-day forecasts with confidence intervals")

# Example 3: Individual Model Usage
def example_individual_models():
    """Example of using individual forecasting models"""
    
    print("\n📈 Individual Model Examples")
    print("=" * 50)
    
    df = create_sample_data()
    
    # Split data for demo
    train_size = int(len(df) * 0.8)
    train_data = df[:train_size]
    test_data = df[train_size:]
    
    print(f"Training data: {len(train_data)} days")
    print(f"Test data: {len(test_data)} days")
    
    # ARIMA Example
    print("\n🔹 ARIMA Model Example:")
    """
    from src.ml.timeseries_trainer import ARIMAForecaster
    
    arima_model = ARIMAForecaster()
    fit_info = arima_model.auto_fit(train_data.set_index('date')['sales'], seasonal=True)
    
    print(f"  ARIMA order: {fit_info['order']}")
    print(f"  AIC: {fit_info['aic']:.2f}")
    
    # Generate forecast
    forecast_result = arima_model.forecast(steps=len(test_data))
    forecast_values = forecast_result["forecast"]
    confidence_intervals = forecast_result["confidence_intervals"]
    """
    print("  • Automatically selects optimal (p,d,q) parameters")
    print("  • Handles seasonal patterns with SARIMA")
    print("  • Provides confidence intervals for uncertainty quantification")
    
    # Prophet Example  
    print("\n🔹 Prophet Model Example:")
    """
    from src.ml.timeseries_trainer import ProphetForecaster
    
    prophet_model = ProphetForecaster()
    fit_info = prophet_model.fit_with_seasonality(
        train_data, 'date', 'sales',
        yearly=True, weekly=True, daily=False
    )
    
    forecast_df = prophet_model.forecast_with_uncertainty(periods=len(test_data), freq='D')
    """
    print("  • Automatically detects yearly and weekly seasonality")
    print("  • Handles holidays and special events")
    print("  • Robust to missing data and outliers")
    
    # LSTM Example
    print("\n🔹 LSTM Model Example:")
    """
    from src.ml.timeseries_trainer import LSTMTimeSeriesModel
    
    lstm_model = LSTMTimeSeriesModel(
        input_size=1, hidden_size=64, num_layers=2, output_size=1
    )
    
    # Prepare sequential data
    X_train, y_train = lstm_model.prepare_data(train_data.set_index('date')['sales'])
    
    # Train neural network
    training_info = lstm_model.train_model(X_train, y_train, epochs=50)
    
    # Generate forecasts
    last_sequence = train_data['sales'].tail(lstm_model.sequence_length).values
    forecast_values = lstm_model.forecast(last_sequence, steps=len(test_data))
    """
    print("  • Learns complex non-linear temporal patterns")
    print("  • Captures long-term dependencies through memory")
    print("  • Automatically scales data for optimal training")

# Example 4: Metrics and Evaluation
def example_metrics():
    """Example of time series evaluation metrics"""
    
    print("\n📊 Time Series Metrics Example")
    print("=" * 50)
    
    # Simulate actual vs predicted values
    actual = np.array([1200, 1250, 1180, 1300, 1220, 1380, 1350, 1290])
    forecast = np.array([1180, 1240, 1200, 1280, 1250, 1360, 1320, 1310])
    
    print("Sample actual vs forecast values:")
    print("Actual:   ", actual)
    print("Forecast: ", forecast)
    
    """
    from src.ml.timeseries_trainer import TimeSeriesMetrics
    
    metrics = TimeSeriesMetrics.comprehensive_evaluation(
        actual=actual, 
        forecast=forecast, 
        seasonal_period=7  # Weekly seasonality
    )
    
    print(f"\n📈 Forecast Quality Metrics:")
    print(f"  MAE:  {metrics['mae']:.2f}")
    print(f"  RMSE: {metrics['rmse']:.2f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    print(f"  SMAPE: {metrics['smape']:.2f}%") 
    print(f"  MASE: {metrics['mase']:.2f}")
    print(f"  Directional Accuracy: {metrics['directional_accuracy']:.2f}%")
    """
    
    # Calculate simple metrics for demo
    mae = np.mean(np.abs(actual - forecast))
    rmse = np.sqrt(np.mean((actual - forecast) ** 2))
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100
    
    print(f"\n📈 Forecast Quality Metrics:")
    print(f"  MAE:  {mae:.2f} (Mean Absolute Error)")
    print(f"  RMSE: {rmse:.2f} (Root Mean Square Error)")
    print(f"  MAPE: {mape:.2f}% (Mean Absolute Percentage Error)")
    print("\n✅ Lower values indicate better forecast accuracy")

def main():
    """Run all examples"""
    
    print("🚀 Cipher Platform - Time Series Forecasting Examples")
    print("=" * 60)
    print("This demonstrates the new time series capabilities integrated")
    print("into your Cipher ML platform.")
    print()
    
    example_data_profiling()
    example_model_training()
    example_individual_models()
    example_metrics()
    
    print("\n" + "=" * 60)
    print("🎉 Time Series Integration Complete!")
    print("\n📖 For full usage guide, see: TIME_SERIES_INTEGRATION_GUIDE.md")
    print("🔧 To install dependencies: pip install -r requirements.txt")
    print("🧪 To test integration: python test_timeseries_integration.py")

if __name__ == "__main__":
    main() 