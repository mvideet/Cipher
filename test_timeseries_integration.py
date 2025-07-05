"""
Simple test to demonstrate time series integration
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def create_sample_time_series_data():
    """Create sample time series data for testing"""
    
    # Create date range
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create synthetic time series with trend and seasonality
    n_points = len(dates)
    
    # Base trend
    trend = np.linspace(100, 200, n_points)
    
    # Seasonal component (yearly)
    seasonal = 20 * np.sin(2 * np.pi * np.arange(n_points) / 365.25)
    
    # Weekly seasonality
    weekly = 10 * np.sin(2 * np.pi * np.arange(n_points) / 7)
    
    # Random noise
    noise = np.random.normal(0, 5, n_points)
    
    # Combine components
    values = trend + seasonal + weekly + noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'sales': values,
        'feature1': np.random.normal(50, 10, n_points),
        'feature2': np.random.choice(['A', 'B', 'C'], n_points)
    })
    
    return df

def test_time_series_profiling():
    """Test the time series profiling functionality"""
    
    print("🧪 Testing Time Series Data Profiling...")
    
    # Create sample data
    df = create_sample_time_series_data()
    print(f"Created sample data: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Test basic data profiling
    try:
        from src.ml.data_profiler import DataProfiler
        
        profiler = DataProfiler()
        
        # Standard profiling
        profile = profiler.profile_dataset(df)
        print(f"✅ Basic profiling completed - {profile.n_rows} rows, {profile.n_cols} columns")
        print(f"Issues found: {len(profile.issues)}")
        print(f"Recommendations: {len(profile.recommendations)}")
        
        # Time series profiling
        ts_profile = profiler.profile_time_series(df, 'date', 'sales')
        print(f"✅ Time series profiling completed")
        print(f"Frequency detected: {ts_profile['temporal_analysis']['frequency']['inferred_freq']}")
        print(f"Seasonality detected: {ts_profile['temporal_analysis']['target_patterns']['seasonality']['detected']}")
        print(f"Trend detected: {ts_profile['temporal_analysis']['target_patterns']['trend']['detected']}")
        print(f"TS Recommendations: {len(ts_profile['recommendations'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Time series profiling failed: {str(e)}")
        return False

def test_time_series_data_pipeline():
    """Test the time series data pipeline"""
    
    print("\n🧪 Testing Time Series Data Pipeline...")
    
    try:
        from src.ml.timeseries_trainer import TimeSeriesDataPipeline
        
        # Create sample data
        df = create_sample_time_series_data()
        
        # Initialize pipeline
        pipeline = TimeSeriesDataPipeline('date', 'sales')
        
        # Prepare data
        prepared_df, characteristics = pipeline.prepare_time_series_data(df)
        
        print(f"✅ Data pipeline completed")
        print(f"Original shape: {df.shape}")
        print(f"Prepared shape: {prepared_df.shape}")
        print(f"Frequency detected: {characteristics['frequency']}")
        print(f"Data length: {characteristics['length']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data pipeline test failed: {str(e)}")
        return False

def test_time_series_metrics():
    """Test time series evaluation metrics"""
    
    print("\n🧪 Testing Time Series Metrics...")
    
    try:
        from src.ml.timeseries_trainer import TimeSeriesMetrics
        
        # Create sample actual and forecast data
        actual = np.array([100, 110, 105, 115, 120, 125, 130, 135])
        forecast = np.array([98, 112, 107, 113, 118, 127, 132, 133])
        
        # Test metrics
        metrics = TimeSeriesMetrics.comprehensive_evaluation(actual, forecast, seasonal_period=4)
        
        print(f"✅ Metrics calculation completed")
        print(f"MAE: {metrics['mae']:.2f}")
        print(f"RMSE: {metrics['rmse']:.2f}")
        print(f"MAPE: {metrics['mape']:.2f}%")
        print(f"SMAPE: {metrics['smape']:.2f}%")
        print(f"Directional Accuracy: {metrics['directional_accuracy']:.2f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Metrics test failed: {str(e)}")
        return False

def main():
    """Run all time series integration tests"""
    
    print("🚀 Time Series Integration Test Suite")
    print("=" * 50)
    
    tests = [
        test_time_series_profiling,
        test_time_series_data_pipeline,
        test_time_series_metrics
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {str(e)}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"✅ Passed: {sum(results)}/{len(results)}")
    print(f"❌ Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("\n🎉 All time series integration tests passed!")
        print("The time series forecasting pipeline is ready for use.")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    main() 