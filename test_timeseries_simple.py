"""
Simple test script for time series functionality without complex dependencies
"""

import pandas as pd
import numpy as np
import sys
import os

def test_basic_data_loading():
    """Test basic data loading and inspection"""
    
    print("🔍 Testing Basic Data Loading...")
    
    try:
        # Load the generated data
        df = pd.read_csv("daily_sales_data.csv")
        
        print(f"✅ Data loaded successfully!")
        print(f"📊 Shape: {df.shape}")
        print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"📈 Sales range: ${df['sales'].min():.2f} - ${df['sales'].max():.2f}")
        
        # Check for missing values
        missing_count = df['sales'].isnull().sum()
        print(f"🔍 Missing values: {missing_count} ({missing_count/len(df)*100:.1f}%)")
        
        # Basic statistics
        print(f"\n📊 Sales Statistics:")
        print(f"   Mean: ${df['sales'].mean():.2f}")
        print(f"   Std:  ${df['sales'].std():.2f}")
        print(f"   Min:  ${df['sales'].min():.2f}")
        print(f"   Max:  ${df['sales'].max():.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading failed: {str(e)}")
        return False

def test_simple_forecasting():
    """Test simple forecasting methods"""
    
    print("\n🎯 Testing Simple Forecasting...")
    
    try:
        # Load data
        df = pd.read_csv("daily_sales_data.csv")
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # Remove missing values for simplicity
        df = df.dropna(subset=['sales'])
        
        # Split data: 80% train, 20% test
        split_idx = int(len(df) * 0.8)
        train_data = df.iloc[:split_idx]
        test_data = df.iloc[split_idx:]
        
        print(f"📊 Train size: {len(train_data)}")
        print(f"📊 Test size: {len(test_data)}")
        
        # Method 1: Simple Moving Average
        window = 7  # 7-day moving average
        train_sales = train_data['sales'].values
        
        # Calculate moving average
        if len(train_sales) >= window:
            last_values = train_sales[-window:]
            ma_forecast = np.mean(last_values)
            
            # Create forecast for test period
            ma_predictions = np.full(len(test_data), ma_forecast)
            
            # Calculate error
            actual = test_data['sales'].values
            mae = np.mean(np.abs(actual - ma_predictions))
            rmse = np.sqrt(np.mean((actual - ma_predictions) ** 2))
            
            print(f"\n🔮 Moving Average Forecast (window={window}):")
            print(f"   Forecast value: ${ma_forecast:.2f}")
            print(f"   MAE: ${mae:.2f}")
            print(f"   RMSE: ${rmse:.2f}")
        
        # Method 2: Linear Trend
        days = np.arange(len(train_data))
        sales = train_data['sales'].values
        
        # Fit linear trend using numpy
        A = np.vstack([days, np.ones(len(days))]).T
        slope, intercept = np.linalg.lstsq(A, sales, rcond=None)[0]
        
        # Forecast using trend
        test_days = np.arange(len(train_data), len(train_data) + len(test_data))
        trend_predictions = slope * test_days + intercept
        
        # Calculate error
        mae_trend = np.mean(np.abs(actual - trend_predictions))
        rmse_trend = np.sqrt(np.mean((actual - trend_predictions) ** 2))
        
        print(f"\n📈 Linear Trend Forecast:")
        print(f"   Slope: ${slope:.2f}/day")
        print(f"   Intercept: ${intercept:.2f}")
        print(f"   MAE: ${mae_trend:.2f}")
        print(f"   RMSE: ${rmse_trend:.2f}")
        
        # Method 3: Seasonal Naive (use same day from previous week)
        if len(train_data) >= 7:
            seasonal_predictions = []
            for i in range(len(test_data)):
                # Use value from 7 days ago (same day of week)
                lookup_idx = len(train_data) - 7 + (i % 7)
                if lookup_idx >= 0 and lookup_idx < len(train_data):
                    seasonal_pred = train_data.iloc[lookup_idx]['sales']
                else:
                    seasonal_pred = train_data['sales'].mean()
                seasonal_predictions.append(seasonal_pred)
            
            seasonal_predictions = np.array(seasonal_predictions)
            mae_seasonal = np.mean(np.abs(actual - seasonal_predictions))
            rmse_seasonal = np.sqrt(np.mean((actual - seasonal_predictions) ** 2))
            
            print(f"\n🔄 Seasonal Naive Forecast:")
            print(f"   MAE: ${mae_seasonal:.2f}")
            print(f"   RMSE: ${rmse_seasonal:.2f}")
        
        print(f"\n🏆 Best Method Comparison:")
        methods = [
            ("Moving Average", mae, rmse),
            ("Linear Trend", mae_trend, rmse_trend),
            ("Seasonal Naive", mae_seasonal, rmse_seasonal)
        ]
        
        # Sort by MAE (lower is better)
        methods.sort(key=lambda x: x[1])
        
        for i, (name, mae_val, rmse_val) in enumerate(methods):
            status = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
            print(f"   {status} {name}: MAE=${mae_val:.2f}, RMSE=${rmse_val:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Forecasting test failed: {str(e)}")
        return False

def test_data_patterns():
    """Test pattern detection in the data"""
    
    print("\n🔍 Testing Pattern Detection...")
    
    try:
        # Load data
        df = pd.read_csv("daily_sales_data.csv")
        df['date'] = pd.to_datetime(df['date'])
        df = df.dropna(subset=['sales']).sort_values('date').reset_index(drop=True)
        
        # Add derived features
        df['day_of_week_num'] = df['date'].dt.dayofweek
        df['month_num'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        
        # Weekly pattern analysis
        weekly_avg = df.groupby('day_of_week')['sales'].mean()
        print(f"\n📅 Weekly Sales Pattern:")
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for i, day in enumerate(days):
            print(f"   {day}: ${weekly_avg.iloc[i]:.2f}")
        
        # Monthly pattern analysis
        monthly_avg = df.groupby('month')['sales'].mean()
        print(f"\n📅 Monthly Sales Pattern:")
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for i, month in enumerate(months):
            if i+1 in monthly_avg.index:
                print(f"   {month}: ${monthly_avg[i+1]:.2f}")
        
        # Yearly trend
        yearly_avg = df.groupby('year')['sales'].mean()
        print(f"\n📈 Yearly Sales Trend:")
        for year in yearly_avg.index:
            print(f"   {year}: ${yearly_avg[year]:.2f}")
        
        # Weekend vs Weekday
        weekend_sales = df[df['is_weekend'] == 1]['sales'].mean()
        weekday_sales = df[df['is_weekend'] == 0]['sales'].mean()
        print(f"\n🏢 Weekend vs Weekday:")
        print(f"   Weekday Average: ${weekday_sales:.2f}")
        print(f"   Weekend Average: ${weekend_sales:.2f}")
        print(f"   Weekend Premium: ${weekend_sales - weekday_sales:.2f} ({((weekend_sales/weekday_sales-1)*100):+.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ Pattern detection failed: {str(e)}")
        return False

def main():
    """Run all simple tests"""
    
    print("🚀 Simple Time Series Testing Suite")
    print("=" * 50)
    
    # Check if data files exist
    required_files = ["daily_sales_data.csv"]
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Required data files not found:")
        for file in missing_files:
            print(f"   • {file}")
        print("\n💡 Run 'python3 generate_sample_timeseries_data.py' first to create sample data.")
        return
    
    # Run tests
    tests = [
        ("Data Loading", test_basic_data_loading),
        ("Pattern Detection", test_data_patterns),
        ("Simple Forecasting", test_simple_forecasting)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 Test Results Summary:")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your time series setup is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    main() 