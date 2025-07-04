"""
Standalone script to generate sample time series data for testing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def create_sample_time_series_data(output_file="sample_timeseries_data.csv"):
    """Create sample time series data for testing"""
    
    print("🚀 Generating sample time series data...")
    
    # Create date range - 4 years of daily data
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create synthetic time series with trend and seasonality
    n_points = len(dates)
    print(f"📅 Generating {n_points} data points from {start_date.date()} to {end_date.date()}")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Base trend (increasing sales over time)
    trend = np.linspace(100, 300, n_points)
    
    # Seasonal component (yearly cycle - higher sales in winter)
    yearly_seasonal = 50 * np.sin(2 * np.pi * np.arange(n_points) / 365.25 + np.pi/2)
    
    # Weekly seasonality (lower sales on weekends)
    weekly_seasonal = 20 * np.sin(2 * np.pi * np.arange(n_points) / 7 + np.pi)
    
    # Monthly pattern (end of month spike)
    monthly_pattern = 15 * np.sin(2 * np.pi * np.arange(n_points) / 30.44)
    
    # Random noise
    noise = np.random.normal(0, 10, n_points)
    
    # Special events (random spikes for holidays/promotions)
    special_events = np.zeros(n_points)
    event_days = np.random.choice(n_points, size=20, replace=False)
    special_events[event_days] = np.random.normal(50, 15, 20)
    
    # Combine all components to create realistic sales data
    sales = trend + yearly_seasonal + weekly_seasonal + monthly_pattern + noise + special_events
    
    # Ensure no negative sales
    sales = np.maximum(sales, 10)
    
    # Create additional features
    # Temperature (affects sales - inverse relationship in this example)
    temperature = 20 + 15 * np.sin(2 * np.pi * np.arange(n_points) / 365.25) + np.random.normal(0, 3, n_points)
    
    # Marketing spend (correlated with some sales spikes)
    marketing_spend = 1000 + 500 * np.sin(2 * np.pi * np.arange(n_points) / 90) + np.random.normal(0, 200, n_points)
    marketing_spend = np.maximum(marketing_spend, 0)
    
    # Product category (categorical feature)
    categories = ['Electronics', 'Clothing', 'Home', 'Sports', 'Books']
    category_weights = [0.3, 0.25, 0.2, 0.15, 0.1]
    product_category = np.random.choice(categories, size=n_points, p=category_weights)
    
    # Store location
    locations = ['North', 'South', 'East', 'West', 'Central']
    store_location = np.random.choice(locations, size=n_points)
    
    # Day of week names
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_of_week = [day_names[date.weekday()] for date in dates]
    
    # Is weekend flag
    is_weekend = [1 if date.weekday() >= 5 else 0 for date in dates]
    
    # Month names
    month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    month_name = [month_names[date.month - 1] for date in dates]
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'sales': np.round(sales, 2),
        'temperature': np.round(temperature, 1),
        'marketing_spend': np.round(marketing_spend, 2),
        'product_category': product_category,
        'store_location': store_location,
        'day_of_week': day_of_week,
        'is_weekend': is_weekend,
        'month': month_name,
        'year': [date.year for date in dates],
        'quarter': [f"Q{((date.month-1)//3)+1}" for date in dates]
    })
    
    # Add some missing values to make it realistic (about 1% missing)
    missing_indices = np.random.choice(n_points, size=int(n_points * 0.01), replace=False)
    df.loc[missing_indices, 'sales'] = np.nan
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    print(f"✅ Sample data generated successfully!")
    print(f"📁 Saved to: {output_file}")
    print(f"📊 Data shape: {df.shape}")
    print(f"📈 Sales range: ${df['sales'].min():.2f} - ${df['sales'].max():.2f}")
    print(f"📅 Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"🔍 Missing values: {df['sales'].isnull().sum()} ({df['sales'].isnull().sum()/len(df)*100:.1f}%)")
    
    # Display sample of the data
    print("\n📋 Sample of generated data:")
    print(df.head(10).to_string(index=False))
    
    # Display summary statistics
    print("\n📊 Summary statistics for sales:")
    print(df['sales'].describe())
    
    return df

def create_multiple_datasets():
    """Create multiple sample datasets for different scenarios"""
    
    print("🔄 Creating multiple sample datasets...")
    
    # Dataset 1: Daily sales data (main dataset)
    df1 = create_sample_time_series_data("daily_sales_data.csv")
    
    # Dataset 2: Weekly aggregated data
    print("\n📅 Creating weekly aggregated dataset...")
    df_weekly = df1.groupby(pd.Grouper(key='date', freq='W')).agg({
        'sales': 'sum',
        'temperature': 'mean',
        'marketing_spend': 'sum',
        'is_weekend': 'sum'  # Count of weekend days in week
    }).reset_index()
    df_weekly.to_csv("weekly_sales_data.csv", index=False)
    print(f"✅ Weekly data saved: {df_weekly.shape}")
    
    # Dataset 3: Monthly aggregated data
    print("\n📅 Creating monthly aggregated dataset...")
    df_monthly = df1.groupby(pd.Grouper(key='date', freq='M')).agg({
        'sales': 'sum',
        'temperature': 'mean',
        'marketing_spend': 'sum'
    }).reset_index()
    df_monthly.to_csv("monthly_sales_data.csv", index=False)
    print(f"✅ Monthly data saved: {df_monthly.shape}")
    
    # Dataset 4: Hourly data (smaller sample for testing)
    print("\n⏰ Creating hourly dataset (30 days)...")
    hourly_dates = pd.date_range(start='2023-11-01', end='2023-11-30 23:00:00', freq='H')
    n_hours = len(hourly_dates)
    
    # Hourly patterns
    hourly_base = 50
    daily_pattern = 30 * np.sin(2 * np.pi * np.arange(n_hours) / 24 - np.pi/2)  # Peak at afternoon
    weekly_pattern = 20 * np.sin(2 * np.pi * np.arange(n_hours) / (24*7))
    noise = np.random.normal(0, 5, n_hours)
    
    hourly_sales = hourly_base + daily_pattern + weekly_pattern + noise
    hourly_sales = np.maximum(hourly_sales, 1)
    
    df_hourly = pd.DataFrame({
        'datetime': hourly_dates,
        'sales': np.round(hourly_sales, 2),
        'hour': [dt.hour for dt in hourly_dates],
        'day_of_week': [dt.strftime('%A') for dt in hourly_dates],
        'is_business_hours': [1 if 9 <= dt.hour <= 17 else 0 for dt in hourly_dates]
    })
    
    df_hourly.to_csv("hourly_sales_data.csv", index=False)
    print(f"✅ Hourly data saved: {df_hourly.shape}")
    
    print("\n🎉 All sample datasets created successfully!")
    print("📁 Generated files:")
    for filename in ["daily_sales_data.csv", "weekly_sales_data.csv", "monthly_sales_data.csv", "hourly_sales_data.csv"]:
        if os.path.exists(filename):
            size = os.path.getsize(filename) / 1024  # Size in KB
            print(f"   • {filename} ({size:.1f} KB)")

def main():
    """Main function to generate sample data"""
    print("🔢 Time Series Sample Data Generator")
    print("=" * 50)
    
    # Ask user what type of data to generate
    print("\nChoose data generation option:")
    print("1. Single daily dataset (recommended)")
    print("2. Multiple datasets (daily, weekly, monthly, hourly)")
    
    try:
        choice = input("\nEnter choice (1 or 2, default=1): ").strip()
        if choice == "2":
            create_multiple_datasets()
        else:
            create_sample_time_series_data()
            
        print("\n✨ Data generation complete!")
        print("💡 You can now use these files to test the time series forecasting pipeline.")
        
    except KeyboardInterrupt:
        print("\n\n❌ Data generation cancelled by user.")
    except Exception as e:
        print(f"\n❌ Error generating data: {str(e)}")

if __name__ == "__main__":
    main() 