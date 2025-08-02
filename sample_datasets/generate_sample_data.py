#!/usr/bin/env python3
"""
Generate sample datasets for demonstrating the Data Analysis Platform.

This script creates realistic sample datasets covering different domains:
- Sales performance data
- Student academic performance
- Housing market data
- E-commerce customer data
- Financial stock data

All datasets are designed to be educationally valuable and demonstrate
various data analysis concepts.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path

# Set random seed for reproducible data
np.random.seed(42)

def create_output_directory():
    """Create output directory for sample datasets."""
    output_dir = Path("sample_datasets")
    output_dir.mkdir(exist_ok=True)
    return output_dir

def generate_sales_performance_data(n_records=500):
    """Generate realistic sales performance dataset."""
    
    # Sales regions and products
    regions = ['North', 'South', 'East', 'West', 'Central']
    products = ['Product_A', 'Product_B', 'Product_C', 'Product_D', 'Product_E']
    sales_reps = [f'Rep_{i:03d}' for i in range(1, 51)]
    
    # Generate data
    data = []
    start_date = datetime(2023, 1, 1)
    
    for i in range(n_records):
        # Random date in 2023
        random_days = np.random.randint(0, 365)
        sale_date = start_date + timedelta(days=random_days)
        
        region = np.random.choice(regions)
        product = np.random.choice(products)
        sales_rep = np.random.choice(sales_reps)
        
        # Base price varies by product
        base_prices = {'Product_A': 100, 'Product_B': 150, 'Product_C': 200, 'Product_D': 75, 'Product_E': 300}
        base_price = base_prices[product]
        
        # Regional multiplier
        regional_multipliers = {'North': 1.1, 'South': 0.9, 'East': 1.05, 'West': 1.15, 'Central': 1.0}
        regional_mult = regional_multipliers[region]
        
        # Generate correlated data
        quantity = np.random.poisson(5) + 1  # 1-15 typical range
        unit_price = base_price * regional_mult * np.random.uniform(0.8, 1.2)
        revenue = quantity * unit_price
        
        # Customer satisfaction (correlated with price and region)
        satisfaction_base = 7 + (unit_price - 100) / 50 + np.random.choice([0.5, -0.5]) if region == 'West' else 0
        customer_satisfaction = max(1, min(10, satisfaction_base + np.random.normal(0, 1)))
        
        # Discount (affects satisfaction and revenue)
        discount_pct = np.random.uniform(0, 0.25) if np.random.random() < 0.3 else 0
        
        data.append({
            'sale_id': f'SALE_{i+1:05d}',
            'date': sale_date.strftime('%Y-%m-%d'),
            'region': region,
            'sales_rep': sales_rep,
            'product': product,
            'quantity': quantity,
            'unit_price': round(unit_price, 2),
            'discount_pct': round(discount_pct, 3),
            'revenue': round(revenue * (1 - discount_pct), 2),
            'customer_satisfaction': round(customer_satisfaction, 1),
            'quarter': f'Q{(sale_date.month - 1) // 3 + 1}',
            'month': sale_date.strftime('%B')
        })
    
    return pd.DataFrame(data)

def generate_student_performance_data(n_students=300):
    """Generate student academic performance dataset."""
    
    subjects = ['Math', 'Science', 'English', 'History', 'Art']
    schools = ['Lincoln High', 'Washington Academy', 'Roosevelt School', 'Jefferson Prep']
    grades = ['9th', '10th', '11th', '12th']
    
    data = []
    
    for i in range(n_students):
        student_id = f'STU_{i+1:04d}'
        school = np.random.choice(schools)
        grade = np.random.choice(grades)
        
        # Base ability (affects all subjects)
        base_ability = np.random.normal(75, 15)
        
        # Study hours per week
        study_hours = max(0, np.random.normal(10, 5))
        
        # Attendance rate
        attendance_rate = min(100, max(60, np.random.normal(85, 10)))
        
        # Generate scores for each subject
        subject_scores = {}
        for subject in subjects:
            # Subject-specific factors
            subject_difficulty = {'Math': 1.2, 'Science': 1.1, 'English': 0.9, 'History': 1.0, 'Art': 0.8}
            difficulty = subject_difficulty[subject]
            
            # Individual aptitude for subject
            aptitude = np.random.normal(0, 10)
            
            # Calculate score with correlations
            score = base_ability + aptitude + (study_hours * 1.5) + (attendance_rate * 0.2) - (difficulty * 5)
            score = max(0, min(100, score + np.random.normal(0, 5)))
            
            subject_scores[subject] = round(score, 1)
        
        # Calculate GPA
        gpa = sum(subject_scores.values()) / len(subject_scores) / 25  # Convert to 4.0 scale
        
        # Extracurricular activities
        extracurricular = np.random.choice(['Sports', 'Music', 'Drama', 'Debate', 'None'], 
                                         p=[0.3, 0.2, 0.15, 0.1, 0.25])
        
        data.append({
            'student_id': student_id,
            'school': school,
            'grade': grade,
            'study_hours_per_week': round(study_hours, 1),
            'attendance_rate': round(attendance_rate, 1),
            'math_score': subject_scores['Math'],
            'science_score': subject_scores['Science'],
            'english_score': subject_scores['English'],
            'history_score': subject_scores['History'],
            'art_score': subject_scores['Art'],
            'gpa': round(gpa, 2),
            'extracurricular': extracurricular,
            'parent_education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 
                                               p=[0.3, 0.4, 0.2, 0.1])
        })
    
    return pd.DataFrame(data)

def generate_housing_market_data(n_houses=400):
    """Generate realistic housing market dataset."""
    
    neighborhoods = ['Downtown', 'Suburban', 'Riverside', 'Hillside', 'Industrial', 'University']
    house_types = ['Single Family', 'Townhouse', 'Condo', 'Duplex']
    
    data = []
    
    for i in range(n_houses):
        house_id = f'HOUSE_{i+1:04d}'
        neighborhood = np.random.choice(neighborhoods)
        house_type = np.random.choice(house_types)
        
        # Base characteristics
        age = np.random.randint(1, 50)  # Years old
        bedrooms = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.15, 0.35, 0.35, 0.1])
        bathrooms = max(1, bedrooms - np.random.choice([0, 1], p=[0.7, 0.3]))
        
        # Square footage (correlated with bedrooms)
        base_sqft = 800 + (bedrooms * 400) + np.random.normal(0, 200)
        square_feet = max(500, int(base_sqft))
        
        # Lot size
        lot_size = np.random.uniform(0.1, 2.0) if house_type in ['Single Family', 'Duplex'] else 0
        
        # Neighborhood price multipliers
        neighborhood_multipliers = {
            'Downtown': 1.3, 'Suburban': 1.0, 'Riverside': 1.2, 
            'Hillside': 1.4, 'Industrial': 0.7, 'University': 0.9
        }
        
        # Calculate price
        base_price_per_sqft = 150
        neighborhood_mult = neighborhood_multipliers[neighborhood]
        age_discount = 1 - (age * 0.01)  # 1% per year depreciation
        
        price = (square_feet * base_price_per_sqft * 
                neighborhood_mult * age_discount * 
                np.random.uniform(0.8, 1.2))
        
        # Add premium for certain features
        has_garage = np.random.choice([True, False], p=[0.7, 0.3])
        has_pool = np.random.choice([True, False], p=[0.15, 0.85])
        has_fireplace = np.random.choice([True, False], p=[0.4, 0.6])
        
        if has_garage:
            price *= 1.05
        if has_pool:
            price *= 1.1
        if has_fireplace:
            price *= 1.03
        
        # Days on market (correlated with price competitiveness)
        market_competitiveness = price / (square_feet * base_price_per_sqft * neighborhood_mult)
        days_on_market = max(1, int(np.random.exponential(30) * (2 - market_competitiveness)))
        
        data.append({
            'house_id': house_id,
            'neighborhood': neighborhood,
            'house_type': house_type,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'square_feet': square_feet,
            'lot_size_acres': round(lot_size, 2),
            'age_years': age,
            'has_garage': has_garage,
            'has_pool': has_pool,
            'has_fireplace': has_fireplace,
            'price': int(price),
            'price_per_sqft': round(price / square_feet, 2),
            'days_on_market': days_on_market,
            'listing_date': (datetime.now() - timedelta(days=days_on_market)).strftime('%Y-%m-%d')
        })
    
    return pd.DataFrame(data)

def generate_ecommerce_customer_data(n_customers=350):
    """Generate e-commerce customer behavior dataset."""
    
    countries = ['USA', 'Canada', 'UK', 'Germany', 'France', 'Australia', 'Japan']
    acquisition_channels = ['Organic Search', 'Paid Search', 'Social Media', 'Email', 'Direct', 'Referral']
    customer_segments = ['Premium', 'Standard', 'Budget']
    
    data = []
    
    for i in range(n_customers):
        customer_id = f'CUST_{i+1:05d}'
        country = np.random.choice(countries)
        acquisition_channel = np.random.choice(acquisition_channels)
        
        # Customer tenure (days since first purchase)
        tenure_days = np.random.randint(1, 730)  # Up to 2 years
        
        # Age affects spending patterns
        age = np.random.randint(18, 70)
        
        # Income level (affects spending)
        income_level = np.random.choice(['Low', 'Medium', 'High'], p=[0.3, 0.5, 0.2])
        income_multiplier = {'Low': 0.7, 'Medium': 1.0, 'High': 1.5}[income_level]
        
        # Base spending influenced by multiple factors
        base_monthly_spend = 100 * income_multiplier * (1 + age/100)
        
        # Total orders (correlated with tenure and spending)
        orders_per_month = np.random.poisson(2) + 1
        total_orders = int((tenure_days / 30) * orders_per_month * np.random.uniform(0.7, 1.3))
        total_orders = max(1, total_orders)
        
        # Total spent
        avg_order_value = base_monthly_spend / orders_per_month * np.random.uniform(0.8, 1.2)
        total_spent = total_orders * avg_order_value
        
        # Customer satisfaction (affects return rate)
        satisfaction_score = np.random.normal(4, 0.8)
        satisfaction_score = max(1, min(5, satisfaction_score))
        
        # Return rate (inversely correlated with satisfaction)
        return_rate = max(0, min(0.5, 0.3 - (satisfaction_score - 3) * 0.1 + np.random.normal(0, 0.05)))
        
        # Email engagement
        email_open_rate = np.random.beta(4, 2) * 0.6  # Typically 20-40%
        email_click_rate = email_open_rate * np.random.uniform(0.1, 0.3)  # 10-30% of opens
        
        # Segment classification
        if total_spent > 1000 and total_orders > 10:
            segment = 'Premium'
        elif total_spent > 300:
            segment = 'Standard'
        else:
            segment = 'Budget'
        
        data.append({
            'customer_id': customer_id,
            'country': country,
            'age': age,
            'income_level': income_level,
            'acquisition_channel': acquisition_channel,
            'tenure_days': tenure_days,
            'total_orders': total_orders,
            'total_spent': round(total_spent, 2),
            'avg_order_value': round(total_spent / total_orders, 2),
            'return_rate': round(return_rate, 3),
            'satisfaction_score': round(satisfaction_score, 1),
            'email_open_rate': round(email_open_rate, 3),
            'email_click_rate': round(email_click_rate, 3),
            'customer_segment': segment,
            'last_purchase_days_ago': np.random.randint(1, min(60, tenure_days))
        })
    
    return pd.DataFrame(data)

def generate_financial_stock_data(n_days=252):  # ~1 trading year
    """Generate realistic stock price data."""
    
    # Multiple stocks for comparison
    stocks = ['TECH_A', 'TECH_B', 'HEALTH_C', 'FINANCE_D', 'ENERGY_E']
    
    data = []
    start_date = datetime(2023, 1, 1)
    
    for stock in stocks:
        # Stock-specific parameters
        stock_params = {
            'TECH_A': {'start_price': 150, 'volatility': 0.25, 'trend': 0.12},
            'TECH_B': {'start_price': 200, 'volatility': 0.30, 'trend': 0.08},
            'HEALTH_C': {'start_price': 75, 'volatility': 0.15, 'trend': 0.06},
            'FINANCE_D': {'start_price': 50, 'volatility': 0.20, 'trend': 0.04},
            'ENERGY_E': {'start_price': 80, 'volatility': 0.35, 'trend': -0.02}
        }
        
        params = stock_params[stock]
        current_price = params['start_price']
        
        for day in range(n_days):
            date = start_date + timedelta(days=day)
            
            # Skip weekends
            if date.weekday() >= 5:
                continue
            
            # Generate daily return with trend and volatility
            daily_return = np.random.normal(params['trend']/252, params['volatility']/np.sqrt(252))
            current_price *= (1 + daily_return)
            
            # Generate OHLCV data
            open_price = current_price * np.random.uniform(0.995, 1.005)
            high_price = max(open_price, current_price) * np.random.uniform(1.0, 1.03)
            low_price = min(open_price, current_price) * np.random.uniform(0.97, 1.0)
            close_price = current_price
            
            # Volume (higher on big price moves)
            price_change = abs(daily_return)
            base_volume = 1000000
            volume = int(base_volume * (1 + price_change * 5) * np.random.uniform(0.5, 2.0))
            
            # Technical indicators (simplified)
            # Moving averages would be calculated in analysis
            
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'stock_symbol': stock,
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': volume,
                'daily_return': round(daily_return, 4),
                'sector': stock.split('_')[0].title()
            })
            
            current_price = close_price
    
    return pd.DataFrame(data)

def create_dataset_metadata():
    """Create metadata for all sample datasets."""
    metadata = {
        'sales_performance.csv': {
            'title': 'Sales Performance Dataset',
            'description': 'Regional sales data with performance metrics and customer satisfaction',
            'size_records': 500,
            'use_cases': ['Sales analytics', 'Regional performance comparison', 'Revenue forecasting'],
            'target_variables': ['revenue', 'customer_satisfaction'],
            'key_features': ['region', 'product', 'quantity', 'unit_price', 'discount_pct'],
            'analysis_suggestions': [
                'Compare regional performance',
                'Analyze product profitability',
                'Predict revenue based on quantity and pricing',
                'Examine seasonal trends by quarter'
            ]
        },
        'student_performance.csv': {
            'title': 'Student Academic Performance',
            'description': 'Student grades and factors affecting academic performance',
            'size_records': 300,
            'use_cases': ['Educational analytics', 'Performance prediction', 'Factor analysis'],
            'target_variables': ['gpa', 'math_score', 'science_score'],
            'key_features': ['study_hours_per_week', 'attendance_rate', 'extracurricular', 'parent_education'],
            'analysis_suggestions': [
                'Predict GPA from study habits and attendance',
                'Analyze impact of extracurricular activities',
                'Compare performance across schools',
                'Examine correlation between subjects'
            ]
        },
        'housing_market.csv': {
            'title': 'Housing Market Analysis',
            'description': 'Real estate data with property features and pricing',
            'size_records': 400,
            'use_cases': ['Price prediction', 'Market analysis', 'Investment decisions'],
            'target_variables': ['price', 'price_per_sqft', 'days_on_market'],
            'key_features': ['neighborhood', 'square_feet', 'bedrooms', 'bathrooms', 'age_years'],
            'analysis_suggestions': [
                'Predict house prices from features',
                'Analyze neighborhood price differences',
                'Examine factors affecting time on market',
                'Calculate price per square foot trends'
            ]
        },
        'ecommerce_customers.csv': {
            'title': 'E-commerce Customer Behavior',
            'description': 'Customer data with purchasing patterns and engagement metrics',
            'size_records': 350,
            'use_cases': ['Customer segmentation', 'CLV prediction', 'Marketing analytics'],
            'target_variables': ['total_spent', 'customer_segment', 'satisfaction_score'],
            'key_features': ['age', 'tenure_days', 'total_orders', 'acquisition_channel'],
            'analysis_suggestions': [
                'Segment customers by behavior patterns',
                'Predict customer lifetime value',
                'Analyze acquisition channel effectiveness',
                'Examine factors affecting satisfaction'
            ]
        },
        'financial_stocks.csv': {
            'title': 'Stock Market Financial Data',
            'description': 'Daily stock prices with technical indicators across sectors',
            'size_records': 1260,  # ~252 days * 5 stocks
            'use_cases': ['Financial analysis', 'Price prediction', 'Risk assessment'],
            'target_variables': ['close', 'daily_return'],
            'key_features': ['open', 'high', 'low', 'volume', 'sector'],
            'analysis_suggestions': [
                'Calculate moving averages and technical indicators',
                'Analyze sector performance comparison',
                'Examine volatility patterns',
                'Predict price movements'
            ]
        }
    }
    
    return metadata

def main():
    """Generate all sample datasets."""
    print("🏭 Generating Sample Datasets for Data Analysis Platform...")
    
    # Create output directory
    output_dir = create_output_directory()
    
    # Generate datasets
    datasets = {
        'sales_performance.csv': generate_sales_performance_data(500),
        'student_performance.csv': generate_student_performance_data(300),
        'housing_market.csv': generate_housing_market_data(400),
        'ecommerce_customers.csv': generate_ecommerce_customer_data(350),
        'financial_stocks.csv': generate_financial_stock_data(252)
    }
    
    # Save datasets
    for filename, df in datasets.items():
        filepath = output_dir / filename
        df.to_csv(filepath, index=False)
        print(f"✅ Generated {filename}: {len(df)} records")
        
        # Display basic info
        print(f"   Columns: {list(df.columns)}")
        print(f"   Data types: {dict(df.dtypes)}")
        print()
    
    # Save metadata
    metadata = create_dataset_metadata()
    metadata_path = output_dir / 'datasets_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"📄 Saved metadata to {metadata_path}")
    print(f"📊 Total datasets created: {len(datasets)}")
    print(f"💾 Output directory: {output_dir.absolute()}")
    
    print("\n🎯 Next Steps:")
    print("1. Upload these datasets using the file upload interface")
    print("2. Use the EDA template to explore the data")
    print("3. Apply linear regression models to predict target variables")
    print("4. Try different analysis approaches on each dataset")

if __name__ == "__main__":
    main()