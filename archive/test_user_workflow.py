#!/usr/bin/env python3
"""
User Testing Workflow Script
Tests the complete data analysis workflow as a user would experience it.
"""

import requests
import json
import time
import sys
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000"
TEST_USER = {
    "username": "testuser123",
    "email": "testuser123@example.com", 
    "password": "securepass123",
    "role": "analyst"
}

def test_server_health():
    """Test if server is running"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is healthy")
            return True
        else:
            print(f"❌ Server health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot reach server: {e}")
        print("\n💡 To start the server, run:")
        print("   source venv/bin/activate && python3 main.py")
        return False

def register_user():
    """Register a test user"""
    try:
        response = requests.post(
            f"{BASE_URL}/auth/register",
            json=TEST_USER,
            timeout=10
        )
        if response.status_code == 200:
            user_data = response.json()
            print(f"✅ User registered: {user_data['username']}")
            return user_data
        else:
            print(f"⚠️  Registration response: {response.status_code}")
            print(f"   {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Registration failed: {e}")
        return None

def login_user():
    """Login and get JWT token"""
    try:
        login_data = {
            "username": TEST_USER["username"],
            "password": TEST_USER["password"]
        }
        response = requests.post(
            f"{BASE_URL}/auth/login",
            json=login_data,
            timeout=10
        )
        if response.status_code == 200:
            token_data = response.json()
            print(f"✅ User logged in: {token_data['user_info']['username']}")
            print(f"   Permissions: {', '.join(token_data['user_info']['permissions'])}")
            return token_data["access_token"]
        else:
            print(f"❌ Login failed: {response.status_code}")
            print(f"   {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Login request failed: {e}")
        return None

def upload_sample_data(token):
    """Upload sample CSV data"""
    try:
        # Create sample data if it doesn't exist
        sample_file = Path("sample_data.csv")
        if not sample_file.exists():
            print("📁 Creating sample dataset...")
            sample_data = """age,income,education_years,experience,salary
25,35000,16,2,40000
30,45000,18,5,55000
35,55000,16,8,65000
28,40000,14,3,45000
42,75000,20,15,85000
38,60000,18,10,70000
33,50000,16,6,58000
29,42000,17,4,48000
45,80000,22,18,95000
36,58000,19,9,68000
31,47000,16,5,52000
27,38000,15,3,43000
40,70000,20,12,78000
34,52000,17,7,62000
26,36000,14,2,41000
39,65000,19,11,75000
32,49000,16,6,56000
44,78000,21,16,88000
37,62000,18,10,72000
28,41000,15,4,47000"""
            sample_file.write_text(sample_data)
        
        headers = {"Authorization": f"Bearer {token}"}
        files = {"file": ("sample_data.csv", sample_file.open("rb"), "text/csv")}
        data = {
            "name": "Sample HR Dataset",
            "description": "Employee data for salary prediction"
        }
        
        response = requests.post(
            f"{BASE_URL}/data/upload",
            headers=headers,
            files=files,
            data=data,
            timeout=30
        )
        
        if response.status_code == 200:
            dataset_info = response.json()
            print(f"✅ Dataset uploaded: {dataset_info['dataset_id']}")
            print(f"   File size: {dataset_info['file_size']} bytes")
            print(f"   Encrypted: {dataset_info['is_encrypted']}")
            return dataset_info["dataset_id"]
        else:
            print(f"❌ Upload failed: {response.status_code}")
            print(f"   {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return None

def analyze_dataset(token, dataset_id):
    """Analyze the uploaded dataset"""
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(
            f"{BASE_URL}/data/{dataset_id}/analyze",
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            analysis = response.json()
            print(f"✅ Dataset analyzed: {analysis['shape']} shape")
            print(f"   Columns: {', '.join(analysis['columns'][:3])}...")
            print(f"   Missing values: {sum(analysis['missing_values'].values())}")
            return analysis
        else:
            print(f"❌ Analysis failed: {response.status_code}")
            print(f"   {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        return None

def train_model(token, dataset_id):
    """Train a linear regression model"""
    try:
        headers = {"Authorization": f"Bearer {token}"}
        training_request = {
            "dataset_id": dataset_id,
            "target_column": "salary",
            "feature_columns": ["age", "income", "education_years", "experience"],
            "model_type": "linear_regression",
            "test_size": 0.2,
            "cross_validation": True
        }
        
        response = requests.post(
            f"{BASE_URL}/models/train",
            headers=headers,
            json=training_request,
            timeout=60
        )
        
        if response.status_code == 200:
            model_info = response.json()
            print(f"✅ Model trained: {model_info['model_id']}")
            print(f"   R² Score: {model_info['performance']['r2_score']}")
            print(f"   RMSE: {model_info['performance']['rmse']}")
            return model_info["model_id"]
        else:
            print(f"❌ Training failed: {response.status_code}")
            print(f"   {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Training error: {e}")
        return None

def make_prediction(token, model_id):
    """Make a prediction with the trained model"""
    try:
        headers = {"Authorization": f"Bearer {token}"}
        prediction_request = {
            "input_data": {
                "age": 32,
                "income": 48000,
                "education_years": 16,
                "experience": 6
            }
        }
        
        response = requests.post(
            f"{BASE_URL}/models/{model_id}/predict",
            headers=headers,
            json=prediction_request,
            timeout=30
        )
        
        if response.status_code == 200:
            prediction = response.json()
            print(f"✅ Prediction made: ${prediction['predictions'][0]:,.2f}")
            return prediction
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return None

def main():
    """Run complete user testing workflow"""
    print("🧪 Starting User Testing Workflow...")
    print("=" * 50)
    
    # Step 1: Check server
    if not test_server_health():
        return False
    
    # Step 2: Register user
    print("\n📝 Step 1: User Registration")
    user_data = register_user()
    
    # Step 3: Login
    print("\n🔐 Step 2: User Login")
    token = login_user()
    if not token:
        return False
    
    # Step 4: Upload data
    print("\n📤 Step 3: Data Upload")
    dataset_id = upload_sample_data(token)
    if not dataset_id:
        return False
    
    # Step 5: Analyze data
    print("\n📊 Step 4: Data Analysis")
    analysis = analyze_dataset(token, dataset_id)
    if not analysis:
        return False
    
    # Step 6: Train model
    print("\n🤖 Step 5: Model Training")
    model_id = train_model(token, dataset_id)
    if not model_id:
        return False
    
    # Step 7: Make prediction
    print("\n🔮 Step 6: Making Prediction")
    prediction = make_prediction(token, model_id)
    if not prediction:
        return False
    
    print("\n" + "=" * 50)
    print("🎉 COMPLETE USER WORKFLOW SUCCESSFUL!")
    print("   ✅ Registration & Authentication")
    print("   ✅ Data Upload & Storage") 
    print("   ✅ Exploratory Data Analysis")
    print("   ✅ Machine Learning Model Training")
    print("   ✅ Model Predictions")
    print("\n🚀 Platform is ready for user testing!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)