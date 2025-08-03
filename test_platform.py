"""End-to-end test script for the Data Analysis Platform."""

import requests
import json
import pandas as pd
import numpy as np
from pathlib import Path

BASE_URL = "http://localhost:8003"

def test_basic_endpoints():
    """Test basic endpoints that don't require authentication."""
    print("🧪 Testing Basic Endpoints...")
    
    # Test health
    response = requests.get(f"{BASE_URL}/health")
    print(f"Health Check: {response.status_code} - {response.json()}")
    
    # Test root
    response = requests.get(f"{BASE_URL}/")
    print(f"Root Endpoint: {response.status_code} - {response.json()}")
    
    return response.status_code == 200

def test_api_documentation():
    """Test if API documentation is accessible."""
    print("\n📚 Testing API Documentation...")
    
    try:
        response = requests.get(f"{BASE_URL}/docs")
        print(f"API Docs: {response.status_code} - {'Available' if response.status_code == 200 else 'Not Available'}")
        
        response = requests.get(f"{BASE_URL}/openapi.json")
        if response.status_code == 200:
            openapi_spec = response.json()
            print(f"OpenAPI Spec: Found {len(openapi_spec.get('paths', {}))} endpoint paths")
            return True
    except Exception as e:
        print(f"Documentation test failed: {e}")
    
    return False

def test_auth_endpoints():
    """Test authentication endpoints."""
    print("\n🔐 Testing Authentication...")
    
    # Test registration
    reg_data = {
        "username": "e2etest",
        "email": "e2etest@example.com",
        "password": "password123",
        "role": "analyst"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/auth/register", json=reg_data)
        print(f"Registration: {response.status_code}")
        if response.status_code != 200:
            print(f"Registration Error: {response.text}")
    except Exception as e:
        print(f"Registration failed: {e}")
    
    # Test login
    login_data = {
        "username": "e2etest", 
        "password": "password123"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/auth/login", json=login_data)
        print(f"Login: {response.status_code}")
        if response.status_code == 200:
            token_data = response.json()
            print(f"Login successful: {token_data.get('access_token', 'No token')[:20]}...")
            return token_data.get('access_token')
        else:
            print(f"Login Error: {response.text}")
    except Exception as e:
        print(f"Login failed: {e}")
    
    return None

def create_test_data():
    """Create test CSV data for upload."""
    print("\n📊 Creating Test Data...")
    
    # Create sample dataset
    np.random.seed(42)
    data = {
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(0, 1, 100),
        'feature3': np.random.randint(1, 5, 100),
        'target': np.random.normal(10, 3, 100)
    }
    
    df = pd.DataFrame(data)
    test_file = Path("test_data.csv")
    df.to_csv(test_file, index=False)
    print(f"Created test file: {test_file} ({len(df)} rows, {len(df.columns)} columns)")
    
    return test_file

def test_data_upload(token=None):
    """Test data upload functionality."""
    print("\n📤 Testing Data Upload...")
    
    test_file = create_test_data()
    
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    try:
        with open(test_file, 'rb') as f:
            files = {'file': (test_file.name, f, 'text/csv')}
            response = requests.post(f"{BASE_URL}/data/upload", files=files, headers=headers)
        
        print(f"Upload: {response.status_code}")
        if response.status_code == 200:
            upload_result = response.json()
            print(f"Upload successful: Dataset ID {upload_result.get('dataset_id')}")
            return upload_result.get('dataset_id')
        else:
            print(f"Upload Error: {response.text}")
    except Exception as e:
        print(f"Upload failed: {e}")
    finally:
        # Clean up test file
        if test_file.exists():
            test_file.unlink()
    
    return None

def test_database_connection():
    """Test if database connection works."""
    print("\n💾 Testing Database Connection...")
    
    try:
        import sys
        sys.path.append('.')
        
        from backend.models.database import db_manager
        
        # Test basic database operations
        connection = db_manager.get_connection()
        if connection:
            print("✅ Database connection successful")
            
            # Test table creation
            cursor = connection.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            print(f"📋 Found {len(tables)} database tables")
            
            return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
    
    return False

def main():
    """Run comprehensive end-to-end tests."""
    print("🚀 STARTING END-TO-END PLATFORM TESTS")
    print("=" * 50)
    
    results = {}
    
    # Test basic functionality
    results['basic_endpoints'] = test_basic_endpoints()
    results['api_docs'] = test_api_documentation()
    results['database'] = test_database_connection()
    
    # Test authentication (expect some issues)
    token = test_auth_endpoints()
    results['auth'] = token is not None
    
    # Test data operations (may fail without proper auth)
    dataset_id = test_data_upload(token)
    results['data_upload'] = dataset_id is not None
    
    # Summary
    print("\n" + "=" * 50)
    print("🎯 TEST RESULTS SUMMARY:")
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test.replace('_', ' ').title()}: {status}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests >= total_tests * 0.6:  # 60% threshold
        print("🎉 Platform is functional for basic testing!")
    else:
        print("⚠️  Platform needs fixes before full testing")

if __name__ == "__main__":
    main()