#!/usr/bin/env python3
"""Debug test script to isolate issues"""

import requests
import json

BASE_URL = "http://localhost:8000"

# Get token from previous login
TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiMGRlYjQzYWEtNDAyZi00M2ZhLThjNGItMGI3YWQ5YmZlY2UzIiwidXNlcm5hbWUiOiJ1c2VyMDAxIiwicm9sZSI6ImFuYWx5c3QiLCJleHAiOjE3NTQwODExNTksImlhdCI6MTc1NDA3NzU1OX0.1vj6jfla0aaKCQmqWwXQ5e6-VrBeJ_OO-nbefs9odUk"

def test_simple_analysis():
    """Test analysis with a simple dataset ID"""
    # First check datasets
    headers = {"Authorization": f"Bearer {TOKEN}"}
    
    print("Testing dataset list...")
    response = requests.get(f"{BASE_URL}/data/datasets", headers=headers)
    print(f"Datasets response: {response.status_code}")
    if response.status_code == 200:
        datasets = response.json()
        print(f"Found {len(datasets.get('datasets', []))} datasets")
        
        if datasets.get('datasets'):
            dataset_id = datasets['datasets'][0]['dataset_id'] 
            print(f"Testing analysis for dataset: {dataset_id}")
            
            # Test analysis
            analysis_response = requests.get(
                f"{BASE_URL}/data/{dataset_id}/analyze",
                headers=headers
            )
            print(f"Analysis response: {analysis_response.status_code}")
            print(f"Analysis content: {analysis_response.text[:500]}...")
    else:
        print(f"Datasets error: {response.text}")

if __name__ == "__main__":
    test_simple_analysis()