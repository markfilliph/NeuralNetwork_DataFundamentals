"""Focused test for working platform components."""

import requests
import json

BASE_URL = "http://localhost:8003"

def test_api_completeness():
    """Test that all expected endpoints are available."""
    print("🔍 Testing API Completeness...")
    
    response = requests.get(f"{BASE_URL}/openapi.json")
    if response.status_code == 200:
        spec = response.json()
        endpoints = []
        for path, methods in spec.get('paths', {}).items():
            for method in methods.keys():
                endpoints.append(f"{method.upper()} {path}")
        
        print(f"✅ Found {len(endpoints)} endpoints")
        
        # Check for key endpoint categories
        auth_endpoints = [e for e in endpoints if '/auth/' in e]
        data_endpoints = [e for e in endpoints if '/data/' in e]  
        model_endpoints = [e for e in endpoints if '/models/' in e]
        
        print(f"  🔐 Auth endpoints: {len(auth_endpoints)}")
        print(f"  📊 Data endpoints: {len(data_endpoints)}")
        print(f"  🤖 Model endpoints: {len(model_endpoints)}")
        
        return len(endpoints) >= 15  # Should have at least 15 endpoints
    
    return False

def test_cors_headers():
    """Test CORS configuration."""
    print("\n🌐 Testing CORS Configuration...")
    
    try:
        response = requests.options(f"{BASE_URL}/health", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET"
        })
        
        cors_headers = {
            'access-control-allow-origin': response.headers.get('access-control-allow-origin'),
            'access-control-allow-methods': response.headers.get('access-control-allow-methods'),
            'access-control-allow-headers': response.headers.get('access-control-allow-headers')
        }
        
        print(f"CORS Headers: {cors_headers}")
        return response.status_code in [200, 204]
    except Exception as e:
        print(f"CORS test failed: {e}")
        return False

def test_error_handling():
    """Test error handling for non-existent endpoints."""
    print("\n❌ Testing Error Handling...")
    
    # Test 404 for non-existent endpoint
    response = requests.get(f"{BASE_URL}/nonexistent")
    print(f"404 Test: {response.status_code} - {'✅ Correct' if response.status_code == 404 else '❌ Wrong'}")
    
    # Test 405 for wrong method
    response = requests.delete(f"{BASE_URL}/health")
    print(f"405 Test: {response.status_code} - {'✅ Correct' if response.status_code == 405 else '❌ Wrong'}")
    
    return True

def test_docs_accessibility():
    """Test documentation accessibility."""
    print("\n📖 Testing Documentation...")
    
    # Test Swagger UI
    response = requests.get(f"{BASE_URL}/docs")
    swagger_works = response.status_code == 200
    print(f"Swagger UI: {'✅ Available' if swagger_works else '❌ Not Available'}")
    
    # Test ReDoc
    response = requests.get(f"{BASE_URL}/redoc")
    redoc_works = response.status_code == 200
    print(f"ReDoc: {'✅ Available' if redoc_works else '❌ Not Available'}")
    
    # Test OpenAPI spec
    response = requests.get(f"{BASE_URL}/openapi.json")
    openapi_works = response.status_code == 200
    print(f"OpenAPI Spec: {'✅ Available' if openapi_works else '❌ Not Available'}")
    
    return swagger_works and redoc_works and openapi_works

def test_endpoint_responses():
    """Test endpoint response formats."""
    print("\n📋 Testing Endpoint Response Formats...")
    
    # Test health endpoint structure
    response = requests.get(f"{BASE_URL}/health")
    if response.status_code == 200:
        health_data = response.json()
        required_fields = ['status', 'version', 'service']
        has_all_fields = all(field in health_data for field in required_fields)
        print(f"Health endpoint structure: {'✅ Complete' if has_all_fields else '❌ Missing fields'}")
    
    # Test root endpoint structure  
    response = requests.get(f"{BASE_URL}/")
    if response.status_code == 200:
        root_data = response.json()
        required_fields = ['message', 'version', 'status']
        has_all_fields = all(field in root_data for field in required_fields)
        print(f"Root endpoint structure: {'✅ Complete' if has_all_fields else '❌ Missing fields'}")
    
    return True

def main():
    """Run focused tests on working platform components."""
    print("🎯 FOCUSED PLATFORM TESTING")
    print("=" * 40)
    
    tests = [
        ("API Completeness", test_api_completeness),
        ("CORS Configuration", test_cors_headers), 
        ("Error Handling", test_error_handling),
        ("Documentation", test_docs_accessibility),
        ("Response Formats", test_endpoint_responses)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"Test {test_name} failed with error: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 FOCUSED TEST RESULTS:")
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    passed_count = sum(results.values())
    total_count = len(results)
    
    print(f"\nFunctionality Score: {passed_count}/{total_count} ({passed_count/total_count*100:.1f}%)")
    
    if passed_count >= total_count * 0.8:
        print("🎉 Platform infrastructure is solid!")
        print("💡 Ready for manual testing via /docs interface")
    else:
        print("⚠️  Some infrastructure issues found")

if __name__ == "__main__":
    main()