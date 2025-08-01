#!/usr/bin/env python3
"""Test script to verify server functionality."""

import sys
import time
import requests
import subprocess
import signal
import os
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

def start_server():
    """Start the server in background."""
    print("🚀 Starting Data Analysis Platform server...")
    
    # Activate virtual environment and start server
    cmd = [
        "bash", "-c", 
        "source venv/bin/activate && uvicorn main:app --host 127.0.0.1 --port 8000"
    ]
    
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid  # Create new process group
    )
    
    # Wait for server to start
    print("⏳ Waiting for server to start...")
    time.sleep(3)
    
    return process

def test_endpoints():
    """Test various API endpoints."""
    base_url = "http://127.0.0.1:8000"
    
    tests = [
        ("Health Check", f"{base_url}/health"),
        ("API Docs", f"{base_url}/docs"),
        ("OpenAPI Schema", f"{base_url}/openapi.json"),
    ]
    
    results = []
    
    for name, url in tests:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                results.append(f"✅ {name}: Working (Status: {response.status_code})")
            else:
                results.append(f"⚠️ {name}: Status {response.status_code}")
        except requests.exceptions.RequestException as e:
            results.append(f"❌ {name}: Failed - {e}")
    
    return results

def main():
    """Main test function."""
    print("🧪 Data Analysis Platform - Server Test")
    print("=" * 50)
    
    # Test import first
    try:
        from main import app
        print("✅ Application imports successfully")
        print(f"   - Title: {app.title}")
        print(f"   - Version: {app.version}")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return 1
    
    # Start server
    server_process = None
    try:
        server_process = start_server()
        
        # Test endpoints
        print("\\n🔍 Testing API endpoints...")
        results = test_endpoints()
        
        for result in results:
            print(f"   {result}")
        
        # Check if server is responsive
        try:
            response = requests.get("http://127.0.0.1:8000/health", timeout=2)
            if response.status_code == 200:
                print("\\n🎉 SUCCESS: Server is fully operational!")
                print("🌐 Access your application at:")
                print("   - API Documentation: http://127.0.0.1:8000/docs")
                print("   - Health Check: http://127.0.0.1:8000/health")
                print("   - Alternative Docs: http://127.0.0.1:8000/redoc")
                return 0
            else:
                print(f"\\n⚠️ Server responding but with status: {response.status_code}")
                return 1
        except Exception as e:
            print(f"\\n❌ Server not responding: {e}")
            return 1
            
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return 1
    
    finally:
        # Clean up server process
        if server_process:
            try:
                os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
                print("\\n🛑 Server stopped")
            except:
                pass

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)