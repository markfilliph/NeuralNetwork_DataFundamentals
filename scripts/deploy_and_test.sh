#!/bin/bash
# DAPP Complete Deployment and Testing Script
# This script will properly deploy both backend and frontend and test functionality

set -e  # Exit on any error

echo "🚀 DAPP Complete Deployment and Testing"
echo "========================================"

# Check if we're in the right directory (adjust for new structure)
if [ ! -f "main.py" ] || [ ! -d "backend" ]; then
    echo "❌ Error: Must run from project root directory"
    echo "Expected to find main.py and backend/ directory"
    exit 1
fi

# Function to cleanup on exit
cleanup() {
    echo "🧹 Cleaning up processes..."
    # Kill background processes if they exist
    jobs -p | xargs -r kill
}
trap cleanup EXIT

echo "📦 Step 1: Backend Environment Setup"
echo "-----------------------------------"

# Check virtual environment
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please create it first:"
    echo "python3 -m venv venv"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate
echo "✅ Virtual environment activated"

# Set environment variables
export SECRET_KEY="development-secret-key-for-testing-32chars"
export ENCRYPTION_KEY="development-encryption-key-32chars"
export DATABASE_URL="sqlite:///./data/app.db"
export DEBUG="true"
echo "✅ Environment variables set"

# Check database exists
if [ ! -f "data/app.db" ]; then
    echo "❌ Database not found at data/app.db"
    exit 1
fi
echo "✅ Database found"

echo ""
echo "🖥️  Step 2: Start Backend Server"
echo "--------------------------------"

# Start backend server in background
python3 main.py &
BACKEND_PID=$!
echo "✅ Backend server started (PID: $BACKEND_PID)"

# Wait for backend to be ready
echo "⏳ Waiting for backend to start..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        break
    fi
    sleep 1
done

# Test backend health
echo "🔍 Testing backend health..."
HEALTH_RESPONSE=$(curl -s http://localhost:8000/health)
if echo "$HEALTH_RESPONSE" | grep -q "healthy"; then
    echo "✅ Backend is healthy: $HEALTH_RESPONSE"
else
    echo "❌ Backend health check failed: $HEALTH_RESPONSE"
    exit 1
fi

echo ""
echo "🌐 Step 3: Frontend Setup"
echo "-------------------------"

# Check if frontend dependencies are installed
if [ ! -d "frontend/node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    cd frontend
    npm install
    cd ..
fi
echo "✅ Frontend dependencies ready"

# Start frontend server in background
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..
echo "✅ Frontend server started (PID: $FRONTEND_PID)"

# Wait for frontend to be ready
echo "⏳ Waiting for frontend to start..."
sleep 10

echo ""
echo "🧪 Step 4: Testing Complete Workflow"
echo "======================================"

# Test 1: Register a new user
echo "🔐 Test 1: User Registration"
REGISTER_RESPONSE=$(curl -s -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username": "testuser_deploy", "email": "deploy@test.com", "password": "testpass123"}')

if echo "$REGISTER_RESPONSE" | grep -q "access_token"; then
    echo "✅ User registration successful"
    # Extract token for further tests
    TOKEN=$(echo "$REGISTER_RESPONSE" | grep -o '"access_token":"[^"]*"' | cut -d'"' -f4)
else
    echo "ℹ️  Registration response: $REGISTER_RESPONSE"
    # Try to login instead
    echo "🔐 Trying login instead..."
    LOGIN_RESPONSE=$(curl -s -X POST http://localhost:8000/auth/login \
      -H "Content-Type: application/json" \
      -d '{"username": "testuser_deploy", "password": "testpass123"}')
    
    if echo "$LOGIN_RESPONSE" | grep -q "access_token"; then
        echo "✅ User login successful"
        TOKEN=$(echo "$LOGIN_RESPONSE" | grep -o '"access_token":"[^"]*"' | cut -d'"' -f4)
    else
        echo "❌ Both registration and login failed: $LOGIN_RESPONSE"
        exit 1
    fi
fi

# Test 2: Authentication check
echo "🔍 Test 2: Authentication Check"
AUTH_RESPONSE=$(curl -s -H "Authorization: Bearer $TOKEN" http://localhost:8000/auth/me)
if echo "$AUTH_RESPONSE" | grep -q "user_id"; then
    echo "✅ Authentication working"
else
    echo "❌ Authentication failed: $AUTH_RESPONSE"
    exit 1
fi

# Test 3: File Upload
echo "📁 Test 3: File Upload"
# Create test file
echo "name,age,score
Alice,25,85.5
Bob,30,92.0
Charlie,22,78.3" > /tmp/test_upload.csv

UPLOAD_RESPONSE=$(curl -s -X POST http://localhost:8000/data/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@/tmp/test_upload.csv")

if echo "$UPLOAD_RESPONSE" | grep -q "dataset_id"; then
    echo "✅ File upload successful"
    DATASET_ID=$(echo "$UPLOAD_RESPONSE" | grep -o '"dataset_id":"[^"]*"' | cut -d'"' -f4)
else
    echo "❌ File upload failed: $UPLOAD_RESPONSE"
    exit 1
fi

# Test 4: List datasets
echo "📋 Test 4: List Datasets"
DATASETS_RESPONSE=$(curl -s -H "Authorization: Bearer $TOKEN" http://localhost:8000/data/datasets)
if echo "$DATASETS_RESPONSE" | grep -q "datasets"; then
    echo "✅ Dataset listing working"
else
    echo "❌ Dataset listing failed: $DATASETS_RESPONSE"
    exit 1
fi

# Cleanup test file
rm -f /tmp/test_upload.csv

echo ""
echo "🎉 DEPLOYMENT SUCCESSFUL!"
echo "========================="
echo ""
echo "📡 Backend API: http://localhost:8000"
echo "📚 API Documentation: http://localhost:8000/docs"
echo "🌐 Frontend Dashboard: http://localhost:3000"
echo ""
echo "🧪 All tests passed:"
echo "  ✅ User registration/login"
echo "  ✅ Authentication"
echo "  ✅ File upload"
echo "  ✅ Dataset listing"
echo ""
echo "🔑 Test credentials:"
echo "  Username: testuser_deploy"
echo "  Password: testpass123"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Keep script running until user stops it
wait