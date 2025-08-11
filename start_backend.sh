#!/bin/bash
# Start DAPP Backend Server

echo "🚀 Starting DAPP Backend Server..."

# Activate virtual environment
source venv/bin/activate

# Set environment variables
export SECRET_KEY="development-secret-key-for-testing-32chars"
export ENCRYPTION_KEY="development-encryption-key-32chars"
export DATABASE_URL="sqlite:///./data/app.db"
export DEBUG="true"

# Start the server
echo "✅ Environment configured"
echo "📡 Starting FastAPI server on http://localhost:8000"
echo "📚 API docs will be available at http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python main.py