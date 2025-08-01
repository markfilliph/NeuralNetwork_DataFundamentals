# Testing Setup Guide

## Pre-Testing Checklist ✅

Before running tests, ensure the following setup steps are completed:

### 1. **Dependencies Installation** 🔧
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. **Environment Configuration** 🌍
```bash
# Option 1: Use default generated keys (for development)
python3 main.py  # Will auto-generate SECRET_KEY and ENCRYPTION_KEY

# Option 2: Set custom environment variables
export SECRET_KEY="your-secure-secret-key-here"
export ENCRYPTION_KEY="your-base64-encoded-32-byte-key-here"
export REDIS_URL="redis://localhost:6379"  # Optional
```

### 3. **Directory Structure** 📁
The following directories will be created automatically if they don't exist:
```
data/
├── uploads/      # User uploaded files
├── processed/    # Processed datasets  
└── app.db        # SQLite database

logs/
├── audit.log     # Audit events
└── security.log  # Security events
```

### 4. **Database Initialization** 💾
The SQLite database and tables are created automatically on first run via:
- `backend/models/database.py` - DatabaseManager._init_database()
- Creates: users, user_sessions, datasets, models, audit_logs, api_keys tables
- Includes all necessary indexes for performance

### 5. **Redis Setup (Optional)** 🔴
For production caching and async processing:
```bash
# Install Redis (Ubuntu/Debian)
sudo apt-get install redis-server
sudo systemctl start redis-server

# Or use Docker
docker run -p 6379:6379 -d redis:alpine
```

### 6. **Test Dependencies** 🧪
Additional testing packages (already in requirements.txt):
- pytest>=6.2.0
- pytest-cov>=2.12.0
- pytest-asyncio>=0.15.0
- hypothesis>=6.0.0

## **Quick Start Testing Commands** 🚀

### Run All Tests
```bash
# Basic test run
pytest backend/tests/

# With coverage report
pytest backend/tests/ --cov=backend --cov-report=xml

# Verbose output
pytest backend/tests/ -v

# Run specific test file
pytest backend/tests/test_auth_service.py -v
```

### Run Application
```bash
# Start the API server
python3 main.py

# Will be available at:
# - API Documentation: http://localhost:8000/docs
# - Health Check: http://localhost:8000/health
# - Alternative docs: http://localhost:8000/redoc
```

### Test API Endpoints Manually
```bash
# Health check
curl http://localhost:8000/health

# Register user
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"testuser","email":"test@example.com","password":"securepass123"}'

# Login
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"testuser","password":"securepass123"}'
```

## **Architecture Components Status** ✅

All major components are implemented and ready for testing:

### **✅ Security Layer**
- Input validation and sanitization
- Authentication service (JWT with jose)
- Authorization service (RBAC)
- Encryption service (Fernet)
- Rate limiting and audit logging

### **✅ Data Layer** 
- Database models and repositories (SQLite)
- File handlers with security validation
- Data sanitizers for Excel/CSV content

### **✅ API Layer**
- FastAPI application with comprehensive routes
- Authentication middleware
- Error handling and validation
- OpenAPI documentation

### **✅ Scalability Layer**
- Load balancer with multiple algorithms
- Session affinity management
- Health monitoring and metrics
- Async processing (Celery + fallback)
- Multi-level caching (Redis + memory)

### **✅ Configuration**
- Environment-based configuration
- Automatic key generation for development
- Production-ready security settings

## **Known Dependencies** 📦

Core runtime dependencies that must be installed:
- fastapi>=0.68.0
- uvicorn>=0.15.0  
- python-jose[cryptography]>=3.3.0
- passlib[bcrypt]>=1.7.4
- cryptography>=3.0.0
- pandas>=1.3.0
- scikit-learn>=1.0.0
- redis>=4.0.0
- celery>=5.2.0
- psutil>=5.8.0

## **Testing Strategy** 🎯

1. **Unit Tests**: Test individual components and services
2. **Integration Tests**: Test component interactions  
3. **Security Tests**: Test authentication, authorization, input validation
4. **Performance Tests**: Test caching, database queries, async processing
5. **API Tests**: Test all endpoints with various scenarios

## **Troubleshooting** 🔧

### Common Issues:
1. **ModuleNotFoundError**: Run `pip install -r requirements.txt`
2. **Permission Denied**: Check file system permissions for data/ and logs/ directories
3. **Redis Connection Failed**: Caching will fallback to memory-only mode
4. **Database Locked**: Ensure no other processes are accessing the SQLite database

### Debug Mode:
```bash
export DEBUG=true
python3 main.py
```

This will enable:
- Detailed error messages
- SQL query logging  
- Enhanced audit logging
- Auto-reload on code changes

---

**Ready for Testing!** 🎉

The platform is now fully configured and ready for comprehensive testing. All security measures, scalability features, and performance optimizations are in place.