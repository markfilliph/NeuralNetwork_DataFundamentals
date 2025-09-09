# Environment Configuration for Neural Network Data Platform

# Application Settings
DEBUG=true
HOST=0.0.0.0
PORT=8000

# Security
SECRET_KEY=your-super-secret-key-change-in-production-please
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Database Configuration
DATABASE_URL=sqlite:///./data/app.db

# File Upload Settings
MAX_FILE_SIZE=104857600  # 100MB in bytes
UPLOAD_DIR=data/uploads
PROCESSED_DIR=data/processed
ALLOWED_EXTENSIONS=.csv,.xlsx,.xls,.pdf

# Neural Network Settings
DEFAULT_MODEL_TYPE=neural_network
MODEL_SAVE_DIR=models/saved
BATCH_SIZE=32
EPOCHS=100
LEARNING_RATE=0.001

# Logging Configuration
LOG_LEVEL=INFO
LOG_DIR=logs

# CORS Settings (comma-separated list)
CORS_ORIGINS=*

# Performance Settings
WORKERS=1
WORKER_CONNECTIONS=1000

# TensorFlow Settings
TF_CPP_MIN_LOG_LEVEL=2  # Suppress TensorFlow info/warning messages
CUDA_VISIBLE_DEVICES=-1  # Use CPU only (set to 0,1,... for GPU)

# Production Settings (uncomment for production)
# DEBUG=false
# SECRET_KEY=generate-a-real-secret-key-with-openssl-rand-base64-32
# DATABASE_URL=postgresql://user:password@localhost/dbname
# CORS_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
# LOG_LEVEL=WARNING
# MAX_FILE_SIZE=52428800  # 50MB for production

# Optional: External Services
# REDIS_URL=redis://localhost:6379
# CELERY_BROKER_URL=redis://localhost:6379
# SENTRY_DSN=your-sentry-dsn-for-error-tracking