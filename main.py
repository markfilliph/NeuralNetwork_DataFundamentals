"""Main application entry point."""

import os
import sys
from pathlib import Path

# Add backend to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Set environment variables if not already set
if not os.getenv("SECRET_KEY"):
    os.environ["SECRET_KEY"] = "development-secret-key-change-in-production"

if not os.getenv("ENCRYPTION_KEY"):
    # Generate a development encryption key
    import base64
    import os as _os
    key = base64.urlsafe_b64encode(_os.urandom(32)).decode()
    os.environ["ENCRYPTION_KEY"] = key
    print(f"🔑 Generated development encryption key: {key}")

from backend.api.routes import app
from backend.core.logging import audit_logger, EventType

# Log application startup
audit_logger.log_event(
    EventType.SYSTEM_START,
    outcome="success",
    details={
        "app_name": "Data Analysis Platform",
        "version": "1.0.0"
    }
)

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Data Analysis Platform API...")
    print(f"📊 Dashboard: http://localhost:8000/docs")
    print(f"🔐 Health Check: http://localhost:8000/health")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        reload=True,
        log_level="info"
    )