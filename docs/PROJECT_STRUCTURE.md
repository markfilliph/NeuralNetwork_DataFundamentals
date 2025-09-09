# Project Structure - Data Analysis Platform (DAPP)

## Overview
This document outlines the clean, organized structure of the Data Analysis Platform after reorganization and cleanup.

## Directory Structure

```
NN_DataFundamentals/
├── README.md                    # Main project documentation
├── CLAUDE.md                    # Claude Code instructions
├── DEPLOYMENT_FIXES.md          # Deployment issues and fixes
├── main.py                      # Application entry point
├── requirements.txt             # Python dependencies
├── pyproject.toml              # Python project configuration
├── pytest.ini                 # Pytest configuration
├── .env.template               # Environment variables template
├── .gitignore                  # Git ignore rules
│
├── backend/                    # Backend Python application
│   ├── api/                    # FastAPI routes and middleware
│   │   ├── routes/             # API route modules
│   │   │   ├── auth.py         # Authentication endpoints
│   │   │   ├── data.py         # Data management endpoints
│   │   │   └── models.py       # ML model endpoints
│   │   ├── main.py             # FastAPI app initialization
│   │   └── middleware.py       # Custom middleware
│   ├── core/                   # Core functionality
│   │   ├── config.py           # Configuration management
│   │   ├── exceptions.py       # Custom exceptions
│   │   └── logging.py          # Logging configuration
│   ├── models/                 # Database models
│   │   └── database.py         # Database operations and models
│   ├── services/               # Business logic services
│   │   ├── auth_service.py     # Authentication service
│   │   ├── cache_service.py    # Redis cache service (optional)
│   │   ├── cache_service_simple.py # In-memory cache service
│   │   ├── data_service.py     # Data processing service
│   │   ├── encryption_service.py # Data encryption service
│   │   ├── export_service.py   # Data export service
│   │   ├── model_service.py    # ML model service
│   │   └── rbac_service_db.py  # Role-based access control
│   ├── utils/                  # Utility modules
│   │   ├── cache_utils.py      # Cache decorators and utilities
│   │   ├── file_handlers.py    # File upload/processing utilities
│   │   ├── sanitizers.py       # Data sanitization utilities
│   │   ├── validators.py       # Input validation utilities
│   │   └── visualization_utils.py # Data visualization utilities
│   └── tests/                  # Backend unit tests
│       ├── test_auth_service.py
│       ├── test_cache_service.py
│       ├── test_data_service.py
│       ├── test_database.py
│       ├── test_encryption_service.py
│       ├── test_file_handlers.py
│       ├── test_middleware.py
│       ├── test_model_service.py
│       └── [other test files]
│
├── frontend/                   # Next.js React frontend
│   ├── package.json            # Node.js dependencies
│   ├── next.config.js          # Next.js configuration
│   ├── tsconfig.json           # TypeScript configuration
│   ├── src/                    # Source code
│   │   ├── app/                # Next.js app router
│   │   │   ├── datasets/       # Dataset management pages
│   │   │   ├── models/         # ML model pages
│   │   │   └── [other pages]
│   │   ├── components/         # React components
│   │   ├── contexts/           # React contexts (auth, websocket)
│   │   ├── hooks/              # Custom React hooks
│   │   ├── lib/                # Utility libraries
│   │   └── types/              # TypeScript type definitions
│   └── node_modules/           # Node.js dependencies
│
├── data/                       # Data storage
│   ├── app.db                  # SQLite database
│   ├── uploads/                # Temporary file uploads
│   └── processed/              # Processed/encrypted datasets
│
├── logs/                       # Application logs
│   ├── audit.log               # Security audit logs
│   └── security.log            # Security event logs
│
├── notebooks/                  # Jupyter notebooks
│   ├── templates/              # Reusable notebook templates
│   ├── examples/               # Tutorial notebooks
│   └── user_notebooks/         # User-created analysis notebooks
│
├── sample_datasets/            # Example datasets for testing
│
├── scripts/                    # Deployment and utility scripts
│   ├── deploy_and_test.sh      # Complete deployment testing script
│   ├── start_backend.sh        # Backend startup script
│   └── env_file.sh             # Environment setup script
│
├── tests/                      # Integration and end-to-end tests
│   └── test_complete_workflow.py # Full workflow integration test
│
├── docs/                       # Project documentation
│   ├── Backend.md              # Backend architecture documentation
│   ├── Frontend.md             # Frontend architecture documentation
│   ├── ImplementationPlan.md   # Implementation roadmap
│   ├── Requirements.md         # Project requirements
│   ├── CodeReviewChecklist.md  # Code review guidelines
│   ├── REFACTORING_SUMMARY.md  # Refactoring history
│   ├── README_HYBRID.md        # Hybrid architecture notes
│   ├── INSTALLATION_STATUS.md  # Installation and setup status
│   ├── TESTING_SETUP.md        # Testing configuration
│   ├── USER_TESTING_GUIDE.md   # User testing instructions
│   ├── DASHBOARD_ACCESS.md     # Dashboard access guide
│   └── PROJECT_STRUCTURE.md    # This file
│
└── venv/                       # Python virtual environment
```

## Key Files and Their Purpose

### Root Level
- **main.py**: Primary application entry point with environment setup
- **requirements.txt**: Python dependencies list
- **pyproject.toml**: Python project configuration and tool settings
- **.env.template**: Template for environment variables

### Backend Architecture
- **api/**: RESTful API endpoints using FastAPI
- **core/**: Core functionality (config, logging, exceptions)
- **models/**: Database models and data access layer
- **services/**: Business logic and service layer
- **utils/**: Shared utilities and helper functions
- **tests/**: Comprehensive unit test suite

### Frontend Architecture  
- **src/app/**: Next.js 14 app router with pages
- **src/components/**: Reusable React components
- **src/contexts/**: React contexts for state management
- **src/hooks/**: Custom React hooks
- **src/lib/**: Frontend utility libraries

### Data Management
- **data/**: SQLite database and file storage
- **logs/**: Application and security logging
- **notebooks/**: Jupyter notebook environment
- **sample_datasets/**: Example datasets for development

### Deployment & Testing
- **scripts/**: Deployment automation scripts
- **tests/**: Integration and workflow tests
- **docs/**: Comprehensive project documentation

## Removed Files
During cleanup, the following redundant files were removed:
- Duplicate server runners (main_py.py, run_server.py, minimal_server.py)
- Redundant test files (test_minimal.py, test_platform.py, etc.)
- Old initialization scripts (backend_init.py, package_inits.py)
- Duplicate documentation files (.txt versions)
- Archive directory with old refactoring files

## Architecture Benefits
1. **Clear Separation**: Backend, frontend, and data storage are clearly separated
2. **Modular Design**: Services and utilities are properly organized
3. **Testing Structure**: Comprehensive test coverage with organized test files
4. **Documentation**: Centralized documentation in docs/ directory
5. **Deployment Ready**: Scripts organized for easy deployment
6. **Development Friendly**: Clear structure for continued development

## Running the Application
From the project root:
```bash
# Deploy and test everything
./scripts/deploy_and_test.sh

# Start backend only  
./scripts/start_backend.sh

# Run integration tests
python tests/test_complete_workflow.py
```