# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **Data Analysis and Prediction Platform (DAPP)** - a data science platform for educational purposes focused on data loading, exploratory data analysis, and linear regression modeling using Jupyter notebooks.

## Architecture

The project is designed around documentation-driven development with a planned modular architecture:

### Core Components
- **Data Service**: Handles Excel/Access file loading, preprocessing, and validation
- **Model Service**: Linear regression training, evaluation, and persistence  
- **API Layer**: FastAPI-based REST endpoints for data and model operations
- **Frontend**: Jupyter notebook interface with IPython widgets

### Directory Structure (Current)
```
backend/           # Python FastAPI backend
├── api/           # FastAPI routes and middleware
├── core/          # Configuration, exceptions, logging
├── models/        # Database models and repositories
├── services/      # Business logic (data, model, export services)
├── utils/         # Validators, transformers, file handlers
└── tests/         # Comprehensive test suite

frontend/          # Next.js React frontend
├── src/app/       # Next.js app router pages
├── src/components/# React components
├── src/contexts/  # React contexts
└── src/types/     # TypeScript definitions

data/              # Data storage
├── uploads/       # Temporary file uploads
├── processed/     # Encrypted processed datasets
└── app.db         # SQLite database

scripts/           # Deployment and utility scripts
tests/             # Integration tests
docs/              # Project documentation
logs/              # Application and audit logs
notebooks/         # Jupyter notebooks
sample_datasets/   # Example datasets
```

## Technology Stack

- **Language**: Python 3.8+
- **Core Libraries**: pandas, scikit-learn, numpy, matplotlib, seaborn
- **File Handling**: openpyxl (Excel), pyodbc (Access)
- **API Framework**: FastAPI (implemented with modular routes)
- **Interface**: Jupyter Lab/Notebook with IPywidgets
- **Testing**: pytest with 80% coverage requirement

## Development Approach

The project is now fully implemented with production-ready architecture. The platform follows these principles:

### Code Standards
- Follow PEP 8 with 88-character line limit
- Type hints required for all function signatures
- Google-style docstrings for all public functions/classes
- Meaningful variable names (no single letters except loops)

### Security Requirements
- Input validation for all file uploads
- File type and size validation (max 100MB)
- Sanitization of Excel formulas and potential malicious content
- No hardcoded credentials

### Testing Strategy
- Minimum 80% code coverage
- Unit tests for all public methods
- Property-based testing for data processing functions
- Integration tests for API endpoints

## Common Development Tasks

*Note: No actual build/test commands exist yet as this is in planning phase*

When implemented, the project will likely use:
- `pytest tests/ --cov=project --cov-report=xml` (testing)
- `black .` and `isort .` (formatting)
- `flake8 . --max-line-length=88` (linting)
- `mypy . --strict` (type checking)

## Key Constraints

- Memory usage should not exceed 4GB
- Must work offline after initial setup
- 4-hour implementation timeframe for MVP
- Support files up to 100MB
- Target datasets up to 1GB

## Code Review Requirements

All code changes should follow the checklist in `CodeReviewChecklist.md`:
- Functionality verification and edge case handling
- PEP 8 compliance and meaningful naming
- Comprehensive docstrings and comments
- Unit tests with 80%+ coverage
- Security validation (no hardcoded credentials, input validation)
- Performance optimization and error handling