# Data Analysis and Prediction Platform (DAPP)

A comprehensive data science platform for educational purposes, focused on data loading, exploratory data analysis, and linear regression modeling using Jupyter notebooks.

## 🚀 Features

- **Data Loading**: Support for Excel and Access files with robust validation
- **Data Processing**: Automated preprocessing and cleaning pipelines
- **Exploratory Data Analysis**: Interactive visualizations and statistical summaries
- **Machine Learning**: Linear regression modeling with evaluation metrics
- **Security**: Input validation, file sanitization, and secure authentication
- **API**: FastAPI-based REST endpoints for data and model operations
- **Interactive Interface**: Jupyter notebook integration with IPython widgets

## 📋 Requirements

- Python 3.8+
- Memory: Up to 4GB usage
- Storage: Support for files up to 100MB, datasets up to 1GB
- Works offline after initial setup

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/markfilliph/NeuralNetwork_DataFundamentals.git
   cd NeuralNetwork_DataFundamentals
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python main.py
   ```

## 📁 Project Structure

```
├── backend/
│   ├── api/           # FastAPI routes and middleware
│   ├── core/          # Configuration, exceptions, logging
│   ├── models/        # ML models and database schemas
│   ├── services/      # Business logic (auth, data, model services)
│   ├── utils/         # Validators, transformers, file handlers
│   └── tests/         # Comprehensive test suite
├── notebooks/
│   ├── templates/     # Reusable notebook templates
│   ├── examples/      # Tutorial notebooks
│   └── user_notebooks/# User-created analysis notebooks
├── data/
│   ├── uploads/       # User uploaded files
│   └── processed/     # Processed datasets
├── logs/              # Application and security logs
├── main.py           # Application entry point
└── requirements.txt  # Python dependencies
```

## 🔧 Development

### Code Standards
- Follow PEP 8 with 88-character line limit
- Type hints required for all function signatures
- Google-style docstrings for all public functions/classes
- Minimum 80% test coverage

### Testing
```bash
pytest tests/ --cov=backend --cov-report=xml
```

### Code Quality
```bash
black .                    # Code formatting
isort .                   # Import sorting
flake8 . --max-line-length=88  # Linting
mypy . --strict           # Type checking
```

## 🔒 Security Features

- Input validation for all file uploads
- File type and size validation (max 100MB)
- Excel formula sanitization
- Authentication and authorization system
- Audit logging for security events
- No hardcoded credentials

## 📊 Usage

### Basic Data Analysis Workflow

1. **Upload Data**: Use the web interface or API to upload Excel/Access files
2. **Data Exploration**: Access interactive notebooks for EDA
3. **Model Building**: Create linear regression models with automated evaluation
4. **Visualization**: Generate insights with integrated plotting tools
5. **Export Results**: Download processed data and model predictions

### API Endpoints

- `POST /api/data/upload` - Upload data files
- `GET /api/data/summary` - Get dataset summary statistics
- `POST /api/model/train` - Train linear regression model
- `GET /api/model/predict` - Generate predictions
- `GET /api/model/evaluate` - Get model performance metrics

## 📝 Documentation

- [Requirements Specification](Requirements.md)
- [Implementation Plan](ImplementationPlan.md)
- [Backend Architecture](Backend.md)
- [Frontend Design](Frontend.md)
- [Code Review Checklist](CodeReviewChecklist.md)
- [Development Guidelines](CLAUDE.md)

## 🧪 Testing

The project includes comprehensive testing with:
- Unit tests for all public methods
- Integration tests for API endpoints
- Property-based testing for data processing
- Security validation tests
- Performance benchmarks

Run tests with coverage:
```bash
pytest tests/ --cov=backend --cov-report=html
```

## 📈 Performance

- Target processing time: <30 seconds for 1GB datasets
- Memory optimization for large file handling
- Efficient caching for repeated operations
- Scalable architecture for future enhancements

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure all tests pass and code follows the established standards.

## 📄 License

This project is for educational purposes. Please refer to the institution's guidelines for usage and distribution.

## 🐛 Issues and Support

For bug reports, feature requests, or questions, please open an issue on the GitHub repository.

## 🏗️ Roadmap

- [ ] Advanced ML model support (Random Forest, SVM)
- [ ] Real-time data streaming capabilities
- [ ] Enhanced visualization dashboard
- [ ] Mobile-responsive interface
- [ ] Cloud deployment options
- [ ] Advanced statistical analysis tools

---

**Built with**: Python, FastAPI, Pandas, Scikit-learn, Jupyter, and ❤️