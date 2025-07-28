# Iris Classification Project

A machine learning project for classifying Iris flower species using Logistic Regression with a Streamlit web interface.

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Details](#-model-details)
- [Configuration](#-configuration)
- [API Reference](#-api-reference)
- [Logging](#-logging)
- [Contributing](#-contributing)
- [Author](#-author)
- [License](#-license)

## 🌸 Overview

This project implements a machine learning pipeline to classify Iris flowers into three species (Setosa, Versicolor, Virginica) based on their sepal and petal measurements. The project includes data preprocessing, model training, evaluation, and a user-friendly Streamlit web application for making predictions.

## ✨ Features

- **Automated Data Pipeline**: Automatically downloads and processes Iris dataset
- **Machine Learning Pipeline**: Complete preprocessing with imputation and standardization
- **Model Training & Evaluation**: Comprehensive model evaluation with cross-validation
- **Web Interface**: Interactive Streamlit app for real-time predictions
- **Logging**: Comprehensive logging system using Loguru
- **Modular Design**: Well-structured, reusable code components

## 📁 Project Structure

```
iris-classification/
├── .gitignore            # Git ignore rules
├── .python-version       # Python version specifications
├── README.md            # Project documentation
├── app.py               # Streamlit web application
├── main.py              # Main training pipeline
├── template.py          # Project template/setup script
├── pyproject.toml       # Project configuration and dependencies (uv)
├── requirements.txt     # Dependencies list
├── uv.lock             # Dependency lock file (uv)
├── data/               # Data directory (auto-generated)
├── models/             # Trained models directory (auto-generated)
├── logs/               # Application logs (auto-generated)
└── src/
    ├── __init__.py
    ├── constants.py      # Project constants and configuration
    ├── data.py          # Data download functionality
    ├── logging_config.py # Logging configuration
    ├── model_evaluator.py # Model evaluation utilities
    ├── model_trainer.py  # Model training pipeline
    └── predict.py       # Prediction utilities
```

## 🚀 Installation

### Prerequisites

- Python 3.8+ (as specified in `.python-version`)
- [uv](https://docs.astral.sh/uv/) package manager (recommended)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd iris-classification
   ```

2. **Install uv** (if not already installed)
   ```bash
   # On macOS and Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # On Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

3. **Install dependencies using uv**
   ```bash
   uv sync
   ```

   **Alternative: Using pip**
   ```bash
   pip install -r requirements.txt
   ```

## 📖 Usage

### Training the Model

Run the complete training pipeline:

```bash
# Using uv
uv run python main.py

# Or activate the environment first
uv run --with-requirements requirements.txt python main.py
```

This will:
- Download the Iris dataset
- Preprocess the data (handle duplicates, split features/target)
- Train a Logistic Regression model with preprocessing pipeline
- Evaluate the model performance
- Save the trained model

### Running the Web Application

Launch the Streamlit app:

```bash
# Using uv
uv run streamlit run app.py

# Traditional method
streamlit run app.py
```

Then open your browser to `http://localhost:8501` and:
1. Enter the sepal length, width, petal length, and width measurements
2. Click "Predict" to get the species classification and prediction probabilities

### Example Usage

```python
from src.predict import IrisPredictor

# Load the trained model
predictor = IrisPredictor()

# Make a prediction
prediction = predictor.predict(predictor.to_dataframe(5.1, 3.5, 1.4, 0.2))
probabilities = predictor.predict_proba(predictor.to_dataframe(5.1, 3.5, 1.4, 0.2))

print(f"Predicted species: {prediction}")
print(f"Prediction probabilities: {probabilities}")
```

## 🤖 Model Details

### Algorithm
- **Model**: Logistic Regression
- **Preprocessing Pipeline**:
  - Simple Imputer (median strategy)
  - Standard Scaler for feature normalization

### Performance Metrics
The model is evaluated using:
- F1-score (macro average)
- Classification report
- 5-fold cross-validation
- Training and testing performance comparison

### Dataset
- **Source**: [Iris Dataset](https://raw.githubusercontent.com/utkarshg1/iris_data/refs/heads/main/iris.csv)
- **Features**: 4 numerical features (sepal_length, sepal_width, petal_length, petal_width)
- **Target**: 3 classes (setosa, versicolor, virginica)
- **Size**: ~150 samples

## 🔧 Configuration

### Project Configuration (`pyproject.toml`)
This project uses `pyproject.toml` for modern Python packaging and dependency management with uv.

### Application Configuration (`src/constants.py`)
Key configuration parameters:

```python
URL = "https://raw.githubusercontent.com/utkarshg1/iris_data/refs/heads/main/iris.csv"
DATA_PATH = Path("data", "iris.csv")
MODEL_PATH = Path("models", "iris_model.joblib")
TARGET = "species"
IMPUTE_STRAT = "median"
TEST_SIZE = 0.33
RANDOM_STATE = 21
```

### Dependencies
- All dependencies are managed through `uv.lock` for reproducible builds
- `requirements.txt` is also available for traditional pip installations

## 📊 API Reference

### IrisPredictor Class

```python
class IrisPredictor:
    def __init__(self, model_path: Path = MODEL_PATH)
    def to_dataframe(self, sep_len: float, sep_wid: float, pet_len: float, pet_wid: float) -> pd.DataFrame
    def predict(self, x: pd.DataFrame) -> str
    def predict_proba(self, x: pd.DataFrame) -> pd.DataFrame
```

### ModelTrainer Class

```python
class ModelTrainer:
    def __init__(self, model_path: Path = MODEL_PATH)
    def create_pipeline(self) -> Pipeline
    def train_model(self, xtrain: pd.DataFrame, ytrain: pd.Series)
    def save_model(self)
```

### ModelEvaluator Class

```python
class ModelEvaluator:
    def __init__(self, model: Pipeline)
    def evaluate(self, xtrain, ytrain, xtest, ytest)
```

## 📝 Logging

The project uses Loguru for comprehensive logging:
- **Console Output**: Colored, formatted logs for development
- **File Output**: Rotating log files in `logs/app.log`
- **Log Rotation**: 10MB rotation with 7-day retention
- **Compression**: Automatic ZIP compression of old logs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Update dependencies if needed:
   ```bash
   uv add <package-name>  # Add new dependency
   uv sync                # Sync dependencies
   ```
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Workflow with uv

```bash
# Add development dependencies
uv add --dev pytest black flake8

# Run scripts
uv run python main.py
uv run streamlit run app.py

# Update dependencies
uv sync --upgrade
```

## 👨‍💻 Author

**Utkarsh Gaikwad**

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

**Note**: This project uses [uv](https://docs.astral.sh/uv/) for fast and reliable Python package management. Make sure to run `uv run python main.py` first to train and save the model before using the Streamlit application.