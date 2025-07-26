from pathlib import Path

# Download file from url
URL = "https://raw.githubusercontent.com/utkarshg1/iris_data/refs/heads/main/iris.csv"

# Data path
DATA_PATH = Path("data", "iris.csv")

# Model path
MODEL_PATH = Path("models", "iris_model.joblib")

# Target feature
TARGET = "species"

# Test size
TEST_SIZE = 0.33
RANDOM_STATE = 42