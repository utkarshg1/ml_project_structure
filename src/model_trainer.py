import joblib
import pandas as pd
from pathlib import Path
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from src.logging_config import logger
from src.constants import MODEL_PATH, IMPUTE_STRAT, RANDOM_STATE


class ModelTrainer:

    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self.model_path = model_path
        self.model = self.create_pipeline()

    def create_pipeline(self) -> Pipeline:
        logger.info("Creating model pipeline")
        model = make_pipeline(
            SimpleImputer(strategy=IMPUTE_STRAT),
            StandardScaler(),
            LogisticRegression(random_state=RANDOM_STATE),
        )
        return model

    def train_model(self, xtrain: pd.DataFrame, ytrain: pd.Series):
        logger.info("Model training started")
        self.model.fit(xtrain, ytrain)
        logger.success("Model training successful")

    def save_model(self):
        logger.info(f"Model saving started")
        joblib.dump(self.model, self.model_path)
        logger.success(f"Model successfuly saved at : {self.model_path}")
