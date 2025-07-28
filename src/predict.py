import joblib
import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.pipeline import Pipeline
from src.constants import MODEL_PATH
from src.logging_config import logger


@st.cache_resource
def load_model(model_path: Path) -> Pipeline:
    logger.info(f"Loading model from : {model_path}")
    return joblib.load(model_path)


class IrisPredictor:

    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self.model_path = model_path
        self.model = load_model(self.model_path)

    def to_dataframe(
        self, sep_len: float, sep_wid: float, pet_len: float, pet_wid: float
    ) -> pd.DataFrame:
        logger.info("Converting to dataframe")
        data = [
            {
                "sepal_length": sep_len,
                "sepal_width": sep_wid,
                "petal_length": pet_len,
                "petal_width": pet_wid,
            }
        ]
        df = pd.DataFrame(data)
        logger.info(f"Converted to dataframe :\n{df}")
        return df

    def predict(self, x: pd.DataFrame):
        preds = self.model.predict(x)[0]
        logger.info(f"Prediction : {preds}")
        return preds

    def predict_proba(self, x: pd.DataFrame):
        probs = self.model.predict_proba(x)
        classes = self.model.classes_
        probs_df = pd.DataFrame(probs, columns=classes)
        logger.info(f"Predicted probabilities :\n{probs_df}")
        return probs_df.round(4)
