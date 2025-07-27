from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, classification_report
from sklearn.pipeline import Pipeline
from src.logging_config import logger


class ModelEvaluator:

    def __init__(self, model: Pipeline) -> None:
        self.model = model

    def evaluate(self, xtrain, ytrain, xtest, ytest):
        logger.info("Model evaluation started")
        ypred_train = self.model.predict(xtrain)
