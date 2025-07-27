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
        ypred_test = self.model.predict(xtest)
        f1_train = f1_score(ytrain, ypred_train, average="macro")
        f1_test = f1_score(ytest, ypred_test, average="macro")
        logger.info(f"Training F1 Macro score : {f1_train:.4f}")
        logger.info(f"Testing F1 Macro score : {f1_test:.4f}")

        report = classification_report(ytest, ypred_test)
        logger.info(f"Classification Report on test :\n{report}")

        scores = cross_val_score(self.model, xtrain, ytrain, cv=5, scoring="f1_macro")
        f1_cv = scores.mean()
        f1_std = scores.std()
        logger.info(f"5 fold F1 macro cross validated : {f1_cv:.4f} +/- {f1_std:.4f}")
        logger.success("Model evaluation successful")
