import pandas as pd
from sklearn.model_selection import train_test_split
from src.logging_config import logger
from src.constants import DATA_PATH, TARGET, TEST_SIZE, RANDOM_STATE
from src.data import download_data
from src.model_trainer import ModelTrainer


def main():
    logger.info("Data Ingestion started")
    download_data()
    df = pd.read_csv(DATA_PATH)
    logger.success(f"Dataframe loaded with shape : {df.shape}")

    logger.info(f"Checking duplicates : {df.duplicated().sum()}")
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    logger.success(f"Dropped duplicates new shape : {df.shape}")

    logger.info("Seperating X and Y")
    X = df.drop(columns=[TARGET])
    Y = df[TARGET]
    logger.success("X and Y seperation complete")

    logger.info("Applying train test split")
    xtrain, xtest, ytrain, ytest = train_test_split(
        X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    logger.success(f"xtrain shape : {xtrain.shape}, ytrain shape : {ytrain.shape}")
    logger.success(f"xtest shape : {xtest.shape}, ytest shape : {ytest.shape}")

    trainer = ModelTrainer()
    trainer.train_model(xtrain, ytrain)
    trainer.save_model()


if __name__ == "__main__":
    main()
