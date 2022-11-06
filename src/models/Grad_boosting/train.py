import pickle
import logging

from catboost import CatBoostClassifier
import pandas as pd

from Utils.params import read_training_pipeline_params

import click


def train_GB(config_path, logger = None):
    logger.info('Reading config')
    training_pipeline_params = read_training_pipeline_params(config_path)
    logger.info('Reading data')
    params = training_pipeline_params.train_params.params
    df = pd.read_csv(training_pipeline_params.input_data_path)
    target_col = training_pipeline_params.feature_params.target_col
    logger.info('Training model')
    model = CatBoostClassifier(**params).fit(df.drop(target_col, axis=1), df[target_col])
    logger.info('Training model is finished')
    logger.info('Saving model')
    with open(training_pipeline_params.model_path, "wb") as f:
        pickle.dump(model, f)
        logging.info("Model saved: " + training_pipeline_params.model_path)



@click.command(name="train")
@click.argument("config_path")
def train_pipeline_command(config_path: str):
    train_GB(config_path)

if __name__ == "__main__":
    train_pipeline_command()