import pickle
import logging

import pandas as pd

from Utils.params import read_training_pipeline_params

import click


def predict_GB(config_path, logger):
    logger.info('Reading config')
    training_pipeline_params = read_training_pipeline_params(config_path)
    
    logger.info('Reading data')
    df = pd.read_csv(training_pipeline_params.input_data_path)
    if training_pipeline_params.feature_params.target_col in df.columns:
        df = df.drop(training_pipeline_params.feature_params.target_col, axis=1)

    logger.info('Loading model')
    with open(training_pipeline_params.model_path, 'rb') as file:
        model = pickle.load(file)

    logger.info('Predicting')
    y_pred = model.predict(df)
    
    logger.info('Saving predicts')
    pd.DataFrame(y_pred).to_csv(training_pipeline_params.result_path + "/Cat_GB_preds.csv")
    logging.info("Model saved: " + training_pipeline_params.result_path)


@click.command(name="train")
@click.argument("config_path")
def train_pipeline_command(config_path: str):
    predict_GB(config_path)

if __name__ == "__main__":
    train_pipeline_command()