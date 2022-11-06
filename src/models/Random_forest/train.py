from sklearn.ensemble import RandomForestClassifier
import pandas as pd

import pickle
import click

from Utils.params import read_training_pipeline_params


def train_RF(config_path, logger):
    logger.info('Reading config')
    training_pipeline_params = read_training_pipeline_params(config_path)
    logger.info('Reading data')
    df = pd.read_csv(training_pipeline_params.input_data_path)
    target_col = training_pipeline_params.feature_params.target_col
    logger.info('Training model')
    model = RandomForestClassifier().fit(df.drop(target_col, axis=1), df[target_col])
    logger.info('Training model is finished')
    logger.info('Saving model')
    with open(training_pipeline_params.model_path, "wb") as f:
        pickle.dump(model, f)
        logger.info("Model saved: " + training_pipeline_params.model_path)
    


@click.command(name="train")
@click.argument("config_path")
def train_pipeline_command(config_path: str):
    train_RF(config_path)


if __name__ == "__main__":
    train_pipeline_command()
