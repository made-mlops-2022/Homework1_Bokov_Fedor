from models.Grad_boosting.train import train_GB
from models.Random_forest.train import train_RF

import click

from Utils.params import read_training_pipeline_params

import logging

def train(config_path):
    logging.basicConfig(filename='src/log/log.log', encoding='utf-8', level=logging.INFO)
    training_pipeline_params = read_training_pipeline_params(config_path)
    model_type = training_pipeline_params.train_params.model_type
    if model_type == "RandomForestClassifier":
        logger.info(f'Start train with model {model_type}')
        train_RF(config_path, logger)
    elif model_type == "CatBoostClassifier":
        logger.info(f'Start train with model {model_type}')
        train_GB(config_path, logger)
    else:
        raise NotImplementedError()

    logger.info('Model training was finished')

@click.command(name="train")
@click.argument("config_path")
def train_pipeline_command(config_path: str):
    train(config_path)

if __name__ =="__main__":
    logger = logging.getLogger("train_model")
    train_pipeline_command()