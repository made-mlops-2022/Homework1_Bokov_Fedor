from models.Grad_boosting.predict import predict_GB
from models.Random_forest.predict import predict_RF

import click

from Utils.params import read_training_pipeline_params

import logging

def predict(config_path):
    logging.basicConfig(filename='src/log/log.log', encoding='utf-8', level=logging.INFO)
    training_pipeline_params = read_training_pipeline_params(config_path)
    model_type = training_pipeline_params.train_params.model_type
    if model_type == "RandomForestClassifier":
        logger.info(f'Start predict with model {model_type}')
        predict_RF(config_path, logger)
    elif model_type == "CatBoostClassifier":
        logger.info(f'Start predict with model {model_type}')
        predict_GB(config_path, logger)
    else:
        raise NotImplementedError()

    logger.info('Predicting is finished')

@click.command(name="predict")
@click.argument("config_path")
def predict_pipeline_command(config_path: str):
    predict(config_path)

if __name__ =="__main__":
    logger = logging.getLogger("predict")
    predict_pipeline_command()