from src.models.Grad_boosting.train import train_GB
from src.models.Random_forest.train import train_RF

import logging

def test_GB_train():
    logger = logging.getLogger('simple_example')
    train_GB("conf/conf_train_GB.yaml", logger)

def test_RF_train():
    logger = logging.getLogger('simple_example')
    train_RF("conf/test_conf_RF.yaml", logger)

def test_GB_predict():
    logger = logging.getLogger('simple_example')
    train_GB("conf/conf_predict_GB.yaml", logger)

def test_RF_predict():
    logger = logging.getLogger('simple_example')
    train_RF("conf/conf_predict_GB.yaml", logger)