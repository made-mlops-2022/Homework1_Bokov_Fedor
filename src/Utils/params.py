from dataclasses import dataclass, field
import yaml
from marshmallow_dataclass import class_schema
from typing import Dict, List, Optional


@dataclass()
class FeatureParams:
    categorical_features: List[str]
    numerical_features: List[str]
    target_col: Optional[str]


@dataclass()
class TrainingParams:
    model_type: str = field(default="RandomForestClassifier")
    random_state: int = field(default=42)
    params: Dict[str, float] = field(default=dict)

@dataclass()
class TrainingPipelineParams():
    input_data_path: str
    model_path: str
    metric_path: str
    train_params: TrainingParams
    feature_params: FeatureParams
    result_path: str = field(default="")


def read_training_pipeline_params(path: str) -> TrainingPipelineParams:
    TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)
    with open(path, "r") as input_stream:
        schema = TrainingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))


@dataclass()
class PredictPipelineParams:
    input_data_path: str
    output_model_path: str
    path_to_model: str
    feature_params: FeatureParams
    train_params: TrainingParams


PredictPipelineParamsSchema = class_schema(PredictPipelineParams)


def read_predict_pipeline_params(config_path: str) -> PredictPipelineParams:
    with open(config_path, "r") as config:
        schema = PredictPipelineParamsSchema()
        return schema.load(yaml.safe_load(config))

