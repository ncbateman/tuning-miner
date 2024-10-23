import os

import yaml

from models.dataset_type import DatasetType
from models.custom_dataset_type import CustomDatasetType
from models.file_format import FileFormat

from loguru import logger

from utilities.dataset import download_dataset

import yaml
import os

async def create_config(dataset_url, dataset_type, job_id, base_model):

    dataset_path = await download_dataset(dataset_url, job_id)

    config_template_path = os.getenv("CONFIG_TEMPLATE")

    if not config_template_path:
        raise Exception("CONFIG_TEMPLATE environment variable is not set")

    with open(config_template_path, "r") as file:
        config = yaml.safe_load(file)

    dataset_entry = create_dataset_entry(dataset_path, dataset_type, FileFormat.JSON)
    config["datasets"] = []
    config["datasets"].append(dataset_entry)

    config["base_model"] = base_model

    config["wandb_runid"] = job_id

    config["mlflow_experiment_name"] = dataset_url

    dataset_entry["ds_type"] = FileFormat.JSON.value

    dataset_entry["data_files"] = [dataset_path]

    config_file_path = save_config(config, f"/workspace/config/{job_id}.yaml")

    return config_file_path

def create_dataset_entry(
    dataset: str,
    dataset_type: DatasetType | CustomDatasetType,
    file_format: FileFormat,
) -> dict:

    dataset_entry = {"path": dataset}

    if isinstance(dataset_type, DatasetType):
        dataset_entry["type"] = dataset_type.value
    elif isinstance(dataset_type, CustomDatasetType):
        custom_type_dict = {key: value for key, value in dataset_type.model_dump().items() if value is not None}
        dataset_entry["type"] = custom_type_dict
    else:
        raise ValueError("Invalid dataset_type provided.")

    if file_format != FileFormat.HF:
        dataset_entry["ds_type"] = file_format.value
        dataset_entry["data_files"] = [os.path.basename(dataset)]

    return dataset_entry

def save_config(config: dict, config_path: str):
    with open(config_path, "w") as file:
        yaml.dump(config, file)
    return config_path
