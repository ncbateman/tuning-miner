import os
import subprocess
from loguru import logger
from fastapi import APIRouter
from fastapi import HTTPException

from utilities.dataset import download_dataset

from models.training_request import TrainingRequest

from models.file_format import FileFormat

from config.config_handler import create_config

router = APIRouter()

@router.post("/train")

async def start_training(request: TrainingRequest):

    config = await create_config(
        request.dataset_url, 
        request.dataset_type, 
        request.job_id,
        request.base_model
    )

    try:
        # preprocessing_command = f"python -m axolotl.cli.preprocess {config} --debug"
        # subprocess.run(preprocessing_command, shell=True, check=True)
        training_command = f"accelerate launch -m axolotl.cli.train {config}"
        subprocess.run(training_command, shell=True, check=True)
        return {"status": "success", "message": "Training started successfully"}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Error during training: {str(e)}")
