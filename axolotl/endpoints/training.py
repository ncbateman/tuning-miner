import os
import subprocess
from loguru import logger
from fastapi import APIRouter, HTTPException

from utilities.dataset import download_dataset
from models.training_request import TrainingRequest
from models.file_format import FileFormat
from config.config_handler import create_config

router = APIRouter()

# Global variable to track the running training process
training_process = None

@router.post("/train")
async def start_training(request: TrainingRequest):
    global training_process

    # Check if a training process is already running
    if training_process and training_process.poll() is None:
        raise HTTPException(status_code=409, detail="Training already in progress. Please wait for the current job to finish.")

    try:
        config = await create_config(
            request.dataset_url, 
            request.dataset_type, 
            request.job_id,
            request.base_model
        )

        preprocessing_command = f"python -m axolotl.cli.preprocess {config} --dataset_prepared_path=/tmp/prepared-data-{request.job_id}"
        subprocess.run(preprocessing_command, shell=True, check=True)
        
        # Run the training step in the background and track the process
        training_command = f"accelerate launch -m axolotl.cli.train {config} --dataset_prepared_path=/tmp/prepared-data-{request.job_id}"
        training_process = subprocess.Popen(training_command, shell=True)
        
        return {"status": "success", "message": "Training started in the background"}
    
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Error during preprocessing: {str(e)}")
