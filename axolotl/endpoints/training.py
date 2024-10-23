import os
import subprocess
import redis
from loguru import logger
from fastapi import APIRouter, HTTPException

from utilities.dataset import download_dataset

from models.training_request import TrainingRequest
from models.miner_task_request import MinerTaskRequest
from models.miner_task_response import MinerTaskResponse
from models.file_format import FileFormat

from config.config_handler import create_config

router = APIRouter()

r = redis.Redis(host='redis', port=6379, db=0)

TRAINING_FLAG_KEY = "is_training"

def is_training():
    return r.exists(TRAINING_FLAG_KEY)

def set_training_flag():
    r.set(TRAINING_FLAG_KEY, "true")

def clear_training_flag():
    r.delete(TRAINING_FLAG_KEY)

@router.post("/train")
async def start_training(request: TrainingRequest):
    
    if is_training():
        raise HTTPException(status_code=409, detail="Training already in progress. Please wait for the current job to finish.")

    set_training_flag()

    config = await create_config(
        request.dataset_url, 
        request.dataset_type, 
        request.job_id,
        request.base_model
    )

    preprocessing_command = f"python -m axolotl.cli.preprocess {config} --dataset_prepared_path=/tmp/prepared-data-{request.job_id}"
    subprocess.run(preprocessing_command, shell=True, check=True)
    
    training_command = f"accelerate launch -m axolotl.cli.train {config} --dataset_prepared_path=/tmp/prepared-data-{request.job_id}"
    training_process = subprocess.Popen(training_command, shell=True)

    training_process.wait()
    clear_training_flag()
    
    return {"status": "success", "message": "Training started in the background"}

@router.post("/task_offer")
async def task_offer(request: MinerTaskRequest) -> MinerTaskResponse:
    if is_training():
        return MinerTaskResponse(message="At capacity", accepted=False)
    else:
        return MinerTaskResponse(message="Yes", accepted=True)

@router.get("/get_latest_model_submission/{task_id}")
async def get_latest_model_submission(task_id: str) -> str:
    try:
        config_filename = f"{task_id}.yml"
        config_path = os.path.join(cst.CONFIG_DIR, config_filename)
        with open(config_path, "r") as file:
            config_data = yaml.safe_load(file)
            return config_data.get("hub_model_id", None)

    except Exception as e:
        logger.error(f"Error retrieving latest model submission for task {task_id}: {str(e)}")
        raise HTTPException(status_code=404, detail=f"No model submission found for task {task_id}")

router.add_api_route(
    "/start_training/",
    start_training,
    tags=["Subnet"],
    methods=["POST"],
)

router.add_api_route(
    "/get_latest_model_submission/{task_id}",
    get_latest_model_submission,
    tags=["Subnet"],
    methods=["GET"],
    response_model=str,
    summary="Get Latest Model Submission",
    description="Retrieve the latest model submission for a given task ID",
)