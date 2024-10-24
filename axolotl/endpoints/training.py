import os
import subprocess
import redis
import yaml
import asyncio
from loguru import logger
from fastapi import APIRouter, HTTPException

from models.training_request import TrainingRequest
from models.miner_task_request import MinerTaskRequest
from models.miner_task_response import MinerTaskResponse

from config.config_handler import create_config

from utilities.huggingface import make_repo_public

router = APIRouter()

r = redis.Redis(host='redis', port=6379, db=0)

TRAINING_FLAG_KEY = "flags:training"
TRAINING_TASK_ID_KEY = "flags:training_task_id"

logger.info(r.exists(TRAINING_FLAG_KEY))


def is_training():
    return r.exists(TRAINING_FLAG_KEY)

def set_training_flag(task_id: str):
    r.set(TRAINING_FLAG_KEY, "true", ex=900)
    r.set(TRAINING_TASK_ID_KEY, task_id, ex=900)

def clear_training_flag():
    r.delete(TRAINING_FLAG_KEY)
    r.delete(TRAINING_TASK_ID_KEY)

def get_training_task_id():
    return r.get(TRAINING_TASK_ID_KEY).decode('utf-8')

async def run_training_process(preprocessing_command: str, training_command: str, task_id: str, config):
    try:
        subprocess.run(preprocessing_command, shell=True, check=True)
        training_process = await asyncio.create_subprocess_shell(training_command)
        await training_process.wait()

    except subprocess.CalledProcessError as e:
        logger.error(f"Error during preprocessing or training for job {task_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during preprocessing or training: {str(e)}")
    
    finally:
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        
        if not hf_token:
            logger.error("HUGGINGFACE_TOKEN environment variable not set.")
            raise HTTPException(status_code=500, detail="Hugging Face token is missing.")
        
        try:
            make_repo_public(task_id, hf_token)
        except Exception as e:
            logger.error(f"Error making the repository public: {str(e)}")
        
        clear_training_flag()

@router.post("/start_training/")
async def start_training(request: TrainingRequest):
    if is_training():
        current_task_id = get_training_task_id()
        if request.task_id != current_task_id:
            raise HTTPException(status_code=409, detail="Another training task is currently in progress.")
        
    try:
        config = await create_config(
            request.dataset, 
            request.dataset_type, 
            request.task_id,
            request.model
        )

        preprocessing_command = f"python -m axolotl.cli.preprocess {config} --dataset_prepared_path=/tmp/prepared-data-{request.task_id}"
        training_command = f"accelerate launch -m axolotl.cli.train {config} --dataset_prepared_path=/tmp/prepared-data-{request.task_id}"

        asyncio.create_task(run_training_process(preprocessing_command, training_command, request.task_id, config))

        return {"status": "success", "message": "Training started in the background"}

    except Exception as e:
        clear_training_flag()
        logger.error(f"Error starting training: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting training: {str(e)}")

@router.post("/task_offer/")
async def task_offer(request: MinerTaskRequest) -> MinerTaskResponse:
    if is_training():
        return MinerTaskResponse(message="At capacity", accepted=False)
    else:
        set_training_flag(request.task_id)
        return MinerTaskResponse(message="Yes", accepted=True)

@router.get("/get_latest_model_submission/{task_id}")
async def get_latest_model_submission(task_id: str) -> str:
    try:
        config_filename = f"{task_id}.yaml"

        config_path = os.path.join('/workspace/config/', config_filename)
        with open(config_path, "r") as file:
            config_data = yaml.safe_load(file)
            return config_data.get("hub_model_id", None)

    except Exception as e:
        logger.error(f"Error retrieving latest model submission for task {task_id}: {str(e)}")
        raise HTTPException(status_code=404, detail=f"No model submission found for task {task_id}")
