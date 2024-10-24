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

def set_training_flag(task_id: str, hours_to_complete: int):
    timeout_seconds = hours_to_complete * 3600
    r.set(TRAINING_FLAG_KEY, "true", ex=timeout_seconds)
    r.set(TRAINING_TASK_ID_KEY, task_id, ex=timeout_seconds)

def clear_training_flag():
    r.delete(TRAINING_FLAG_KEY)
    r.delete(TRAINING_TASK_ID_KEY)

def get_training_task_id():
    return r.get(TRAINING_TASK_ID_KEY).decode('utf-8')

async def run_process_with_timeout(command: str, timeout: int, task_id: str, stage: str):
    """
    Runs a subprocess with a timeout. If the timeout is exceeded, the process is terminated.
    """
    try:
        process = await asyncio.create_subprocess_shell(command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)

        try:
            await asyncio.wait_for(process.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.error(f"{stage.capitalize()} for task {task_id} exceeded the time limit and was terminated.")
            process.terminate()  # Terminate the process if it exceeds the timeout
            raise HTTPException(status_code=408, detail=f"{stage.capitalize()} process exceeded the time limit and was terminated.")

        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            logger.error(f"Error during {stage} for job {task_id}: {stderr.decode()}")
            raise HTTPException(status_code=500, detail=f"Error during {stage}: {stderr.decode()}")

        return stdout, stderr

    except subprocess.CalledProcessError as e:
        logger.error(f"Error during {stage} for job {task_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during {stage}: {str(e)}")


async def run_training_task(preprocessing_command: str, training_command: str, task_id: str, config, timeout_seconds: int):
    try:
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if not hf_token:
            logger.error("HUGGINGFACE_TOKEN environment variable not set.")
            raise HTTPException(status_code=500, detail="Hugging Face token is missing.")
        
        make_repo_public(task_id, hf_token)

        await run_process_with_timeout(preprocessing_command, timeout_seconds, task_id, "preprocessing")

        await run_process_with_timeout(training_command, timeout_seconds, task_id, "training")

    except HTTPException as e:
        logger.error(f"Training for task {task_id} failed: {str(e)}")
        raise e

    finally:        
        try:
            make_repo_public(task_id, hf_token)
        except Exception as e:
            logger.error(f"Error making the repository public: {str(e)}")
        
        clear_training_flag()


@router.post("/start_training/")
async def start_training(request: TrainingRequest):
    logger.info(f"\n\n\n\n{request}\n\n\n\n")
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

        # Set the training flag with timeout
        set_training_flag(request.task_id, request.hours_to_complete)
        timeout_seconds = request.hours_to_complete * 3600

        # Start the entire training task (preprocessing + training) in the background
        asyncio.create_task(run_training_task(preprocessing_command, training_command, request.task_id, config, timeout_seconds))

        return {"status": "success", "message": "Training started in the background"}

    except Exception as e:
        clear_training_flag()
        logger.error(f"Error starting training: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting training: {str(e)}")


@router.post("/task_offer/")
async def task_offer(request: MinerTaskRequest) -> MinerTaskResponse:

    logger.info(f"\n\n\n\n{request}\n\n\n\n")
    if is_training():
        return MinerTaskResponse(message="At capacity", accepted=False)
    else:
        set_training_flag(request.task_id, request.hours_to_complete)
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
    
@router.get("/current_training_task/")
async def current_training_task():
    if not is_training():
        raise HTTPException(status_code=404, detail="No training task is currently in progress.")
    
    try:
        task_id = get_training_task_id()
        return {"current_task_id": task_id}
    except Exception as e:
        logger.error(f"Error retrieving the current training task ID: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving the current training task ID: {str(e)}")
