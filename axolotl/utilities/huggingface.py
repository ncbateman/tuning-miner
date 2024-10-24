from huggingface_hub import HfApi
from loguru import logger

def make_repo_public(task_id: str, token: str):
    api = HfApi()
    repo_id = f"ncbateman/tuning-miner-testbed-{task_id}"

    try:
        api.update_repo_visibility(repo_id=repo_id, private=False, token=token)
        logger.info(f"Repository {repo_id} is now public.")
        
    except Exception:
        logger.info(f"Repository {repo_id} not found. Creating the repository.")
        
        try:
            api.create_repo(repo_id=repo_id, token=token, private=False)
            logger.info(f"Repository {repo_id} created successfully.")
        except Exception as e:
            logger.error(f"Failed to create repository {repo_id}: {e}")
            return
        
        try:
            api.update_repo_visibility(repo_id=repo_id, private=False, token=token)
            logger.info(f"Repository {repo_id} is now public.")
        except Exception as e:
            logger.error(f"Failed to make repository {repo_id} public: {e}")
    
    except Exception as e:
        logger.error(f"Failed to update repository visibility for {repo_id}: {e}")
