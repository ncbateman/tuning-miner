from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

def make_repo_public(task_id: str, token: str):

    api = HfApi()

    try:
        api.update_repo_visibility(repo_id=f"ncbateman/tuning-miner-testbed-{task_id}", private=False, token=token)
    except RepositoryNotFoundError:
        print(f"Repository ncbateman/tuning-miner-testbed-{task_id} not found.")
    except HfHubHTTPError as e:
        print(f"Failed to make repository public: {e}")