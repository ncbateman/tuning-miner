from pydantic import BaseModel

class MinerTaskRequest(BaseModel):
    ds_size: int
    model: str
    hours_to_complete: int
    task_id: str