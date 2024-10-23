from pydantic import BaseModel

class MinerTaskRequst(BaseModel):
    ds_size: int
    model: str
    hours_to_complete: int