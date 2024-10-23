from pydantic import BaseModel

class MinerTaskResponse(BaseModel):
    message: str
    accepted: bool