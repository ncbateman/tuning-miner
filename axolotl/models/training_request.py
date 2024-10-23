from pydantic import BaseModel
from models.dataset_type import DatasetType
from models.custom_dataset_type import CustomDatasetType

class TrainingRequest(BaseModel):
    dataset_url: str
    base_model: str
    dataset_type: DatasetType | CustomDatasetType
    job_id: str