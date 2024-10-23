from pydantic import BaseModel
from models.dataset_type import DatasetType
from models.custom_dataset_type import CustomDatasetType
from models.file_format import FileFormat

class TrainingRequest(BaseModel):
    dataset_url: str
    base_model: str
    dataset_type: DatasetType | CustomDatasetType
    file_format: FileFormat
    task_id: str
    hours_to_complete: int