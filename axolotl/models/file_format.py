from enum import Enum

class FileFormat(str, Enum):
    CSV = "csv"
    JSON = "json"
    HF = "hf"
    S3 = "s3"