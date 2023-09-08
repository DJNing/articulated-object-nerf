from .sapien import SapienDataset, SapienPartDataset
from .sapien_multi import SapienDatasetMulti

dataset_dict = {"sapien": SapienDataset, "sapien_multi": SapienDatasetMulti, "sapien_part":SapienPartDataset}
