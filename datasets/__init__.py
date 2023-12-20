from .sapien import SapienDataset, SapienPartDataset, SapienStaticSegDataset, SapienArtSegDataset, SapienArtSegDataset_v2
from .sapien_multi import SapienDatasetMulti

dataset_dict = {"sapien": SapienDataset, "sapien_multi": SapienDatasetMulti, "sapien_part":SapienPartDataset, "sapien_static_seg":SapienStaticSegDataset,
"sapien_artseg": SapienArtSegDataset, "sapien_artseg_v2": SapienArtSegDataset_v2}
