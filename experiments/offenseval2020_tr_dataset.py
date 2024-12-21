import flair

from flair.datasets import ClassificationCorpus

from huggingface_hub import hf_hub_download

from pathlib import Path
from typing import Optional, Union

class OFFENSEVAL_TR_2020(ClassificationCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        **corpusargs,
    ) -> None:
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)
        dataset_name = self.__class__.__name__.lower()
        data_folder = base_path / dataset_name
        data_path = flair.cache_root / "datasets" / dataset_name

        for split in ["train", "dev", "test"]:
            hf_hub_download(repo_id="stefan-it/offenseval2020_tr", repo_type="dataset",
                            filename=f"{split}.txt", token=True, local_dir=data_folder)

        super().__init__(
            data_path,
            **corpusargs,
        )
