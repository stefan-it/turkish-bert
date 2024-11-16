import flair

from pathlib import Path
from typing import List, Optional, Union

from flair.datasets import UniversalDependenciesCorpus
from flair.file_utils import cached_path


class UD_GENERIC(UniversalDependenciesCorpus):
    def __init__(
        self,
        ud_name: str,
        ud_dataset_prefix: str,
        dataset_splits: List[str] = ["train", "dev", "test"],
        revision: str = "master",
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        split_multiwords: bool = True,
    ) -> None:
        base_path = Path(flair.cache_root) / "datasets" if not base_path else Path(base_path)

        dataset_name = ud_name

        data_folder = base_path / dataset_name

        # download data if necessary
        web_path = f"https://raw.githubusercontent.com/UniversalDependencies/{ud_name}/{revision}"

        for dataset_split in dataset_splits:
            cached_path(f"{web_path}/{ud_dataset_prefix}-ud-{dataset_split}.conllu",
                        Path("datasets") / dataset_name)

        super().__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)
