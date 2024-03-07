import flair

from pathlib import Path
from typing import List, Optional, Union

from flair.datasets import UniversalDependenciesCorpus
from flair.file_utils import cached_path


UD_TURKISH_MAPPING = {
    "UD_Turkish-Tourism":  "tr_tourism",
    "UD_Turkish-Penn":     "tr_penn",
    "UD_Turkish-Kenet":    "tr_kenet",
    "UD_Turkish-IMST":     "tr_imst",
    "UD_Turkish-FrameNet": "tr_framenet",
    "UD_Turkish-BOUN":     "tr_boun",
    "UD_Turkish-Atis":     "tr_atis",
}

UD_TURKISH_REVISION_MAPPING = {
    "UD_Turkish-Tourism":  "cf5aa18",
    "UD_Turkish-Penn":     "99a4683",
    "UD_Turkish-Kenet":    "6510b80",
    "UD_Turkish-IMST":     "7aaa466",
    "UD_Turkish-FrameNet": "0ab8f50",
    "UD_Turkish-BOUN":     "283d255",
    "UD_Turkish-Atis":     "b7d60ab",
}

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
