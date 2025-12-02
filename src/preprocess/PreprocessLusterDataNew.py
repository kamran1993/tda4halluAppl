from dataclasses import dataclass, field
import os

from .preprocessLuster import PreprocessLuster

@dataclass
class LusterDataNew(PreprocessLuster):
    """A class to process and manage new luster dataset."""

    turns_dir_path: str = field(default_factory=lambda: os.environ.get('LUSTER_REPOSITORY_BASE_PATH')\
                     + "/eval_model_luster-full/seed_"
                                                        + str(4) +
                                "/num_dialogues_100")