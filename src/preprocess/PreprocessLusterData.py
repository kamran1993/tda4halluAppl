from dataclasses import dataclass, field
import os

from .preprocessLuster import PreprocessLuster

@dataclass
class LusterData(PreprocessLuster):
    """A class to process and manage one of the two luster datasets."""

    turns_dir_path: str = field(default_factory=lambda: os.environ.get('LUSTER_REPOSITORY_BASE_PATH')\
                     + "/data/training_logs/experiences_succ/analysis_outputs")