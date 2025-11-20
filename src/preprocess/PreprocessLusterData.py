from dataclasses import dataclass

import typing as tp
import sys
import os
import numpy as np
import pandas as pd
from numpy import ndarray
import pickle
import pathlib
from sklearn.model_selection import train_test_split

sys.path.append(os.environ.get('LUSTER_REPOSITORY_BASE_PATH'))
from luster.process_training_logs.prepare_imports_for_pickle_loading import prepare_imports_for_pickle_loading
from luster.memory import PolicyHistoryItem
from src.preprocess import labelLusterData

from .dataset_abc import HallucinationDetectionDataset


@dataclass
class LusterData(HallucinationDetectionDataset):
    """A class to process and manage one of the two luster datasets."""

    model_name: tp.Literal["LUSTER"]
    split: str = "original"
    val_size: int | float = 100
    random_state: int = 42

    def split_data(self, df: pd.DataFrame) -> tuple[np.ndarray[int], np.ndarray[int]]:
        """Split."""
        indices = np.arange(len(df))
        if self.val_size == len(indices):
            return None, indices
        train_test_indices, val_indices = train_test_split(
            indices, test_size=self.val_size, random_state=self.random_state
        )
        return train_test_indices, val_indices


    def load_data(self) -> pd.DataFrame:
        """Load policy history items and convert them to dataframe."""
        prepare_imports_for_pickle_loading()

        turns_path = pathlib.Path(
            os.environ['LUSTER_REPOSITORY_BASE_PATH'],
            "data",
            "training_logs",
            "experiences_succ",
            "analysis_outputs",
            "turns.pkl",
        )

        with turns_path.open(mode="rb") as f:
            policy_histories_list = pickle.load(file=f)

        dataset_name: str = "multiwoz21"
        df = pd.DataFrame(columns=["id", "prompt", "response", "name"])
        ind = 0
        for turn in policy_histories_list:
            ind += 1
            utterance = turn.system_response
            prompt = turn.full_seq_with_top_1_response
            utterance = ' system : ' + utterance + '</s>'
            prompt = prompt.removesuffix(utterance)
            df.loc[ind - 1] = [ind, prompt, utterance, dataset_name]

        return df


    def process(self) -> tuple[pd.DataFrame, pd.Series, ndarray, ndarray | None]:
        if self.model_name != "LUSTER":
            raise NotImplementedError(
                f"This model is not supported yet: {self.model_name}"
            )
        df = self.load_data()
        labels = labelLusterData.load_data_and_create_hallu_labels(
            os.environ['LUSTER_REPOSITORY_BASE_PATH']
            + "/data/training_logs/experiences_succ/analysis_outputs/converted_luster_log.json"
        )
        train_indices, test_indices = self.split_data(df)
        return (
            pd.DataFrame(df[["id", "prompt", "response", "name"]]),
            labels.astype(int),
            train_indices,
            test_indices,
        )