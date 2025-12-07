from dataclasses import dataclass

import typing as tp
import sys
import os
import re
import numpy as np
import pandas as pd
from numpy import ndarray
import pickle
import pathlib
from sklearn.model_selection import train_test_split
import importlib

sys.path.append(os.environ.get('LUSTER_REPOSITORY_BASE_PATH'))
from luster.process_training_logs.prepare_imports_for_pickle_loading import prepare_imports_for_pickle_loading
from luster.memory import PolicyHistoryItem
from luster.evaluation.convert_luster_pkl_to_unified import main as convertLusterPklToJson
from src.preprocess import labelLusterData2, labelLusterData

from .dataset_abc import HallucinationDetectionDataset


@dataclass
class PreprocessLuster(HallucinationDetectionDataset):
    """A class to process LUSTER pkl-datasets."""

    model_name: tp.Literal["LUSTER"]
    turns_dir_path: str
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

        turns_path = pathlib.Path(self.turns_dir_path, "turns.pkl")

        with turns_path.open(mode="rb") as f:
            policy_histories_list = pickle.load(file=f)

        dataset_name: str = "multiwoz21"
        df = pd.DataFrame(columns=["id", "prompt", "response", "name"])
        ind = 0
        for turn in policy_histories_list:
            utterance = turn.system_response
            prompt = turn.full_seq_with_top_1_response
            match = re.search(r"action :.*?</s>", prompt)
            if match:
                prompt = match.group()
            utterance = ' system : ' + utterance + '</s>'
            df.loc[ind] = [ind, prompt, utterance, dataset_name]
            ind += 1

        return df


    def process(self) -> tuple[pd.DataFrame, pd.Series, ndarray, ndarray | None]:
        if self.model_name != "LUSTER":
            raise NotImplementedError(
                f"This model is not supported yet: {self.model_name}"
            )
        df = self.load_data()
        converted_luster_log_path = self.turns_dir_path + "/converted_luster_log.json"
        if not os.path.isfile(converted_luster_log_path):
            convertLusterPklToJson(pathlib.Path(self.turns_dir_path + "/turns.pkl"))
        #labels1, allRedundancies1 = labelLusterData.load_data_and_create_hallu_labels(converted_luster_log_path)
        #labels2, allRedundancies2 = labelLusterData2.load_data_and_create_hallu_labels(converted_luster_log_path)
        #labels = labels1 & labels2
        labels = importlib.import_module("src.preprocess.get_manual_labels").labels
        train_indices, test_indices = self.split_data(df)
        #with open('out.txt', 'w') as f:
        #    for i in range(len(labels)):
        #         print(i, file=f)
        #         print(df["prompt"].iloc[i], file=f)
        #         print(df["response"].iloc[i], file=f)
        #         print(allRedundancies1[i], file=f)
        #         print(allRedundancies2[i], file=f)
        #         print("\n", file=f)
        return (
            pd.DataFrame(df[["id", "prompt", "response", "name"]]),
            labels.astype(int),
            train_indices,
            test_indices,
        )