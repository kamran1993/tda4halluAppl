from dataclasses import dataclass

import typing as tp
import sys
import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn.model_selection import train_test_split
sys.path.append('/gpfs/project/kaali100/models/LUSTER')
from luster.process_training_logs import load_create_dataframe

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

    def process(self) -> tuple[pd.DataFrame, pd.Series, ndarray, ndarray | None]:
        df = load_create_dataframe.createMultiwoz21DataFrame()

        prompt_for_nlg = "Create one response in natural language " \
                         "from this dialogue act. Create nothing else. "

        def insert_context_question(row):
            new_prompt = f"<s><|user|>{prompt_for_nlg}{row['prompt']}<|end|>"
            return new_prompt

        df["id"] = df.index

        if self.model_name == "LUSTER":
            df["prompt"] = df.apply(insert_context_question, axis=1)
            df["response"] = df["response"].apply(lambda x: f"<|assistant|>{x}<|end|><|endoftext|>")
        else:
            raise NotImplementedError(
                f"This model is not supported yet: {self.model_name}"
            )
        train_indices, test_indices = self.split_data(df)
        return (
            pd.DataFrame(df[["id", "prompt", "response", "name"]]),
            df["hallucination"].astype(int),
            train_indices,
            test_indices,
        )