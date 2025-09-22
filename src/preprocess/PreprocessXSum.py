import typing as tp
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn.model_selection import train_test_split

from .dataset_abc import HallucinationDetectionDataset


@dataclass
class XSum(HallucinationDetectionDataset):
    """A class to process and manage CoQA dataset."""

    model_name: tp.Literal["Mistral-7B-Instruct-v0.1", "Phi-3.5-mini-instruct", "LUSTER", "SC-GPT"]
    source_file: str = "data/raw/XSum/xsum_Mistral-7B-Instruct-v0.1.csv"
    split: str = "original"
    val_size: int | float = 100
    random_state: int = 42

    def split_data(self, df: pd.DataFrame) -> tuple[np.ndarray[int], np.ndarray[int]]:
        """Split."""
        indices = np.arange(len(df))  # Create an array of integer indices
        #self.val_size = len(indices)
        if self.val_size == len(indices):
            return None, indices
        train_test_indices, val_indices = train_test_split(
            indices, test_size=self.val_size, random_state=self.random_state
        )
        return train_test_indices, val_indices

    def load_data(self) -> pd.DataFrame:
        """Load csv with model hallucinations."""

        return pd.read_csv(self.source_file)

    def process(self) -> tuple[pd.DataFrame, pd.Series, ndarray, ndarray | None]:
        df = self.load_data()

        def insert_context_question(row):
            # Assuming 'prompt' is the template where you want to insert context and question
            new_prompt = row["prompt"].format(row["document"])
            return new_prompt

        # add dataset name
        df["name"] = "xsum"
        df.rename(columns={"generated_summary": "response"}, inplace=True)

        df["prompt"] = df.apply(insert_context_question, axis=1)
        if self.model_name in [
            "Mistral-7B-Instruct-v0.1",
        ]:
            df["prompt"] = df["prompt"].apply(lambda x: f"<s>[INST] {x} [/INST]")
            df["response"] = df["response"].apply(lambda x: f"{x} </s>")

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
