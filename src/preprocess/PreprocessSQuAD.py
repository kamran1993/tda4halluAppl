import typing as tp
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn.model_selection import train_test_split

from .dataset_abc import HallucinationDetectionDataset


@dataclass
class SQuAD(HallucinationDetectionDataset):
    """A class to process and manage SQuAD dataset."""

    model_name: tp.Literal["Mistral-7B-Instruct-v0.1", "Phi-3.5-mini-instruct", "LUSTER", "SC-GPT"]
    source_dir: str = "data/raw/SQuAD"
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

        return pd.read_csv(f"{self.source_dir}/squad_{self.model_name}.csv")

    def process(self) -> tuple[pd.DataFrame, pd.Series, ndarray, ndarray | None]:
        df = self.load_data()

        # add dataset name
        df["name"] = "squad"

        df.rename(columns={"generated_answer": "response"}, inplace=True)
        if self.model_name in [
            "Mistral-7B-Instruct-v0.1",
        ]:
            df["prompt"] = (
                "<s>[INST] "
                + df["prompt"]
                + "Context: "
                + df["context"]
                + "\nQuestion: "
                + df["question"]
                + "\nAnswer: [/INST]"
            )
            if self.model_name in ["Mistral-7B-Instruct-v0.1"]:
                df["id"] = df.index
            df["response"] = df["response"].apply(lambda x: f"{x}</s>")
        else:
            raise NotImplementedError(
                f"This model is not supported yet: {self.model_name}"
            )
        # logger.debug(df.columns)
        train_indices, test_indices = self.split_data(df)

        return (
            pd.DataFrame(df[["id", "prompt", "response", "name"]]),
            df["hallucination"].astype(int),
            train_indices,
            test_indices,
        )
