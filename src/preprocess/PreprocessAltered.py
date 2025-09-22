import typing as tp
from dataclasses import dataclass

import numpy as np
import pandas as pd
import json
from numpy import ndarray
from sklearn.model_selection import train_test_split

from .dataset_abc import HallucinationDetectionDataset


@dataclass
class Altered(HallucinationDetectionDataset):
    """A class to process and manage CoQA dataset."""

    model_name: tp.Literal["Mistral-7B-Instruct-v0.1", "Phi-3.5-mini-instruct", "LUSTER", "SC-GPT"]
    source_file: str = "data/raw/altered/altered.jsonl"
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
        """Load jsonl with model hallucinations."""

        return pd.read_json(self.source_file, lines=True)

    def process(self) -> tuple[pd.DataFrame, pd.Series, ndarray, ndarray | None]:
        df = self.load_data()

        def insert_context_question(row):
            prompt_for_nlg = "Create one response in natural language " \
                             "from this dialogue act. Create nothing else. "
            new_prompt = "<s>[INST] " + prompt_for_nlg + json.dumps(row["prompt"]) + "[/INST]"
            return new_prompt

        # add dataset name
        df["name"] = "altered"
        df = df.drop("prompt", axis=1)
        df.rename(columns={"dialogue_acts": "prompt"}, inplace=True)
        df.rename(columns={"alteration_meta": "hallucination"}, inplace=True)
        df.rename(columns={"utterance": "response"}, inplace=True)

        if self.model_name in [
            "Mistral-7B-Instruct-v0.1",
        ]:
            df["prompt"] = df.apply(insert_context_question, axis=1)
            # Non-empty alteration_meta means that the utterance is "hallucinated"
            df["hallucination"] = df["hallucination"] != {'field_drops': [], 'injected_noise': [], 'fallback_kept': []}
            df["id"] = df.index
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