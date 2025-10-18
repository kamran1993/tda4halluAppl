import typing as tp
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn.model_selection import train_test_split

from .dataset_abc import HallucinationDetectionDataset


@dataclass
class Altered(HallucinationDetectionDataset):
    """A class to process and manage altered dataset."""

    model_name: tp.Literal["Mistral-7B-Instruct-v0.1", "Phi-3.5-mini-instruct", "LUSTER", "SC-GPT"]
    source_file: str = "data/raw/altered/altered.jsonl"
    split: str = "original"
    val_size: int | float = 100
    random_state: int = 42
    sample_size: int | float = 1000
    part_hallu: float = 0.3

    def sample_and_stratify(self, df: pd.DataFrame) -> pd.DataFrame:
        # the whole dataset is too big and has too many hallucinated samples, therefore we sample it down and stratify it
        df = df.groupby("hallucination").sample(n=self.sample_size, random_state=self.random_state)
        n_hallu_samples = round(self.part_hallu * self.sample_size)
        n_grounded_samples = self.sample_size - n_hallu_samples
        df = pd.concat([df.iloc[0:n_grounded_samples,:], df.iloc[self.sample_size:(self.sample_size+n_hallu_samples),:]])
        df = df.sample(frac=1, random_state=self.random_state) # shuffle
        return df

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
        """Load jsonl with model hallucinations."""

        return pd.read_json(self.source_file, lines=True)

    def process(self) -> tuple[pd.DataFrame, pd.Series, ndarray, ndarray | None]:
        df = self.load_data()

        prompt_for_nlg = "Create one response in natural language " \
                         "from this dialogue act. Create nothing else. "

        def insert_context_question_mistral(row):
            new_prompt = f"<s>[INST]{prompt_for_nlg}{row['prompt']}[/INST]"
            return new_prompt

        def insert_context_question(row):
            new_prompt = f"<s><|user|>{prompt_for_nlg}{row['prompt']}<|end|>"
            return new_prompt

        df.rename(columns={"condition": "name"}, inplace=True)
        df.rename(columns={"alteration_meta": "hallucination"}, inplace=True)
        df.rename(columns={"utterance": "response"}, inplace=True)

        df["id"] = df.index
        # Non-empty alteration_meta means that the utterance is "hallucinated"
        df["hallucination"] = df["hallucination"] != {'field_drops': [], 'injected_noise': [], 'fallback_kept': []}
        df = self.sample_and_stratify(df)

        if self.model_name =="Mistral-7B-Instruct-v0.1":
            df["prompt"] = df.apply(insert_context_question_mistral, axis=1)
            df["response"] = df["response"].apply(lambda x: f"{x}</s>")
        elif self.model_name in ["Phi-3.5-mini-instruct", "LUSTER", "SC-GPT"]:
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