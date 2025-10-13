import typing as tp

import numpy as np
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split

from ..methods.hallucination_detection_abc import HallucinationDetectionMethod, T

# TODO: fix all docstrings

def evaluate(
    model: HallucinationDetectionMethod,
    X: tp.Sequence[T],
    y: list[int],
    k: int = 5,
    seed: int = 42,
    test_size: float = 0.25,
    val_size: float = 0.15,
    pretrained: bool = False,
) -> list[dict[str, str | float]]:
    """Evaluate the performance of a given model.

    Args:
        model (object): The model to be evaluated, with `fit`, `predict`, and `predict_score` methods.
        X (list): All features.
        y (list): All labels.
        test_indices (list) : indices of the test set. If None, the KFoldCV is performed.
        k (int): The number of splits for cross-validation or bootstrap.
        seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: A dataframe containing performance metrics (ROC AUC, F1 score, PR AUC) for each dataset
                      (train, validation, test) and each split.

    """
    result = []
    # Iterate over splits and log results
    idxs = np.arange(len(X))
    for random_state in range(seed, seed + k):
        train_val_idxs, test_idxs = train_test_split(
            idxs, test_size=test_size, random_state=random_state
        )
        val_numel = int(len(train_val_idxs) * val_size)
        train_idxs, val_idxs = train_val_idxs[:-val_numel], train_val_idxs[-val_numel:]

        X_train, X_val, X_test = (
            [X[i] for i in train_idxs],
            [X[i] for i in val_idxs],
            [X[i] for i in test_idxs],
        )
        y_train, y_val, y_test = (
            [y[i] for i in train_idxs],
            [y[i] for i in val_idxs],
            [y[i] for i in test_idxs],
        )

        # Fit the model
        if not pretrained:
            model.fit(X_train, y_train)
            model.fit_threshold(X_val, y_val) # edited

        # Evaluate on train, val, and test sets
        for dataset, X_split, y_split in zip(
            ["train", "val", "test"], [X_train, X_val, X_test], [y_train, y_val, y_test]
        ):
            # Predict probabilities and classes
            y_pred_score = model.predict_score(X_split)
            y_pred = model.predict(X_split)
            # Calculate metrics
            roc_auc = roc_auc_score(y_split, y_pred_score)
            f1 = f1_score(y_split, y_pred)

            # Generate precision-recall curve
            precision, recall, _ = precision_recall_curve(y_split, y_pred_score)

            metrics_current = {
                "dataset": dataset,
                "roc_auc": roc_auc,
                "f1": f1,
                "pr_auc": auc(recall, precision),
                "hallu_prop": sum(y_split)/len(y_split),
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "seed": random_state,
            }

            result.append(metrics_current)

    return result
