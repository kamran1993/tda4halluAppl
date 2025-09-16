import numpy as np
import pandas as pd
from loguru import logger
from scipy.interpolate import interp1d
from tabulate import tabulate


def interpolate_precision_recall(curves, mean_recall):
    """Interpolate and average precision-recall curves."""
    interpolated_precisions = [
        interp1d(recall, precision)(mean_recall) for precision, recall in curves
    ]
    return np.mean(interpolated_precisions, axis=0)


def process_metrics(metrics: list[dict[str, str | float]], experiment=None):
    """Process metrics without logging to Comet, just return formatted results."""
    
    df = pd.DataFrame(metrics)

    # TODO: fixthis
    # This code is bad! But consider it as temporary solution
    if len(metrics) == 1:
        # Aggregation and formatting of the metrics
        agg_df = df.groupby("dataset")[["roc_auc", "pr_auc", "f1"]].agg(["mean", "std"])
        formatted_df = agg_df.apply(
            lambda row: row.xs("mean", level=1).apply(lambda x: f"{x:.3f}")
            + " ± "
            + row.xs("std", level=1).apply(lambda x: f"{x:.3f}"),
            axis=1,
        ).loc[["test"], :]

        return tabulate(
            formatted_df, headers="keys", tablefmt="pipe", showindex=True
        ), agg_df

    # Mean recall for interpolation
    mean_recall = np.linspace(0, 1, 100)

    # Log dataset-specific metrics (just print instead of logging to Comet)
    for dataset in df["dataset"].unique():
        df_dataset = df[df["dataset"] == dataset]
        records = df_dataset.drop(columns=["precision", "recall", "dataset"]).to_dict("records")
        for record in records:
            metrics_str = ", ".join([f"{k}: {v:.3f}" for k, v in record.items() if k != "seed"])
            logger.debug(f"{dataset} metrics (seed {record.get('seed', 'N/A')}): {metrics_str}")

    # Aggregation and formatting of the metrics
    agg_df = df.groupby("dataset")[["roc_auc", "pr_auc", "f1"]].agg(["mean", "std"])

    formatted_df = agg_df.apply(
        lambda row: row.xs("mean", level=1).apply(lambda x: f"{x:.3f}")
        + " ± "
        + row.xs("std", level=1).apply(lambda x: f"{x:.3f}"),
        axis=1,
    ).loc[["train", "val", "test"], :]

    return tabulate(
        formatted_df, headers="keys", tablefmt="pipe", showindex=True
    ), agg_df
