import os
import typing as tp
from pathlib import Path

import hydra
import yaml
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf

from src.evaluation import evaluate
from src.evaluation.process_metrics import process_metrics
from src.methods.hallucination_detection_abc import HallucinationDetectionMethod, T
from src.preprocess.dataset_abc import HallucinationDetectionDataset

load_dotenv()

@hydra.main(version_base=None, config_path="config", config_name="master")
def main(cfg: OmegaConf):
    hydra_cfg = HydraConfig.get()
    preprocess_name: str = hydra_cfg.runtime.choices["preprocess"]
    method_name: str = hydra_cfg.runtime.choices["method"]
    model_name: str = cfg["model_name"]
    transfer_names: list[str] = cfg["transfer_names"]
    experiment_name = f"{preprocess_name}_{method_name}_{model_name}"
    
    print(experiment_name)
    logger.info(f"Starting experiment: {experiment_name}")
    logger.info(f"Configuration: {OmegaConf.to_container(cfg, resolve=True)}")

    dataset: HallucinationDetectionDataset = instantiate(cfg["preprocess"])
    X, y, train_test_indices, val_indices = dataset.process()

    model: HallucinationDetectionMethod = instantiate(cfg["method"], _convert_="all")
    if method_name == "mtopdiv" and model.analysis_sites == "all":
        model.select_heads(X.iloc[val_indices], y.iloc[val_indices])

    if method_name == "llm_check":
        model.select_layer(X.iloc[val_indices], y.iloc[val_indices])

    if method_name == "haloscope":
        X_transformed: tp.Sequence[T] = model.transform(X)
        model.fit([X_transformed[i] for i in train_test_indices], y.iloc[train_test_indices])
        model.fit_threshold([X_transformed[i] for i in train_test_indices], y.iloc[train_test_indices])
        metrics = evaluate(
            model,
            [X_transformed[i] for i in val_indices],
            y.iloc[val_indices].values,
            pretrained=True,
            **cfg["evaluation"],
        )
    else:
        X_transformed: tp.Sequence[T] = model.transform(X)
        X_transformed = [X_transformed[i] for i in train_test_indices]
        y = y.iloc[train_test_indices]

        metrics = evaluate(
            model,
            X_transformed,
            y.values,
            **cfg["evaluation"],
        )

    table_str, raw_table = process_metrics(metrics, None)  # Pass None instead of experiment
    logger.success(f"Results for cross validation on {dataset.__class__.__name__}")
    logger.info(f"Final test AUROC: {raw_table['roc_auc'].loc['test']['mean']:.3f}")
    print("Results for cross validation on {dataset.__class__.__name__}")
    print(f"Final test AUROC: {raw_table['roc_auc'].loc['test']['mean']:.3f}")
    print(table_str)

    logger.info("Transfering model on another dataset")
    print("Transfering model on another dataset")
    for transfer_name in transfer_names:
        with open(f"config/transfer/{transfer_name}.yaml") as f:
            transfer_cfg = yaml.load(f, Loader=yaml.FullLoader)
        transfer_cfg["model_name"] = model_name
        transfer_dataset: HallucinationDetectionDataset = instantiate(transfer_cfg)
        X, y, train_test_indices, _ = transfer_dataset.process()
        if method_name == "mtopdiv":
            model.cache_dir = (
                Path(cfg["method"]["cache_dir"]).parent
                / transfer_name
                / f"zero_out_{cfg['method']['zero_out']}"
                / model_name
            )
        X_transformed: tp.Sequence[T] = model.transform(X)
        X_transformed = [X_transformed[i] for i in train_test_indices]
        y = y.iloc[train_test_indices]

        metrics = evaluate(
            model,
            X_transformed,
            y.values,
            pretrained=True,
            **cfg["evaluation"],
        )

        table_str, raw_table = process_metrics(metrics, None)  # Pass None instead of experiment
        logger.success(
            f"Transfer for {transfer_name} on {transfer_dataset.model_name} model"
        )
        logger.info(f"{transfer_name} ROC AUC: {raw_table['roc_auc'].loc['test']['mean']:.3f}")
        print(
            f"Transfer for {transfer_name} on {transfer_dataset.model_name} model"
        )
        print(f"{transfer_name} ROC AUC: {raw_table['roc_auc'].loc['test']['mean']:.3f}")
        print(table_str)


if __name__ == "__main__":
    main()
