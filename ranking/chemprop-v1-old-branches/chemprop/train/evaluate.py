from collections import defaultdict
import logging
from typing import Dict, List
import numpy as np
import pandas as pd

from .predict import predict
from chemprop.data import MoleculeDataLoader, StandardScaler, AtomBondScaler
from chemprop.models import MoleculeModel
from chemprop.train import get_metric_func
from chemprop.utils import features_to_group_ids


def evaluate_predictions(
    preds: List[List[float]],
    targets: List[List[float]],
    num_tasks: int,
    metrics: List[str],
    logger=None,
    group_ids: np.ndarray = None
):
    """
    Evaluates predictions using metric functions.
    For 'packet_luce_ranking', computes true per-slate NDCG averaged across slates.
    """
    log = logger.info if logger is not None else print
    metric_to_func = {m: get_metric_func(m) for m in metrics}
    results = defaultdict(list)

    if len(preds) == 0:
        return {m: [float('nan')] * num_tasks for m in metrics}

    preds = np.array(preds)
    targets = np.array(targets)

    # --------------------
    # Main metric loop
    # --------------------
    for i in range(num_tasks):
        for metric, metric_func in metric_to_func.items():
            # === Per-slate NDCG ===
            if metric == "packet_luce_ranking":
                if group_ids is not None:
                    unique_groups = np.unique(group_ids)
                    slate_scores = []

                    for gid in unique_groups:
                        idx = np.where(group_ids == gid)[0]
                        if len(idx) < 2:
                            continue
                        y_true = targets[idx, i]
                        y_pred = preds[idx, i]

                        # Filter invalid entries
                        mask = ~np.isnan(y_true)
                        y_true = y_true[mask]
                        y_pred = y_pred[mask]
                        if len(y_true) < 2 or np.sum(y_true) == 0:
                            continue

                        ndcg_g = metric_func([y_true.tolist()], [y_pred.tolist()])
                        slate_scores.append(ndcg_g)

                    if len(slate_scores) == 0:
                        score = float("nan")
                    else:
                        score = float(np.mean(slate_scores))

                    results[metric].append(score)

                else:
                    # Fallback: global evaluation if no grouping
                    score = metric_func([targets[:, i].tolist()],
                                        [preds[:, i].tolist()])
                    results[metric].append(score)

            # === All other metrics ===
            else:
                mask = ~np.isnan(targets[:, i])
                y_true = targets[mask, i]
                y_pred = preds[mask, i]
                score = metric_func(y_true.tolist(), y_pred.tolist())
                results[metric].append(score)

    return dict(results)


def evaluate(model: MoleculeModel,
             data_loader: MoleculeDataLoader,
             num_tasks: int,
             metrics: List[str],
             scaler: StandardScaler = None,
             atom_bond_scaler: AtomBondScaler = None,
             logger: logging.Logger = None) -> Dict[str, List[float]]:
    """
    Evaluates a Chemprop model on a dataset.
    For ranking tasks, this also computes per-slate (grouped) NDCG using the same logic as training.
    """
    # === Make predictions ===
    preds = predict(
        model=model,
        data_loader=data_loader,
        scaler=scaler,
        atom_bond_scaler=atom_bond_scaler,
    )

    # === Compute group IDs (same as training) ===
    all_features = []
    for batch in data_loader:
        features_batch = batch.features()
        all_features.extend(features_batch)

    group_ids = None
    try:
        group_ids = features_to_group_ids(all_features, device="cpu").numpy()
    except Exception as e:
        if logger is not None:
            logger.debug(f"Could not compute group IDs for per-slate ranking: {e}")

    # === Evaluate predictions ===
    results = evaluate_predictions(
        preds=preds,
        targets=data_loader.targets,
        num_tasks=num_tasks,
        metrics=metrics,
        logger=logger,
        group_ids=group_ids,
    )

    return results
