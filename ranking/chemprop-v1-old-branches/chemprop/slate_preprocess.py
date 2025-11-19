# slate_postprocess.py
from __future__ import annotations
from typing import Optional, Sequence, Tuple, Dict, List
import re
import numpy as np
import pandas as pd

# ------------------------
# CSV loaders / detectors
# ------------------------

_SMILES_NAME_RE = re.compile(r"(?:^|_)(smiles?)(?:$|_)", re.I)
_SMILES_TOKEN_RE = re.compile(r"^[A-Za-z0-9@+\-\[\]\(\)=#\$\\/\.]+$")

def detect_smiles_column(df: pd.DataFrame) -> Optional[str]:
    """
    Heuristically pick the FIRST SMILES column:
    1) header name contains 'smiles' (case-insensitive)
    2) else first object/string column whose values look like SMILES tokens
    """
    # Rule 1: name contains 'smiles'
    for c in df.columns:
        if _SMILES_NAME_RE.search(str(c)):
            return c

    # Rule 2: token pattern check on string/object columns
    for c in df.columns:
        s = df[c]
        if not pd.api.types.is_object_dtype(s):
            continue
        sample = s.dropna().astype(str).head(50)
        if len(sample) == 0:
            continue
        ok = (sample.str.len().between(2, 512) &
              sample.str.match(_SMILES_TOKEN_RE))
        if ok.mean() > 0.7:
            return c
    return None


def load_test_and_features(test_path: str, features_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Load test CSV and features CSV. Detect smiles column from test CSV.
    Returns (test_df, feat_df, smiles_col).
    """
    test_df = pd.read_csv(test_path)
    smiles_col = detect_smiles_column(test_df) or test_df.columns[0]
    feat_df  = pd.read_csv(features_path)
    if len(feat_df) != len(test_df):
        raise ValueError(
            f"features_path rows ({len(feat_df)}) != test_path rows ({len(test_df)}). "
            "They must be aligned row-by-row."
        )
    return test_df, feat_df, smiles_col


# ------------------------
# Grouping / slates
# ------------------------

def features_to_group_ids_from_df(
    feat_df: pd.DataFrame,
    *,
    decimals: int = 6,
    feat_slice: Optional[slice] = None,
) -> np.ndarray:
    """
    Convert numeric features into group ids (0..G-1) by rounding + hashing.
    Assumes feat_df rows align with test rows (full_data order).

    feat_slice: apply a slice to columns AFTER converting to ndarray,
                e.g., slice(0, 128) to match training.
    """
    # Keep only numeric columns
    num_df = feat_df.select_dtypes(include=[np.number])
    if num_df.shape[1] == 0:
        raise ValueError("No numeric columns found in features CSV.")

    arr = num_df.to_numpy(dtype=np.float32)
    if feat_slice is not None:
        arr = arr[:, feat_slice]

    mapping: Dict[bytes, int] = {}
    gids = np.empty(arr.shape[0], dtype=np.int64)
    next_id = 0
    for i in range(arr.shape[0]):
        key = np.round(arr[i], decimals=decimals).tobytes()
        gid = mapping.get(key)
        if gid is None:
            gid = next_id
            mapping[key] = gid
            next_id += 1
        gids[i] = gid
    return gids


def softmax_stable(x: np.ndarray, tau: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mask = ~np.isfinite(x)
    if np.all(mask):
        return np.zeros_like(x, dtype=np.float64)
    v = x[~mask]
    z = (v - v.max()) / float(tau)
    e = np.exp(z)
    p = np.zeros_like(x, dtype=np.float64)
    p[~mask] = e / e.sum()
    return p


def per_slate_softmax_and_rank(
    scores: Sequence[float],
    group_ids: Sequence[int],
    *,
    tau: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    """
    scores/group_ids are in FULL data order (same length as test CSV).
    Returns:
      prob_top1      : softmax per slate
      rank_in_slate  : 1..|slate| (invalid/non-finite scores become worst)
      global_rank_map: {row_index -> "RankX"} by descending score
    """
    N = len(scores)
    scores = np.asarray(scores, dtype=np.float64)
    gids = np.asarray(group_ids, dtype=np.int64)

    prob_top1 = np.zeros(N, dtype=np.float64)
    rank_in_slate = np.zeros(N, dtype=np.int64)

    for gid in np.unique(gids):
        if gid < 0:
            continue
        idxs = np.where(gids == gid)[0]
        if idxs.size == 0:
            continue
        grp = scores[idxs]
        prob_top1[idxs] = softmax_stable(grp, tau=tau)

        # rank best=1; push non-finite to the end
        order = np.argsort(grp, kind="mergesort")[::-1]
        finite = np.isfinite(grp[order])
        order = np.concatenate([order[finite], order[~finite]])
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, order.size + 1)
        rank_in_slate[idxs] = ranks

    global_order = np.argsort(scores, kind="mergesort")[::-1]
    global_rank_map = {int(i): f"Rank{r+1}" for r, i in enumerate(global_order)}
    return prob_top1, rank_in_slate, global_rank_map


# ------------------------
# High-level entrypoints
# ------------------------

def build_full_group_ids_from_csv(
    test_path: str,
    features_path: str,
    *,
    decimals: int = 6,
    feat_slice: Optional[slice] = None,
) -> Tuple[np.ndarray, str]:
    """
    Build group ids aligned to FULL test CSV rows (including invalid rows if any).
    Also returns the detected smiles column name.
    """
    test_df, feat_df, smiles_col = load_test_and_features(test_path, features_path)
    full_gids = features_to_group_ids_from_df(feat_df, decimals=decimals, feat_slice=feat_slice)
    return full_gids, smiles_col


def attach_slate_outputs_to_full_data(
    full_data,
    *,
    scores: Sequence[float],
    prob_top1: Sequence[float],
    rank_in_slate: Sequence[int],
    global_rank_map: Dict[int, str],
    group_ids: Sequence[int],
    score_col: str = "Score",
    prob_col: str = "prob_top1",
    rank_col: str = "RankInSlate",
    global_rank_col: str = "GlobalRank",
    slate_id_col: Optional[str] = "SlateID",
) -> None:
    for i, dp in enumerate(full_data):
        dp.row[score_col] = float(scores[i])
        dp.row[prob_col] = float(prob_top1[i])
        dp.row[rank_col] = int(rank_in_slate[i]) if rank_in_slate[i] > 0 else ""
        dp.row[global_rank_col] = global_rank_map.get(i, "")
        if slate_id_col is not None:
            dp.row[slate_id_col] = int(group_ids[i]) if group_ids[i] >= 0 else ""
