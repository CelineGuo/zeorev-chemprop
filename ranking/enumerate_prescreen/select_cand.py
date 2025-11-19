import os
import pandas as pd
import numpy as np
import subprocess
from sklearn.preprocessing import MinMaxScaler
import itertools

def load_zeolite_descriptor(path, zeo):
    df = pd.read_csv(path)
    zeo_feaat_only = df.loc[:, 'deriv_dist_46':'nav_cm3_g']
    zeo_feat_col = zeo_feaat_only.columns.tolist()
    scaler = MinMaxScaler()
    normalized_array = scaler.fit_transform(zeo_feaat_only)

    normalized_df = pd.DataFrame(normalized_array, columns=zeo_feat_col)
    normalized_df = pd.concat([df["Code"], normalized_df], axis=1)
    filtered_df = normalized_df[normalized_df["Code"] == zeo].copy()
    filtered_df.drop(columns=["Code"], inplace=True)
    
    filtered_df.fillna(0, inplace=True)
    
    zeo_cols = filtered_df.columns.tolist()
    return filtered_df, zeo_cols

def enumerate_from_file(pred_file, top_k=100, max_osda=3):
    """
    pred_file: CSV file with columns ['osda', 'normalized_yield']
    top_k: keep only top-k single OSDAs before building combos
    max_osda: max number of OSDAs in a combination (default=3)
    """
    # Load predictions
    df = pd.read_csv(pred_file)

    # Sort by score
    df = df.sort_values(by="normalized_yield", ascending=False).reset_index(drop=True)

    # Keep top-k OSDAs
    df = df.head(top_k)

    # Extract OSDAs and scores
    osdas = df["osda"].tolist()
    scores = df["normalized_yield"].tolist()

    # Store combinations
    combos = []
    for r in range(1, max_osda + 1):  # generate 1, 2, 3-OSDA combos
        for idxs in itertools.combinations(range(len(osdas)), r):
            selected_osdas = [osdas[i] for i in idxs]
            selected_scores = [scores[i] for i in idxs]

            # Aggregate scores (mean by default)
            combo_score = sum(selected_scores) / len(selected_scores)

            combos.append((".".join(selected_osdas), combo_score))

    # Sort final combinations
    combos = sorted(combos, key=lambda x: x[1], reverse=True)

    return combos

zeo_type = "SSF"
zeo_path = ".../data_reduced_1.csv" ##where you save the zeolite descriptor file
zeo_df, zeo_cols = load_zeolite_descriptor(zeo_path, zeo=zeo_type)
num_zeo_col = len(zeo_cols)
new_zeo_col_names = [str(i) for i in range(num_zeo_col)]
zeo_df.columns = new_zeo_col_names
zeo_cols = new_zeo_col_names

##extracted unique OSDAs (SMILES or molecules)
with open("/home/CelineGuo73/zeolites/data/valid_osdas.txt", "r") as f:
    osda_pool = [line.strip() for line in f if line.strip()]

n = len(osda_pool)
zeo_repeated = pd.concat([zeo_df] * n, ignore_index=True)
print(zeo_repeated.shape)

dir = os.path.join("...", zeo_type)
#to your folder
os.makedirs(dir, exist_ok=True)

train_data = pd.read_csv("...") ##to your zeolite descriptor data ".../feat_train.csv"
train_zeo_type = pd.read_csv("...") ##to your zeolite descriptor data ".../zeo_desc_train.csv"
merged_df = pd.concat([train_data, train_zeo_type], axis=1)
print(merged_df.columns)
filtered_df = merged_df[merged_df["zeolite_code"] == zeo_type]

zeo_repeated.to_csv(os.path.join(dir, "ssf_zeo_feat.csv"), index=False)
osda_pool_df = pd.DataFrame(osda_pool, columns=["osda"])
osda_pool_df.to_csv(os.path.join(dir, "ssf_osda_pool.csv"), index=False)

subprocess.run([
    "conda", "run", "-n", "chemprop-gpu",
    "python", "ranking/chemprop-v1-old-branches/predict.py",
    "--checkpoint_dir", "...",
    "--test_path", ".../Zeolite_Type/SSF/ssf_osda_pool.csv",
    "--features_path", ".../Zeolite_Type/SSF/ssf_zeo_feat.csv",
    "--preds_path", ".../Zeolite_Type/SSF/ssf1_candidates_threshold_comb.csv",

])