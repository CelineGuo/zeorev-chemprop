import numpy as np
import pandas as pd
from rdkit import Chem
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from rdkit.Chem import AllChem, DataStructs
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import random

def smiles_to_fp(smiles, radius=2, n_bits=2048):
    """Convert SMILES to Morgan fingerprint bit vector."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def generate_artificial_data(
    df,
    zeolite_col,
    osda_col,
    class_col,
    label_smoothing=0.0,
    # tau_o=0.6,
    max_sim_zeo=1,
    # max_sim_osda=1,
    unseen_per_zeo=2,
    max_random_ratio=0.5,
    random_state=42
):
    rng = np.random.default_rng(random_state)

    # df[desc_cols] = df[desc_cols].apply(pd.to_numeric, errors="coerce")

    # zeo_desc = df.groupby(zeolite_col)[desc_cols]
    # zeo_desc_dict = zeo_desc.to_dict(orient="index")
    # scaler = MinMaxScaler()
    # zeo_scaled = scaler.fit_transform(zeo_desc.values)
    # zeo_sims = cosine_similarity(zeo_scaled)
    # zeo_codes = zeo_desc.index.tolist()
    # zeo_index = {z: i for i, z in enumerate(zeo_codes)}
    zeo_codes = df[zeolite_col].unique().tolist()
    zeo_index = {z: i for i, z in enumerate(zeo_codes)}

    # Identity similarity (each zeolite only similar to itself)
    zeo_sims = np.eye(len(zeo_codes))

    #Prepare OSDA fingerprints
    unique_osdas = pd.unique(df[osda_col])
    osda_fps = {smi: smiles_to_fp(smi) for smi in unique_osdas if pd.notna(smi)}
    fps = {smi: AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smi), 2) for smi in osda_fps}
    def osda_sim(o1, o2):
        return DataStructs.TanimotoSimilarity(fps[o1], fps[o2])

    #Unique key tracker
    df["synth_key"] = df[[osda_col, zeolite_col, class_col]].astype(str).agg("_".join, axis=1)
    existing_keys = set(df["synth_key"])

    pure_df = df[df[class_col] == 1]
    artificial_data = []

    # === Zeolite-similarity negatives ===
    for _, row in pure_df.iterrows():
        osda, true_zeo = row[osda_col], row[zeolite_col]
        if true_zeo not in zeo_index:
            continue
        sims = zeo_sims[zeo_index[true_zeo]]
        sorted_idx = np.argsort(-sims)
        count = 0
        for j in sorted_idx:
            neg_zeo = zeo_codes[j]
            if neg_zeo == true_zeo:
                continue
            synth_key = "_".join([str(osda), str(neg_zeo), "0"])
            if synth_key not in existing_keys:
                # desc_values = zeo_desc_dict.get(neg_zeo, {})
                # desc_values = desc_values or {c: np.nan for c in desc_cols}  # fallback

                artificial_data.append({
                    osda_col: osda,
                    zeolite_col: neg_zeo,
                    class_col: label_smoothing,   # smoothed negative
                    "synth_key": synth_key
                })
                existing_keys.add(synth_key)
                count += 1
            if count >= max_sim_zeo:
                break

    # # === OSDA-similarity negatives ===
    # for _, row in pure_df.iterrows():
    #     zeo, true_osda = row[zeolite_col], row[osda_col]
    #     if true_osda not in osda_fps:
    #         continue

    #     sims = [(o, osda_sim(true_osda, o)) for o in osda_fps if o != true_osda]
    #     sims = sorted(sims, key=lambda x: -x[1])
    #     count = 0

    #     for o, sim in sims:
    #         if sim < tau_o:
    #             break
    #         synth_key = "_".join([str(o), str(zeo), "0"])
    #         if synth_key not in existing_keys:
    #             # === inject zeolite descriptors here ===
    #             desc_values = zeo_desc_dict.get(zeo, {})
    #             desc_values = desc_values or {c: np.nan for c in desc_cols}

    #             artificial_data.append({
    #                 osda_col: o,
    #                 zeolite_col: zeo,
    #                 class_col: label_smoothing,   # smoothed negative
    #                 **desc_values,                # expand descriptor columns
    #                 "synth_key": synth_key
    #             })
    #             existing_keys.add(synth_key)
    #             count += 1
    #         if count >= max_sim_osda:
    #             break

    # === Random unseen negatives (regularization) ===
    extra_data = []
    for zeo in zeo_codes:
        osdas_seen = set(df[df[zeolite_col] == zeo][osda_col])
        osdas_unseen = list(set(osda_fps) - osdas_seen)
        rng.shuffle(osdas_unseen)
        for osda in osdas_unseen[:unseen_per_zeo]:
            synth_key = "_".join([str(osda), str(zeo), "0"])
            if synth_key not in existing_keys:
                # desc_values = zeo_desc_dict.get(zeo, {})
                # desc_values = desc_values or {c: np.nan for c in desc_cols}
                extra_data.append({
                    osda_col: osda,
                    zeolite_col: zeo,
                    class_col: label_smoothing,
                    "synth_key": synth_key
                })
                existing_keys.add(synth_key)

    # === Limit total random negatives ===
    orig_pos = int((df[class_col] == 1).sum())
    max_random_total = max_random_ratio * orig_pos
    if len(extra_data) > max_random_total:
        extra_data = random.sample(extra_data, max_random_total)

    df_labeled = df.copy()

    artificial_data_df = pd.DataFrame(artificial_data + extra_data)
    final_df = pd.concat([df_labeled, artificial_data_df], ignore_index=True)
    final_df = final_df.drop(columns=["synth_key"], errors="ignore")

    # === Statistics summary ===
    orig_pos = int((df_labeled[class_col] == 1).sum())
    orig_neg = int((df_labeled[class_col] == 0).sum())
    synth_total = len(artificial_data_df)
    synth_ratio = synth_total / max(orig_pos, 1)

    print("=" * 65)
    print("[DATA SUMMARY]")
    print(f"Original positives (yield = 1): {orig_pos:,}")
    print(f"Original negatives (yield = 0): {orig_neg:,}")
    print(f"Generated synthetic negatives  : {synth_total:,}")
    print(f"Ratio synthetic_neg / positives: {synth_ratio:.2f} : 1")
    print(f"Final total rows after merge   : {len(final_df):,}")
    print("=" * 65)
    print(final_df[class_col].value_counts())
    print("=" * 65)

    return final_df, artificial_data_df

INPUT_FILE = ".../want_to_gen.csv"
df = pd.read_csv(INPUT_FILE)

final_df,_ = generate_artificial_data(
    df,
    zeolite_col="zeolite_code",
    osda_col="osda",
    class_col="class"
)

final_df.to_csv(".../ranking/data/artificial_data_gen_result_file/after_gen.csv", index=False)