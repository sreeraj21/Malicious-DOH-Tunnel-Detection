import pandas as pd
import numpy as np
import os
import time
import warnings

from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

import lightgbm as lgb


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_SIZE = 50000

# Original / final Lasso configuration
LASSO_CV = 5
LASSO_CS = 10
LASSO_MAX_ITER = 1000
LASSO_TOL = 1e-4

RANDOM_STATE = 42

DATA_DIR = "data/processed"
OUTPUT_DIR = "data/pruned"

DATASETS = [
    "cleaned_l1-total-add.csv",
    "cleaned_l2-total-add.csv",
    "cleaned_l3-total-add.csv"
]

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Don't hide convergence warnings during this final run
warnings.filterwarnings("default")


# ============================================================
# STORE FINAL RESULTS FOR ALL DATASETS
# ============================================================

all_results = {}


# ============================================================
# PROCESS EACH DATASET
# ============================================================

for file_name in DATASETS:

    dataset_start = time.time()

    print("\n")
    print("=" * 80)
    print(f"RUNNING MFS-DoH CONSENSUS PIPELINE")
    print(f"Dataset: {file_name}")
    print("=" * 80)

    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------

    print("\n[1/6] Loading dataset...")

    load_start = time.time()

    input_path = os.path.join(DATA_DIR, file_name)

    df = pd.read_csv(input_path)

    load_time = time.time() - load_start

    print(f"    Dataset shape: {df.shape}")
    print(f"    Loading completed: {load_time:.2f} seconds")

    # --------------------------------------------------------
    # 2. SEPARATE FEATURES AND TARGET
    # --------------------------------------------------------

    print("\n[2/6] Separating features and target...")

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    feature_names = X.columns.tolist()

    print(f"    Total features: {len(feature_names)}")
    print(f"    Total samples: {len(df)}")
    print(f"    Target column: {df.columns[-1]}")

    print("\n    Class distribution:")
    print(y.value_counts())

    # --------------------------------------------------------
    # 3. ENCODE TARGET + SAMPLE + SCALE
    # --------------------------------------------------------

    print("\n[3/6] Encoding, sampling and scaling...")

    preprocessing_start = time.time()

    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(f"    Classes: {list(le.classes_)}")

    # Sample 50,000 rows
    if len(X) > SAMPLE_SIZE:

        print(f"    Taking random sample of {SAMPLE_SIZE:,} rows...")

        sample_indices = X.sample(
            n=SAMPLE_SIZE,
            random_state=RANDOM_STATE
        ).index

        X_sample = X.loc[sample_indices]
        y_sample = y_encoded[sample_indices]

    else:

        print(
            f"    Dataset has only {len(X):,} rows. "
            f"Using complete dataset."
        )

        X_sample = X
        y_sample = y_encoded

    print(f"    Sample shape: {X_sample.shape}")

    # Scale features
    print("    Scaling features...")

    scaler = StandardScaler()

    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_sample),
        columns=feature_names,
        index=X_sample.index
    )

    preprocessing_time = time.time() - preprocessing_start

    print(
        f"    Preprocessing completed: "
        f"{preprocessing_time:.2f} seconds"
    )


    # ========================================================
    # JUDGE 1 — LIGHTGBM
    # ========================================================

    print("\n")
    print("-" * 80)
    print("[4/6] JUDGE 1/3: LIGHTGBM FEATURE IMPORTANCE")
    print("-" * 80)

    lgb_start = time.time()

    print("    Training LightGBM models...")

    lgbm_scores_total = np.zeros(X_scaled.shape[1])

    for n_est in [50, 75, 100]:

        print(f"    -> Training LightGBM with {n_est} estimators...")

        clf = lgb.LGBMClassifier(
            n_estimators=n_est,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1
        )

        clf.fit(X_scaled, y_sample)

        lgbm_scores_total += clf.feature_importances_

    # Normalize LightGBM scores
    minmax = MinMaxScaler()

    lgbm_norm = minmax.fit_transform(
        lgbm_scores_total.reshape(-1, 1)
    ).flatten()

    lgb_time = time.time() - lgb_start

    print(
        f"    LightGBM completed: "
        f"{lgb_time:.2f} seconds "
        f"({lgb_time / 60:.2f} minutes)"
    )


    # ========================================================
    # JUDGE 2 — ANOVA + LASSO
    # ========================================================

    print("\n")
    print("-" * 80)
    print("[5/6] JUDGE 2/3: ANOVA-LASSO")
    print("-" * 80)

    lasso_start = time.time()

    # --------------------------------------------------------
    # ANOVA
    # --------------------------------------------------------

    print("    Step 1: Running ANOVA...")

    anova_start = time.time()

    f_scores, _ = f_classif(
        X_scaled,
        y_sample
    )

    top_20_anova_indices = np.argsort(f_scores)[-20:]

    X_anova = X_scaled.iloc[:, top_20_anova_indices]

    anova_time = time.time() - anova_start

    print(
        f"    ANOVA completed: "
        f"{anova_time:.2f} seconds"
    )

    print("    Top 20 ANOVA features:")

    anova_features = [
        feature_names[i]
        for i in top_20_anova_indices
    ]

    for rank, feature in enumerate(
        reversed(anova_features),
        start=1
    ):
        print(f"       {rank:2d}. {feature}")


    # --------------------------------------------------------
    # LASSO
    # --------------------------------------------------------

    print("\n    Step 2: Running LogisticRegressionCV...")
    print(f"       Samples       : {len(X_anova):,}")
    print(f"       Features      : {X_anova.shape[1]}")
    print(f"       CV folds      : {LASSO_CV}")
    print(f"       C values      : {LASSO_CS}")
    print(f"       Total models : {LASSO_CV * LASSO_CS}")
    print(f"       Max iterations: {LASSO_MAX_ITER}")
    print(f"       Tolerance     : {LASSO_TOL}")

    print("\n    Starting Lasso... Please wait.")
    print("    This is the most computationally expensive stage.")

    # --------------------------------------------------------
    # IMPORTANT:
    # verbose=0 prevents thousands of Epoch messages
    # from filling the terminal.
    # --------------------------------------------------------

    lasso = LogisticRegressionCV(
        penalty="l1",
        solver="saga",
        cv=LASSO_CV,
        Cs=LASSO_CS,
        tol=LASSO_TOL,
        random_state=RANDOM_STATE,
        max_iter=LASSO_MAX_ITER,
        n_jobs=-1,
        verbose=0
    )

    lasso.fit(
        X_anova,
        y_sample
    )

    # Create scores for ALL original features
    lasso_scores = np.zeros(
        X_scaled.shape[1]
    )

    # Mean absolute coefficient across classes
    lasso_scores[
        top_20_anova_indices
    ] = np.mean(
        np.abs(lasso.coef_),
        axis=0
    )

    # Normalize Lasso scores
    lasso_norm = minmax.fit_transform(
        lasso_scores.reshape(-1, 1)
    ).flatten()

    lasso_time = time.time() - lasso_start

    print(
        f"\n    >>> Lasso completed in "
        f"{lasso_time:.2f} seconds "
        f"({lasso_time / 60:.2f} minutes)"
    )


    # ========================================================
    # JUDGE 3 — MUTUAL INFORMATION
    # ========================================================

    print("\n")
    print("-" * 80)
    print("[6/6] JUDGE 3/3: MUTUAL INFORMATION")
    print("-" * 80)

    mi_start = time.time()

    print("    Calculating feature-target dependency...")

    mi_scores = mutual_info_classif(
        X_scaled,
        y_sample,
        random_state=RANDOM_STATE
    )

    mi_norm = minmax.fit_transform(
        mi_scores.reshape(-1, 1)
    ).flatten()

    mi_time = time.time() - mi_start

    print(
        f"    Mutual Information completed: "
        f"{mi_time:.2f} seconds "
        f"({mi_time / 60:.2f} minutes)"
    )


    # ========================================================
    # CONSENSUS RANKING
    # ========================================================

    print("\n")
    print("=" * 70)
    print("CONSENSUS RANKING")
    print("=" * 70)

    consensus_scores = (
        lgbm_norm +
        lasso_norm +
        mi_norm
    ) / 3.0

    # Top 20 consensus features
    top_20_consensus_idx = np.argsort(
        consensus_scores
    )[-20:]

    # Sort highest score first for display
    sorted_consensus_idx = (
        top_20_consensus_idx[
            np.argsort(
                consensus_scores[top_20_consensus_idx]
            )[::-1]
        ]
    )

    print("\nTop 20 Consensus Features:")

    for rank, idx in enumerate(
        sorted_consensus_idx,
        start=1
    ):

        print(
            f"    {rank:2d}. "
            f"{feature_names[idx]} "
            f"(score={consensus_scores[idx]:.6f})"
        )


    # ========================================================
    # FINAL TOP 5 REFINEMENT
    # ========================================================

    print("\nFinal refinement:")
    print(
        "    Applying Mutual Information "
        "to consensus Top 20..."
    )

    X_consensus = X_scaled.iloc[
        :,
        top_20_consensus_idx
    ]

    final_mi_scores = mutual_info_classif(
        X_consensus,
        y_sample,
        random_state=RANDOM_STATE
    )

    final_5_relative_idx = np.argsort(
        final_mi_scores
    )[-5:]

    final_5_absolute_idx = (
        top_20_consensus_idx[
            final_5_relative_idx
        ]
    )

    final_features = [
        feature_names[i]
        for i in final_5_absolute_idx
    ]


    # ========================================================
    # PRINT FINAL 5 FEATURES
    # ========================================================

    print("\n")
    print("=" * 70)
    print(f"FINAL 5 FEATURES FOR {file_name}")
    print("=" * 70)

    for rank, feature in enumerate(
        final_features,
        start=1
    ):
        print(
            f"    {rank}. {feature}"
        )


    # ========================================================
    # SAVE PRUNED DATASET
    # ========================================================

    df_pruned = df[
        final_features +
        [df.columns[-1]]
    ]

    output_path = os.path.join(
        OUTPUT_DIR,
        f"pruned_{file_name}"
    )

    df_pruned.to_csv(
        output_path,
        index=False
    )

    print("\n[SUCCESS] Pruned dataset saved:")
    print(f"    {output_path}")

    dataset_time = time.time() - dataset_start

    print("\nTotal processing time for this dataset:")
    print(
        f"    {dataset_time:.2f} seconds "
        f"({dataset_time / 60:.2f} minutes)"
    )

    print("=" * 70)


    # ========================================================
    # SAVE RESULT FOR FINAL SUMMARY
    # ========================================================

    all_results[file_name] = {
        "features": final_features,
        "total_time": dataset_time
    }


# ============================================================
# FINAL SUMMARY — ALL 3 DATASETS
# ============================================================

print("\n\n")
print("#" * 80)
print("#" * 80)
print("#")
print("#              FINAL FEATURE SELECTION SUMMARY")
print("#")
print("#              50,000 SAMPLE / 5-FOLD / 10-C")
print("#")
print("#" * 80)
print("#" * 80)


for file_name in DATASETS:

    print("\n")
    print("-" * 80)

    if file_name in all_results:

        print(
            f"DATASET: {file_name}"
        )

        print("-" * 80)

        features = all_results[file_name]["features"]
        total_time = all_results[file_name]["total_time"]

        print("\nFINAL 5 FEATURES:")

        for rank, feature in enumerate(
            features,
            start=1
        ):
            print(
                f"    {rank}. {feature}"
            )

        print(
            f"\nProcessing time: "
            f"{total_time:.2f} seconds "
            f"({total_time / 60:.2f} minutes)"
        )

    else:

        print(
            f"DATASET: {file_name}"
        )

        print(
            "STATUS: NOT COMPLETED"
        )

    print("-" * 80)


# ============================================================
# VERY SHORT COPY-PASTE SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("EASY-TO-FIND FINAL RESULTS")
print("=" * 80)

for file_name in DATASETS:

    if file_name in all_results:

        print(
            f"\n{file_name}:"
        )

        for feature in all_results[file_name]["features"]:
            print(
                f"  - {feature}"
            )

    else:

        print(
            f"\n{file_name}: NOT COMPLETED"
        )

print("\n")
print("=" * 80)
print("ALL DATASETS COMPLETED")
print("=" * 80)