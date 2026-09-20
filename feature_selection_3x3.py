import pandas as pd
import numpy as np
import os
import time

from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

import lightgbm as lgb


# ============================================================
# CONFIGURATION - DIAGNOSTIC TEST
# ============================================================

SAMPLE_SIZE = 10000

LASSO_CV = 3
LASSO_CS = 3
LASSO_MAX_ITER = 300
LASSO_TOL = 1e-3

RANDOM_STATE = 42

datasets = [
    'cleaned_l1-total-add.csv',
    'cleaned_l2-total-add.csv',
    'cleaned_l3-total-add.csv'
]

os.makedirs('data/pruned', exist_ok=True)


# ============================================================
# HELPER FUNCTION
# ============================================================

def print_time(start_time, message):
    elapsed = time.time() - start_time
    print(f"    {message}: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)")


# ============================================================
# MAIN PIPELINE
# ============================================================

for file_name in datasets:

    total_start = time.time()

    print("\n" + "=" * 70)
    print(f"Running MFS-DoH Consensus Pipeline on {file_name}")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------

    stage_start = time.time()

    print("\n[1/6] Loading dataset...")

    file_path = f'data/processed/{file_name}'

    df = pd.read_csv(file_path)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    feature_names = X.columns.tolist()

    print(f"    Original dataset shape : {df.shape}")
    print(f"    Original samples      : {len(df):,}")
    print(f"    Number of features    : {X.shape[1]}")
    print(f"    Number of classes     : {y.nunique()}")

    print("\n    Class distribution:")
    print(y.value_counts())

    print_time(stage_start, "Dataset loading completed")


    # --------------------------------------------------------
    # 2. ENCODE TARGET + SCALE
    # --------------------------------------------------------

    stage_start = time.time()

    print("\n[2/6] Encoding target and scaling features...")

    # Encode target labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(f"    Classes: {list(le.classes_)}")

    # Standardize features
    scaler = StandardScaler()

    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=feature_names,
        index=X.index
    )

    print_time(stage_start, "Scaling completed")


    # --------------------------------------------------------
    # 3. TAKE 10,000 SAMPLE
    # --------------------------------------------------------

    stage_start = time.time()

    print(f"\n[3/6] Creating sample of {SAMPLE_SIZE:,} rows...")

    if len(X_scaled) > SAMPLE_SIZE:

        X_sample = X_scaled.sample(
            n=SAMPLE_SIZE,
            random_state=RANDOM_STATE
        )

        # IMPORTANT:
        # Use the same original indices to obtain matching labels
        y_sample = y_encoded[X_sample.index.values]

    else:

        X_sample = X_scaled
        y_sample = y_encoded

    print(f"    Sample shape: {X_sample.shape}")
    print(f"    Sample rows : {len(X_sample):,}")

    print_time(stage_start, "Sampling completed")


    # Min-Max scaler for feature-selection scores
    minmax = MinMaxScaler()


    # ========================================================
    # METHOD 1: LIGHTGBM
    # ========================================================

    stage_start = time.time()

    print("\n[4/6] Judge 1/3: LightGBM feature importance...")
    print("    Evaluating tree-based non-linear relationships...")

    lgbm_scores_total = np.zeros(X_sample.shape[1])

    for n_est in [50, 75, 100]:

        print(f"    -> Training LightGBM with {n_est} trees...")

        clf = lgb.LGBMClassifier(
            n_estimators=n_est,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1
        )

        clf.fit(X_sample, y_sample)

        lgbm_scores_total += clf.feature_importances_

    lgbm_norm = minmax.fit_transform(
        lgbm_scores_total.reshape(-1, 1)
    ).flatten()

    print_time(stage_start, "LightGBM completed")


    # ========================================================
    # METHOD 2: HYBRID ANOVA-LASSO
    # ========================================================

    stage_start = time.time()

    print("\n[5/6] Judge 2/3: Hybrid ANOVA-Lasso...")
    print("    Step 1: ANOVA selecting top 20 features...")

    # --------------------------------------------------------
    # ANOVA
    # --------------------------------------------------------

    f_scores, _ = f_classif(
        X_sample,
        y_sample
    )

    top_20_anova_indices = np.argsort(f_scores)[-20:]

    print("    Top 20 ANOVA features:")

    for rank, idx in enumerate(
        top_20_anova_indices[::-1],
        start=1
    ):
        print(
            f"       {rank:2d}. "
            f"{feature_names[idx]} "
            f"(F-score={f_scores[idx]:.4f})"
        )

    X_anova = X_sample.iloc[:, top_20_anova_indices]

    print(f"\n    ANOVA output shape: {X_anova.shape}")


    # --------------------------------------------------------
    # LASSO LOGISTIC REGRESSION
    # --------------------------------------------------------

    print("\n    Step 2: L1 Logistic Regression...")
    print(f"       CV folds       : {LASSO_CV}")
    print(f"       C values       : {LASSO_CS}")
    print(f"       Maximum iter   : {LASSO_MAX_ITER}")
    print(f"       Total models  : {LASSO_CV * LASSO_CS}")
    print("       Starting Lasso fitting...")
    print()

    lasso = LogisticRegressionCV(
        penalty='l1',
        solver='saga',
        cv=LASSO_CV,
        Cs=LASSO_CS,
        tol=LASSO_TOL,
        random_state=RANDOM_STATE,
        max_iter=LASSO_MAX_ITER,
        n_jobs=-1,
        verbose=1
    )

    lasso_start = time.time()

    lasso.fit(
        X_anova,
        y_sample
    )

    lasso_elapsed = time.time() - lasso_start

    print(
        f"\n    >>> Lasso completed in "
        f"{lasso_elapsed:.2f} seconds "
        f"({lasso_elapsed / 60:.2f} minutes)"
    )


    # --------------------------------------------------------
    # LASSO FEATURE SCORES
    # --------------------------------------------------------

    lasso_scores = np.zeros(
        X_sample.shape[1]
    )

    # Average absolute coefficients across classes
    lasso_scores[top_20_anova_indices] = np.mean(
        np.abs(lasso.coef_),
        axis=0
    )

    lasso_norm = minmax.fit_transform(
        lasso_scores.reshape(-1, 1)
    ).flatten()

    print_time(
        stage_start,
        "ANOVA-Lasso completed"
    )


    # ========================================================
    # METHOD 3: MUTUAL INFORMATION
    # ========================================================

    stage_start = time.time()

    print("\n[6/6] Judge 3/3: Mutual Information...")
    print("    Calculating feature-target dependency...")

    mi_scores = mutual_info_classif(
        X_sample,
        y_sample,
        random_state=RANDOM_STATE
    )

    mi_norm = minmax.fit_transform(
        mi_scores.reshape(-1, 1)
    ).flatten()

    print_time(
        stage_start,
        "Mutual Information completed"
    )


    # ========================================================
    # CONSENSUS RANKING
    # ========================================================

    stage_start = time.time()

    print("\n" + "-" * 70)
    print("CONSENSUS RANKING")
    print("-" * 70)

    # Equal-weight consensus
    consensus_scores = (
        lgbm_norm +
        lasso_norm +
        mi_norm
    ) / 3.0

    # Select top 20 consensus features
    top_20_consensus_idx = np.argsort(
        consensus_scores
    )[-20:]

    print("\nTop 20 Consensus Features:")

    for rank, idx in enumerate(
        top_20_consensus_idx[::-1],
        start=1
    ):
        print(
            f"    {rank:2d}. "
            f"{feature_names[idx]} "
            f"(score={consensus_scores[idx]:.6f})"
        )


    # ========================================================
    # FINAL MUTUAL INFORMATION REFINEMENT
    # ========================================================

    print("\nFinal refinement:")
    print("    Applying Mutual Information to consensus Top 20...")

    X_consensus = X_sample.iloc[
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
    # FINAL RESULT
    # ========================================================

    print("\n" + "=" * 70)
    print(f"FINAL 5 FEATURES FOR {file_name}")
    print("=" * 70)

    for i, feature in enumerate(
        final_features,
        start=1
    ):
        print(f"    {i}. {feature}")


    # ========================================================
    # SAVE PRUNED DATASET
    # ========================================================

    df_pruned = df[
        final_features + [df.columns[-1]]
    ]

    output_path = (
        f"data/pruned/pruned_{file_name}"
    )

    df_pruned.to_csv(
        output_path,
        index=False
    )

    print("\n[SUCCESS] Pruned dataset saved:")
    print(f"    {output_path}")

    print(f"\nTotal processing time for {file_name}:")
    total_elapsed = time.time() - total_start

    print(
        f"    {total_elapsed:.2f} seconds "
        f"({total_elapsed / 60:.2f} minutes)"
    )

    print("=" * 70)


print("\n")
print("=" * 70)
print("ALL DATASETS COMPLETED")
print("=" * 70)