import pandas as pd
import numpy as np
import os
import warnings
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import lightgbm as lgb

warnings.filterwarnings("ignore")
os.makedirs('data/pruned', exist_ok=True)

datasets = ['cleaned_l1-total-add.csv', 'cleaned_l2-total-add.csv', 'cleaned_l3-total-add.csv']

for file_name in datasets:
    print(f"\n{'='*50}")
    print(f"Running MFS-DoH Consensus Pipeline on {file_name}...")
    
    # 1. Load Data
    print(" -> Loading massive dataset and scaling features...")
    df = pd.read_csv(f'data/processed/{file_name}')
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    feature_names = X.columns.tolist()
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_names)
    
    if len(X_scaled) > 50000:
        X_sample = X_scaled.sample(n=50000, random_state=42)
        y_sample = y_encoded[X_sample.index.values]
    else:
        X_sample = X_scaled
        y_sample = y_encoded
        
    minmax = MinMaxScaler()

    # --- METHOD 1: LightGBM Importance ---
    print(" -> Judge 1/3: LightGBM evaluating non-linear tree splits...")
    lgbm_scores_total = np.zeros(X_sample.shape[1])
    for n_est in [50, 75, 100]:
        clf = lgb.LGBMClassifier(n_estimators=n_est, random_state=42, n_jobs=-1, verbose=-1)
        clf.fit(X_sample, y_sample)
        lgbm_scores_total += clf.feature_importances_
    lgbm_norm = minmax.fit_transform(lgbm_scores_total.reshape(-1, 1)).flatten()

    # --- METHOD 2: Hybrid ANOVA-Lasso ---
    print(" -> Judge 2/3: ANOVA-Lasso running 5-fold Cross-Validation (This will take 5-10 minutes. Do not close!)...")
    f_scores, _ = f_classif(X_sample, y_sample)
    top_20_anova_indices = np.argsort(f_scores)[-20:]
    
    X_anova = X_sample.iloc[:, top_20_anova_indices]
    
    # Using the rigorous LassoCV to perfectly match the paper's methodology
    lasso = LogisticRegressionCV(penalty='l1', solver='liblinear', cv=5, random_state=42, max_iter=1000)
    lasso.fit(X_anova, y_sample)
    
    lasso_scores = np.zeros(X_sample.shape[1])
    lasso_scores[top_20_anova_indices] = np.abs(lasso.coef_[0])
    lasso_norm = minmax.fit_transform(lasso_scores.reshape(-1, 1)).flatten()

    # --- METHOD 3: Mutual Information ---
    print(" -> Judge 3/3: Mutual Information calculating entropy (This will take 1-3 minutes)...")
    mi_scores = mutual_info_classif(X_sample, y_sample, random_state=42)
    mi_norm = minmax.fit_transform(mi_scores.reshape(-1, 1)).flatten()

    # --- CONSENSUS RANKING ---
    print(" -> Averaging scores and finding the Top 5 Final Features...")
    consensus_scores = (lgbm_norm + lasso_norm + mi_norm) / 3.0
    top_20_consensus_idx = np.argsort(consensus_scores)[-20:]
    
    X_consensus = X_sample.iloc[:, top_20_consensus_idx]
    final_mi_scores = mutual_info_classif(X_consensus, y_sample, random_state=42)
    
    final_5_relative_idx = np.argsort(final_mi_scores)[-5:]
    final_5_absolute_idx = top_20_consensus_idx[final_5_relative_idx]
    final_features = [feature_names[i] for i in final_5_absolute_idx]
    
    print(f"\n[SUCCESS] Final 5 Features Selected: {final_features}")
    
    # Save Output
    df_pruned = df[final_features + [df.columns[-1]]]
    output_path = f"data/pruned/pruned_{file_name}"
    df_pruned.to_csv(output_path, index=False)
    print(f"[SUCCESS] Saved pruned dataset to {output_path}")