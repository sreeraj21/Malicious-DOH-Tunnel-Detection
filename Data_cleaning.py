import pandas as pd
import numpy as np
import os

# Create the output directory if it doesn't exist
os.makedirs('data/processed', exist_ok=True)

datasets = ['l1-total-add.csv', 'l2-total-add.csv', 'l3-total-add.csv']
# Note: Update these column names if the script throws a key error
leakage_columns = ['SourceIP', 'DestinationIP', 'SourcePort', 'DestinationPort', 'TimeStamp']

for file_name in datasets:
    print(f"\n{'='*40}")
    print(f"Processing {file_name}...")
    file_path = f"Data/{file_name}"   
    
    # 1. Load data
    df = pd.read_csv(file_path)
    print(f"Original Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # 2. Drop leakage columns
    cols_to_drop = [col for col in leakage_columns if col in df.columns]
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
        print(f"Dropped Leakage Columns: {cols_to_drop}")
    
    # 3. Clean Missing/Infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    missing_count = df.isnull().sum().sum()
    print(f"Total Missing/Inf Values Found: {missing_count}")
    
    if missing_count > 0:
        df.dropna(inplace=True)
    
    # 4. Print class distribution
    print(f"\nClass Distribution for {file_name}:")
    print(df.iloc[:, -1].value_counts())
    
    # 5. Save processed file
    print(f"\nFinal Cleaned Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    output_path = f"Data/processed/cleaned_{file_name}"
    df.to_csv(output_path, index=False)
    print(f"Saved perfectly clean dataset to {output_path}")