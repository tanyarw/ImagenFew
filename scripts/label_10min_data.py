import pandas as pd
import os

def label_10min_data():
    print("Loading 10-min rainfall data...")
    df10 = pd.read_csv("data/rainfall/train/rainfall_10min.csv")
    
    print("Loading 5-min train dataset (with labels)...")
    df5 = pd.read_csv("data/rainfall/train/train_dataset.csv")
    
    # df5 has columns: datetime, avg_rainfall, block_id, gmm_regime
    # We only need the labels and block_id
    df_labels = df5[['datetime', 'block_id', 'gmm_regime']].rename(columns={'datetime': 'date'})
    
    print("Merging labels into 10-min data...")
    df10_labeled = pd.merge(df10, df_labels, on='date', how='left')
    
    # Verify no missing labels
    missing = df10_labeled['gmm_regime'].isna().sum()
    if missing > 0:
        print(f"Warning: {missing} rows could not be matched and have NaN labels!")
        # Forward fill or drop? Usually if dates align perfectly this won't happen.
    
    out_path = "data/rainfall/train/rainfall_10min_labeled.csv"
    df10_labeled.to_csv(out_path, index=False)
    print(f"Saved labeled 10-min data to {out_path}")
    print(f"Shape: {df10_labeled.shape}")
    print(df10_labeled.head())

if __name__ == "__main__":
    label_10min_data()
