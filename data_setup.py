#file: data_setup.py
#
import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Configuration
DATA_DIR = "data"
PROCESSED_PATH = os.path.join(DATA_DIR, "processed_tensors.pt")

# FD001 sensors that are constant (variance=0) and should be dropped
# We also drop Op settings because they are constant in FD001
DROP_COLS = ['s1', 's5', 's10', 's16', 's18', 's19', 'op1', 'op2', 'op3']

def load_dataset(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}")
        
    col_names = ['unit', 'cycle', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]
    df = pd.read_csv(path, sep=r'\s+', header=None, names=col_names)
    
    # Compute RUL
    max_cycles = df.groupby('unit')['cycle'].transform('max')
    df['RUL'] = max_cycles - df['cycle']
    
    return df

def process_data():
    # 1. Load Train (FD001) and OOD (FD002)
    print("Loading datasets...")
    df_train_full = load_dataset('train_FD001.txt')
    df_ood_full = load_dataset('train_FD002.txt')
    
    # 2. Clean Data (Drop constant sensors and op settings)
    # We apply FD001's logic to FD002 to ensure feature alignment
    df_train_clean = df_train_full.drop(columns=DROP_COLS)
    df_ood_clean = df_ood_full.drop(columns=DROP_COLS)
    
    # 3. Prepare Arrays (Drop Unit ID)
    X_full = df_train_clean.drop(columns=['unit', 'cycle', 'RUL']).values
    y_full = df_train_clean['RUL'].values
    
    # For OOD, we just take a random subsample (e.g., 2000 snapshots) to keep evaluation fast
    df_ood_sample = df_ood_clean.sample(n=2000, random_state=42)
    X_ood = df_ood_sample.drop(columns=['unit', 'cycle', 'RUL']).values
    y_ood = df_ood_sample['RUL'].values

    # 4. Train/Test Split (IID)
    X_train, X_test_iid, y_train, y_test_iid = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42, shuffle=True
    )
    
    # 5. Normalization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test_iid = scaler.transform(X_test_iid)
    X_ood = scaler.transform(X_ood) # Note: FD002 normalized by FD001 stats!
    
    # 6. Create Noisy Set (Aleatoric)
    # Add noise to IID test set. 
    # Noise scale = 1.0 (since data is normalized, this is 1 standard deviation)
    X_test_noisy = X_test_iid + np.random.normal(0, 1.0, X_test_iid.shape)
    y_test_noisy = y_test_iid
    
    # 7. Save
    data_dict = {
        "X_train": torch.tensor(X_train, dtype=torch.float32),
        "y_train": torch.tensor(y_train, dtype=torch.float32),
        "X_test_iid": torch.tensor(X_test_iid, dtype=torch.float32),
        "y_test_iid": torch.tensor(y_test_iid, dtype=torch.float32),
        "X_test_ood": torch.tensor(X_ood, dtype=torch.float32),
        "y_test_ood": torch.tensor(y_ood, dtype=torch.float32),
        "X_test_noisy": torch.tensor(X_test_noisy, dtype=torch.float32),
        "y_test_noisy": torch.tensor(y_test_noisy, dtype=torch.float32)
    }
    
    torch.save(data_dict, PROCESSED_PATH)
    print(f"Saved processed tensors to {PROCESSED_PATH}")
    
    # 8. Output Stats for Cost Matrix Design
    print("\n--- RUL Statistics (Training Set) ---")
    print(pd.Series(y_train).describe())
    
    # Return stats for discussion
    return pd.Series(y_train).describe()

if __name__ == "__main__":
    process_data()
    