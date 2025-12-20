# file: data_setup.py
#
import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Configuration
RAW_PATH = "data/train_FD001.txt"
PROCESSED_PATH = "data/processed_tensors.pt"

def load_and_process_data():
    """
    Loads local raw data, computes RUL, removes Asset ID, and shuffles.
    Returns: X (features), y (RUL targets)
    """
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"Please manually place the file '{RAW_PATH}' in the data directory.")

    # 1. Load Data
    # Columns: Unit, Cycle, Op1, Op2, Op3, Sensor1...Sensor21
    col_names = ['unit', 'cycle', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]
    print(f"Loading data from {RAW_PATH}...")
    df = pd.read_csv(RAW_PATH, sep=r'\s+', header=None, names=col_names)
    
    # 2. Compute RUL (Remaining Useful Life)
    # RUL = Max_Cycle_for_Unit - Current_Cycle
    max_cycles = df.groupby('unit')['cycle'].transform('max')
    df['RUL'] = max_cycles - df['cycle']
    
    # 3. The "Snapshot" Transformation
    # Remove Unit ID, treat rows as independent.
    drop_cols = ['unit'] 
    df_snapshot = df.drop(columns=drop_cols)
    
    # Separate Features and Target
    X = df_snapshot.drop(columns=['RUL']).values
    y = df_snapshot['RUL'].values
    
    return X, y

def save_tensors(X, y):
    """Normalizes features, splits into train/test, and saves as PyTorch tensors."""
    # 1. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    # 2. Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 3. Convert to Tensors
    data_dict = {
        "X_train": torch.tensor(X_train, dtype=torch.float32),
        "y_train": torch.tensor(y_train, dtype=torch.float32),
        "X_test": torch.tensor(X_test, dtype=torch.float32),
        "y_test": torch.tensor(y_test, dtype=torch.float32)
    }
    
    # 4. Save
    torch.save(data_dict, PROCESSED_PATH)
    print(f"Processed data saved to {PROCESSED_PATH}")
    print(f"Training shapes: X={data_dict['X_train'].shape}, y={data_dict['y_train'].shape}")

if __name__ == "__main__":
    X, y = load_and_process_data()
    save_tensors(X, y)
    