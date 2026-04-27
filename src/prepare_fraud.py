import pandas as pd
from sklearn.utils import resample
import os

def prepare_data(filepath="data/creditcard.csv"):
    print("Loading Credit Card Fraud dataset...")
    df = pd.read_csv(filepath)
    
    print(f"Original shape: {df.shape}")
    print(f"Fraud cases: {df['Class'].sum()}")
    print(f"Normal cases: {(df['Class']==0).sum()}")

    # Balance the dataset (undersample majority class)
    fraud    = df[df['Class'] == 1]
    normal   = df[df['Class'] == 0]
    normal_downsampled = resample(normal, 
                                   replace=False,
                                   n_samples=len(fraud)*3,
                                   random_state=42)
    
    balanced = pd.concat([fraud, normal_downsampled])
    balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    os.makedirs("data", exist_ok=True)
    balanced.to_csv("data/fraud_data.csv", index=False)
    print(f"Saved balanced dataset: {balanced.shape}")
    print(f"Class distribution:\n{balanced['Class'].value_counts()}")

if __name__ == "__main__":
    prepare_data()