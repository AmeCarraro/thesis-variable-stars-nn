import pandas as pd
from sklearn.model_selection import KFold
import os

# Define data path and column names
data_path = "C:\\Users\\carra\\Unipd\\tesi\\data\\"
cols = [
    "ID", "SourceID", "RAdeg", "DEdeg", "Per", "R21", "phi21", "T_0", "gmag", "rmag",
    "Per_g", "Per_r", "Num_g", "Num_r", "R21_g", "R21_r", "phi21_g", "phi21_r", "R^2_g",
    "R^2_r", "Amp_g", "Amp_r", "log(FAP_g)", "log(FAP_r)", "Type", "Delta_min_g",
    "Delta_min_r",
]

# Load data and skip header rows
df = pd.read_table(
    data_path + "Table2.txt",
    sep=' ',
    header=None,
    skipinitialspace=True,
    skiprows=34,
    names=cols,
)

# Clean data: replace 'CEPII' with 'CEP' and merge 'Mira' and 'SR' into 'LPV'
df['Type'] = df['Type'].replace('CEPII', 'CEP')
df['Type'] = df['Type'].replace({'Mira': 'LPV', 'SR': 'LPV'})

# Filter based on number of observations, fit quality, and FAP
condition_num = (df["Num_g"] > 50) & (df["Num_r"] > 50)
condition_r2 = (df["R^2_g"] > 0.3) & (df["R^2_r"] > 0.3)
condition_fap = (df["log(FAP_g)"] < -3) & (df["log(FAP_r)"] < -3)
df = df[condition_num & condition_r2 & condition_fap]

# Keep only columns needed for light curve reconstruction
df = df[["Per", "R21", "phi21", "rmag", "Amp_r", "Type"]]

# Remove rows with any zero values
filtered_df = df[~(df == 0).any(axis=1)]

# Select only the specified 5 classes
selected_classes = ['CEP', 'DSCT', 'EA', 'RR', 'LPV']
filtered_df = filtered_df[filtered_df['Type'].isin(selected_classes)]

print("Class count BEFORE balancing:")
print(filtered_df['Type'].value_counts())

# Limit each class to at most 1000 samples
balanced_df = filtered_df.groupby('Type').apply(
    lambda x: x.sample(n=1000, random_state=42) if len(x) > 1000 else x
).reset_index(drop=True)

print("Class count AFTER balancing:")
print(balanced_df['Type'].value_counts())
print("Remaining columns in DataFrame:")
print(balanced_df.columns.tolist())

# 5-fold cross-validation and save train/test files with prefix '5_lstm'
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_index, test_index) in enumerate(kf.split(balanced_df), 1):
    print(f"\nFold {fold}")
    train_df = balanced_df.iloc[train_index]
    test_df = balanced_df.iloc[test_index]
    
    print(f"Training set class counts (Fold {fold}):")
    print(train_df['Type'].value_counts())
    print(f"Test set class counts (Fold {fold}):")
    print(test_df['Type'].value_counts())
    
    train_df.to_csv(f"{data_path}5_lstm_train_fold_{fold}.txt", index=False, sep=' ', na_rep='NaN')
    test_df.to_csv(f"{data_path}5_lstm_test_fold_{fold}.txt", index=False, sep=' ', na_rep='NaN')
