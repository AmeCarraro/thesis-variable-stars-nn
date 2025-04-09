import pandas as pd
from sklearn.model_selection import KFold
import os

# Path to the data file
data_path = "C:\\Users\\carra\\Unipd\\tesi\\data\\"

# Column definitions as in the ZTF catalog
cols = [
    "ID", "SourceID", "RAdeg", "DEdeg", "Per", "R21", "phi21", "T_0", "gmag", "rmag",
    "Per_g", "Per_r", "Num_g", "Num_r", "R21_g", "R21_r", "phi21_g", "phi21_r", "R^2_g",
    "R^2_r", "Amp_g", "Amp_r", "log(FAP_g)", "log(FAP_r)", "Type", "Delta_min_g",
    "Delta_min_r",
]

# Load data into a DataFrame
df = pd.read_table(
    data_path + "Table2.txt",
    sep=' ',
    header=None,
    skipinitialspace=True,
    skiprows=34,
    names=cols,
)

# Data cleaning:
# - Replace 'CEPII' with 'CEP'
df['Type'] = df['Type'].replace('CEPII', 'CEP')
# - Group Mira and SR under LPV
df['Type'] = df['Type'].replace({'Mira': 'LPV', 'SR': 'LPV'})

# Filtering conditions:
# - Only objects with Num_g and Num_r > 50
# - R^2_g and R^2_r > 0.3
# - log(FAP_g) and log(FAP_r) < -3
condition_num = (df["Num_g"] > 50) & (df["Num_r"] > 50)
condition_r2 = (df["R^2_g"] > 0.3) & (df["R^2_r"] > 0.3)
condition_fap = (df["log(FAP_g)"] < -3) & (df["log(FAP_r)"] < -3)
df = df[condition_num & condition_r2 & condition_fap]

# Keep only the necessary columns to reconstruct the light curve:

df = df[["Per", "T_0", "gmag", "rmag", "Amp_g", "Amp_r", "R^2_g", "R^2_r", "Type"]]


filtered_df = df[~(df == 0).any(axis=1)]

# Select only the 5 classes: CEP, DSCT, EA, RR, LPV
selected_classes = ['CEP', 'DSCT', 'EA', 'RR', 'LPV']
filtered_df = filtered_df[filtered_df['Type'].isin(selected_classes)]

print("Class count BEFORE saving (all samples):")
print(filtered_df['Type'].value_counts())
print("Remaining columns in DataFrame:")
print(filtered_df.columns.tolist())


balanced_df = filtered_df.groupby('Type').apply(
    lambda x: x.sample(n=30000, random_state=42) if len(x) > 30000 else x
).reset_index(drop=True)

print("Class count AFTER limiting to 30,000 per class:")
print(balanced_df['Type'].value_counts())


kf = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_index, test_index) in enumerate(kf.split(balanced_df), 1):
    print(f"\nFold {fold}")
    train_df = balanced_df.iloc[train_index]
    test_df = balanced_df.iloc[test_index]
    
    print(f"Training set class counts (Fold {fold}):")
    print(train_df['Type'].value_counts())
    print(f"Test set class counts (Fold {fold}):")
    print(test_df['Type'].value_counts())
    
    train_df.to_csv(f"{data_path}inf_cnn_train_fold_{fold}.txt", index=False, sep=' ', na_rep='NaN')
    test_df.to_csv(f"{data_path}inf_cnn_test_fold_{fold}.txt", index=False, sep=' ', na_rep='NaN')
