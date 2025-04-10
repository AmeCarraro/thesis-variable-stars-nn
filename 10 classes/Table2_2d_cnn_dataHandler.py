import pandas as pd
from sklearn.model_selection import KFold
import os

# Set the path to the data directory
data_path = "C:\\Users\\carra\\Unipd\\tesi\\data\\"

# Define column names as per the ZTF catalog
cols = [
    "ID", "SourceID", "RAdeg", "DEdeg", "Per", "R21", "phi21", "T_0", "gmag", "rmag",
    "Per_g", "Per_r", "Num_g", "Num_r", "R21_g", "R21_r", "phi21_g", "phi21_r", "R^2_g",
    "R^2_r", "Amp_g", "Amp_r", "log(FAP_g)", "log(FAP_r)", "Type", "Delta_min_g",
    "Delta_min_r",
]

# Load the data from the text file, skipping header rows as required
df = pd.read_table(
    data_path + "Table2.txt",
    sep=' ',
    header=None,
    skipinitialspace=True,
    skiprows=34,
    names=cols,
)

# Apply filters: keep only objects with Num_g and Num_r > 50, R^2 > 0.3, and log(FAP) < -3
condition_num = (df["Num_g"] > 50) & (df["Num_r"] > 50)
condition_r2 = (df["R^2_g"] > 0.3) & (df["R^2_r"] > 0.3)
condition_fap = (df["log(FAP_g)"] < -3) & (df["log(FAP_r)"] < -3)
df = df[condition_num & condition_r2 & condition_fap]

# Replace 'CEPII' with 'CEP' in the Type column
df['Type'] = df['Type'].replace('CEPII', 'CEP')

# Select only the relevant columns for further analysis
df = df[["Per", "Amp_g", "Amp_r", "Type"]]

# Show class distribution before balancing
print("Class count BEFORE balancing:")
print(df['Type'].value_counts())

# Balance the dataset by sampling up to 1000 rows per class if available
balanced_df = df.groupby('Type').apply(lambda x: x.sample(n=1000, random_state=42) if len(x) > 1000 else x).reset_index(drop=True)

# Show class distribution after balancing and list remaining columns
print("Class count after balancing:")
print(balanced_df['Type'].value_counts())
print("Remaining columns in DataFrame:")
print(balanced_df.columns.tolist())

# Create a 5-fold cross-validation split and save train/test files for each fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_index, test_index) in enumerate(kf.split(balanced_df), 1):
    print(f"Fold {fold}")
    train_df = balanced_df.iloc[train_index]
    test_df = balanced_df.iloc[test_index]
    
    # Save the train and test splits to files for use in training the CNN-LSTM network
    train_df.to_csv(f"{data_path}cnn_lstm_train_fold_{fold}.txt", index=False, sep=' ', na_rep='NaN')
    test_df.to_csv(f"{data_path}cnn_lstm_test_fold_{fold}.txt", index=False, sep=' ', na_rep='NaN')
