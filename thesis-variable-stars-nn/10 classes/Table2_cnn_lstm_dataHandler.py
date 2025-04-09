import pandas as pd
from sklearn.model_selection import KFold
import os

# Define the path to the data directory
data_path = "C:\\Users\\carra\\Unipd\\tesi\\data\\"

# Define column names as specified in the ZTF variables catalog
cols = [
    "ID", "SourceID", "RAdeg", "DEdeg", "Per", "R21", "phi21", "T_0", "gmag", "rmag",
    "Per_g", "Per_r", "Num_g", "Num_r", "R21_g", "R21_r", "phi21_g", "phi21_r", "R^2_g",
    "R^2_r", "Amp_g", "Amp_r", "log(FAP_g)", "log(FAP_r)", "Type", "Delta_min_g",
    "Delta_min_r",
]

# Load the data from the text file into a DataFrame, skipping the first 34 header lines
df = pd.read_table(
    data_path + "Table2.txt",
    sep=' ',
    header=None,
    skipinitialspace=True,
    skiprows=34,
    names=cols,
)

# Replace 'CEPII' with 'CEP' in the 'Type' column
df['Type'] = df['Type'].replace('CEPII', 'CEP')

# Define filtering conditions based on number of observations, R^2 and FAP values
condition_num = (df["Num_g"] > 50) & (df["Num_r"] > 50)
condition_r2 = (df["R^2_g"] > 0.3) & (df["R^2_r"] > 0.3)
condition_fap = (df["log(FAP_g)"] < -3) & (df["log(FAP_r)"] < -3)

# Apply filters to keep only the rows that meet all conditions
df = df[condition_num & condition_r2 & condition_fap]

# Select a subset of columns that will be used for further analysis
df = df[["Per", "R21", "phi21", "rmag", "Amp_r", "Type"]]

# Remove any rows that contain 0 in any column to avoid invalid data
filtered_df = df[~(df == 0).any(axis=1)]
print("Class count BEFORE balancing:")
print(filtered_df['Type'].value_counts())

# Balance the dataset by sampling at most 1000 rows per class
balanced_df = filtered_df.groupby('Type').apply(
    lambda x: x.sample(n=1000, random_state=42) if len(x) > 1000 else x
).reset_index(drop=True)
print("Class count after balancing:")
print(balanced_df['Type'].value_counts())
print("Remaining columns in DataFrame:")
print(balanced_df.columns.tolist())

# Set up a 5-fold cross-validation split with shuffling
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_index, test_index) in enumerate(kf.split(balanced_df), 1):
    print(f"Fold {fold}")
    train_df = balanced_df.iloc[train_index]
    test_df = balanced_df.iloc[test_index]
    
    # Print the class counts for training and test sets for the current fold
    print(f"Training set class counts (Fold {fold}):")
    print(train_df['Type'].value_counts())
    print(f"Test set class counts (Fold {fold}):")
    print(test_df['Type'].value_counts())
    
    # Save the training and test sets to text files using space as the separator
    train_df.to_csv(f"{data_path}lstm_train_fold_{fold}.txt", index=False, sep=' ', na_rep='NaN')
    test_df.to_csv(f"{data_path}lstm_test_fold_{fold}.txt", index=False, sep=' ', na_rep='NaN')
