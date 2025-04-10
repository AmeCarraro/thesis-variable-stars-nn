import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer, f1_score

# Define file paths for features and labels
data_path = "C:\\Users\\carra\\Unipd\\tesi\\data\\"
train_feature_file = os.path.join(data_path, "X_df.csv")
y_file = os.path.join(data_path, "y_df.csv")

# (Optional) List of feature column names for reference
column_names_x = [
    "oid", "MHPS_ratio_g", "MHPS_low_g", "MHPS_high_g", "MHPS_non_zero_g", "MHPS_PN_flag_g",
    "MHPS_ratio_r", "MHPS_low_r", "MHPS_high_r", "MHPS_non_zero_r", "MHPS_PN_flag_r",
    "Multiband_period", "PPE", "Period_band_g", "delta_period_g", "Period_band_r", "delta_period_r",
    "GP_DRW_sigma_g", "GP_DRW_tau_g", "GP_DRW_sigma_r", "GP_DRW_tau_r", "Psi_CS_g", "Psi_eta_g",
    "Psi_CS_r", "Psi_eta_r", "Harmonics_mag_1_g", "Harmonics_mag_2_g", "Harmonics_mag_3_g",
    "Harmonics_mag_4_g", "Harmonics_mag_5_g", "Harmonics_mag_6_g", "Harmonics_mag_7_g",
    "Harmonics_phase_2_g", "Harmonics_phase_3_g", "Harmonics_phase_4_g", "Harmonics_phase_5_g",
    "Harmonics_phase_6_g", "Harmonics_phase_7_g", "Harmonics_mse_g", "Harmonics_chi_g",
    "Harmonics_mag_1_r", "Harmonics_mag_2_r", "Harmonics_mag_3_r", "Harmonics_mag_4_r",
    "Harmonics_mag_5_r", "Harmonics_mag_6_r", "Harmonics_mag_7_r", "Harmonics_phase_2_r",
    "Harmonics_phase_3_r", "Harmonics_phase_4_r", "Harmonics_phase_5_r", "Harmonics_phase_6_r",
    "Harmonics_phase_7_r", "Harmonics_mse_r", "Harmonics_chi_r", "iqr_g", "iqr_r",
    "Power_rate_1/4", "Power_rate_1/3", "Power_rate_1/2", "Power_rate_2", "Power_rate_3",
    "Power_rate_4", "Amplitude_g", "AndersonDarling_g", "Autocor_length_g", "Beyond1Std_g",
    "Con_g", "Eta_e_g", "Gskew_g", "MaxSlope_g", "Mean_g", "Meanvariance_g", "MedianAbsDev_g",
    "MedianBRP_g", "PairSlopeTrend_g", "PercentAmplitude_g", "Q31_g", "Rcs_g", "Skew_g",
    "SmallKurtosis_g", "Std_g", "StetsonK_g", "Pvar_g", "ExcessVar_g", "SF_ML_amplitude_g",
    "SF_ML_gamma_g", "IAR_phi_g", "LinearTrend_g", "Amplitude_r", "AndersonDarling_r",
    "Autocor_length_r", "Beyond1Std_r", "Con_r", "Eta_e_r", "Gskew_r", "MaxSlope_r", "Mean_r",
    "Meanvariance_r", "MedianAbsDev_r", "MedianBRP_r", "PairSlopeTrend_r", "PercentAmplitude_r",
    "Q31_r", "Rcs_r", "Skew_r", "SmallKurtosis_r", "Std_r", "StetsonK_r", "Pvar_r", "ExcessVar_r",
    "SF_ML_amplitude_r", "SF_ML_gamma_r", "IAR_phi_r", "LinearTrend_r"
]

# Load features and labels from CSV files
features_df = pd.read_csv(train_feature_file, sep=',', decimal='.', header=0)
column_names_y = ["oid", "class"]
features_df_y = pd.read_csv(y_file, sep=',', header=0)

# Merge the features and labels on 'oid'
merged_df = pd.merge(features_df, features_df_y, on='oid', how='inner')
print(merged_df.columns.tolist())
print(merged_df.head(1))

# Remove columns containing "_g"
features_to_keep = [c for c in merged_df.columns if "_g" not in c]
merged_df = merged_df[features_to_keep]

# Filter rows with specific classes and create a copy
filtered_df = merged_df[merged_df['class'].isin(['CEP', 'RR', 'RRc', 'DSCT'])].copy()

# Map class labels to numeric values
label_mapping = { 
    'CEP': 0,
    'RR': 1,
    'RRc': 2,
    'DSCT': 3,
}
filtered_df['Label'] = filtered_df['class'].map(label_mapping)
print(filtered_df["Label"])

# Split into features (X) and target (y)
X_train_df = filtered_df.drop(columns=['oid', 'class', 'Label'])
y_train = filtered_df['Label']

# Remove features that are too highly correlated (>0.9)
corr_matrix = X_train_df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop_corr = [col for col in upper.columns if any(upper[col] > 0.9)]
print(f"Features dropped due to high correlation (> 0.9): {to_drop_corr}")
X_train_df = X_train_df.drop(columns=to_drop_corr, errors='ignore')

# Define f1 weighted scorer for feature selection
f1_weighted_scorer = make_scorer(f1_score, average='weighted')
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Perform forward sequential feature selection to choose top 30 features
sfs_30 = SequentialFeatureSelector(
    estimator=clf,
    n_features_to_select=30,
    direction='forward',
    scoring=f1_weighted_scorer,
    cv=StratifiedKFold(n_splits=5),
    n_jobs=-1
)
sfs_30.fit(X_train_df, y_train)

# Get selected feature names
selected_features_mask = sfs_30.get_support()
selected_features = X_train_df.columns[selected_features_mask].tolist()

print("\n** Top 30 selected features **")
for f in selected_features:
    print(f)

# Train final RandomForest on the selected features
final_rf = RandomForestClassifier(n_estimators=100, random_state=42)
final_rf.fit(X_train_df[selected_features], y_train)

# Extract and print feature importances
importances = final_rf.feature_importances_
feat_import_df = pd.DataFrame({
    'feature': selected_features,
    'importance': importances
}).sort_values(by='importance', ascending=False)

print("\n** Top 30 features sorted by importance **")
for i, row in feat_import_df.iterrows():
    print(f"{row['feature']}: {row['importance']}")
