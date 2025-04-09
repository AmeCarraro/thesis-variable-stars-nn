import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer, f1_score
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Set file paths for the multiclass datasets
data_path = "C:\\Users\\carra\\Unipd\\tesi\\data\\"
train_feature_file = os.path.join(data_path, "HIDDENX_df_multiclass.csv")
y_file = os.path.join(data_path, "HIDDENy_df_multiclass.csv")

# Define column names for reference (not used later)
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

# Load the feature and label data
features_df = pd.read_csv(train_feature_file, sep=',', decimal='.', header=0)
column_names_y = ["oid", "class"]
features_df_y = pd.read_csv(y_file, sep=',', header=0)

# Merge on 'oid' and show initial data overview
merged_df = pd.merge(features_df, features_df_y, on='oid', how='inner')
print(merged_df.columns.tolist())
print(merged_df.head(1))

# Keep only columns that do not contain "_g" and filter for selected classes
features_to_keep = [c for c in merged_df.columns if "_g" not in c]
merged_df = merged_df[features_to_keep]
filtered_df = merged_df[merged_df['class'].isin(['CEP', 'RR', 'RRc', 'DSCT'])].copy()

# Map class labels to integers and save them to a file
label_mapping = {'CEP': 0, 'RR': 1, 'RRc': 2, 'DSCT': 3}
filtered_df['Label'] = filtered_df['class'].map(label_mapping)
label_path = os.path.join(data_path, "labelTE_30_svm.txt")
filtered_df['Label'].to_csv(label_path, index=False, header=False)

# Drop non-feature columns
filtered_df = filtered_df.drop(columns=['oid', 'class', 'Label'])

# Convert object columns to float and fill missing values
for col in filtered_df.columns:
    if filtered_df[col].dtype == 'object':
        filtered_df[col] = filtered_df[col].str.replace(',', '.', regex=False).astype(float)
filtered_df = filtered_df.fillna(0)

# Impute missing values (zeros) using SVR for each feature
imputed_df = filtered_df.copy()
feature_columns = imputed_df.columns.tolist()

for feature in feature_columns:
    print(f"Processing feature: {feature}")
    missing_indices = imputed_df[imputed_df[feature] == 0].index
    observed_indices = imputed_df[imputed_df[feature] != 0].index
    if len(observed_indices) < 1:
        print(f"  Not enough observed data to impute {feature}. Skipping.")
        continue
    if len(missing_indices) == 0:
        continue
    X_train = imputed_df.loc[observed_indices, feature_columns].drop(columns=[feature])
    y_train = imputed_df.loc[observed_indices, feature]
    X_test = imputed_df.loc[missing_indices, feature_columns].drop(columns=[feature])
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    svr = SVR()
    svr.fit(X_train_scaled, y_train)
    y_pred = svr.predict(X_test_scaled)
    imputed_df.loc[missing_indices, feature] = y_pred

# Select specific features for the final output
feature_names = [
    "Multiband_period",
    "Period_band_r",
    "Harmonics_mag_2_r",
    "Harmonics_phase_2_r",
    "MHPS_low_r",
    "IAR_phi_r",
    "GP_DRW_tau_r",
    "Harmonics_mag_1_r",
    "iqr_r",
    "MHPS_high_r",
    "Skew_r",
    "Harmonics_phase_3_r",
    "Power_rate_1/3",
    "MHPS_non_zero_r",
    "Mean_r",
    "Psi_eta_r",
    "Psi_CS_r",
    "Harmonics_mse_r",
    "PPE",
    "Eta_e_r",
    "StetsonK_r",
    "Harmonics_mag_7_r",
    "SmallKurtosis_r",
    "Power_rate_4",
    "delta_period_r",
    "Beyond1Std_r",
    "MaxSlope_r",
    "Harmonics_phase_6_r",
    "LinearTrend_r",
    "Pvar_r"
]
imputed_df = imputed_df[feature_names]
print(imputed_df.columns.tolist())

# Save the imputed features to text and Excel files
output_file = os.path.join(data_path, "featureTE_30_svm.txt")
imputed_df.to_csv(output_file, sep='\t', index=False, header=False)
print(f"Imputed features file saved at: {output_file}")

excel_path = os.path.splitext(output_file)[0] + ".xlsx"
imputed_df.to_excel(excel_path, index=False)
print(f"Imputed Excel file saved at: {excel_path}")

print(imputed_df.head())
