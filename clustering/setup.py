import pandas as pd

##### Setup #####
print("===== SETUP =====")

df = pd.read_csv("data/data_ilr_transformed/combined_data__resid_with_time.csv")
print(f"Combined dataset shape: {df.shape}")
print("Unique time points:", df["time"].unique())

# Variables
original_DVs = ["ina", "na", "nna", "enna"]
control_columns = ['excluded', 'injustice', 'personal', 'violence']
covariate_columns = ['gender', 'age']
ilr_columns = ['ilr1', 'ilr2', 'ilr3']
ilr_resid_columns = ['ilr1_residual', 'ilr2_residual', 'ilr3_residual']

# Handling NA values in covariates:
for covariate in covariate_columns: 
    df = df[df[covariate].notna()]

from sklearn.preprocessing import RobustScaler
df[['age']] = RobustScaler().fit_transform(
    df[['age']]
)

