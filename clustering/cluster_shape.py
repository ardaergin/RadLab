import pandas as pd
import numpy as np
from statsmodels.regression.mixed_linear_model import MixedLM
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import dtw

##### Setup #####
# Reading the 8 Experiment CSV files and combine all datasets into one DataFrame:
path = "data/data_ilr_transformed/"
file_list = [f"{path}d_study{i}_long.csv" for i in range(1, 9)]
df = pd.concat([pd.read_csv(file) for file in file_list], ignore_index=True)
print(f"Combined dataset shape: {df.shape}")

# Defining isometric log-ratio transformed dependent variables:
ilr_columns = ['ilr1', 'ilr2', 'ilr3']
# Defining control variables:
control_columns = ['excluded', 'injustice', 'personal', 'violence']
# Covariates
covariate_columns = ['gender', 'age', 'condition']
# Handling NA values in covariates:
for covariate in covariate_columns: 
    df = df[df[covariate].notna()]

# Residualize ILR variables w.r.t. time and controls
for col in ['ilr1', 'ilr2', 'ilr3']:
    model = MixedLM.from_formula(
        f"{col} ~ relative_time + excluded + injustice + personal + violence",
        groups="Experiment",
        data=df
    )
    result = model.fit()
    df[f'{col}_residual'] = df[col] - result.fittedvalues

# Now we have ilr1_residual, ilr2_residual, ilr3_residual as our primary features.

# Construct time series data:
grouped = df.groupby("ID")
participant_ids = list(grouped.groups.keys())

# Determine a consistent length (e.g., max_length = 15)
max_length = 15  
time_series_data = []

for pid in participant_ids:
    p_data = df[df['ID']==pid].sort_values('time')[['ilr1_residual','ilr2_residual','ilr3_residual']].values
    # Pad or truncate
    length = p_data.shape[0]
    if length < max_length:
        p_data = np.pad(p_data, ((0,max_length-length),(0,0)), 'constant', constant_values=np.nan)
    else:
        p_data = p_data[:max_length]
    time_series_data.append(p_data)

time_series_data = np.array(time_series_data)  # shape: (n_samples, max_length, 3)

# Clustering with DTW on ILR residual time series
n_clusters = 3
model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", random_state=0)
labels = model.fit_predict(time_series_data)

df["cluster"] = df["ID"].map(dict(zip(participant_ids, labels)))

# Evaluate with a DTW-based validity index (Dunn index, silhouette with precomputed DTW distances, etc.)
def dtw_distance(ts1, ts2):
    # ts1 and ts2 shape: (max_length, 3)
    # Reduce dimensionality by norm if desired, or keep as 3D by summing distances of each dimension
    ts1_reduced = np.linalg.norm(ts1, axis=1)
    ts2_reduced = np.linalg.norm(ts2, axis=1)
    return dtw(ts1_reduced, ts2_reduced)

# Compute DTW distance matrix
n_samples = len(participant_ids)
distance_matrix = np.zeros((n_samples, n_samples))
for i in range(n_samples):
    for j in range(i+1, n_samples):
        dist = dtw_distance(time_series_data[i], time_series_data[j])
        distance_matrix[i,j] = dist
        distance_matrix[j,i] = dist

# Now compute Dunn index or custom silhouette using this DTW-based distance_matrix if desired.
from sklearn.metrics import silhouette_score, davies_bouldin_score

# After you run your main clustering and have `df` with `cluster`, `time_series_data`, and `labels`:
# Assume you have a variable `participant_ids` which stores the participant IDs in the exact order they appear in `time_series_data` and `labels`.
# If you don't have it yet, you can create it before clustering:
grouped = df.groupby("ID")
participant_ids = list(grouped.groups.keys())  # Ensure this matches the order used to build time_series_data
id_to_index = {pid: i for i, pid in enumerate(participant_ids)}

experiments = df['Experiment'].unique()

for exp in experiments:
    # Get participant IDs belonging to this experiment
    exp_ids = df.loc[df['Experiment'] == exp, 'ID'].unique()
    
    # Extract their original time series directly from df, no padding
    participant_series = []
    exp_labels = []
    
    for pid in exp_ids:
        # Extract participant's original data (sorted by time)
        p_data = df[df['ID'] == pid].sort_values('time')[ ].values
        
        # Remove any rows that contain NaNs if present
        # (If your original dataset had no NaNs and you only introduced them via padding,
        # this step might not be necessary. But we keep it just in case.)
        p_data = p_data[~np.isnan(p_data).any(axis=1)]
        
        if len(p_data) == 0:
            # If a participant has no valid rows left, skip them (or handle differently)
            continue
        
        participant_series.append(p_data)
        
        # Retrieve the cluster label assigned to this participant
        # and store in exp_labels
        p_index = id_to_index[pid]
        exp_labels.append(labels[p_index])
        
    # Find the minimum length of sequences in this experiment
    lengths = [arr.shape[0] for arr in participant_series]
    min_len = min(lengths)
    print(f"Minimum length in Experiment {exp}: {min_len}")
    
    # Truncate all series to the minimal length
    truncated_series = [arr[:min_len] for arr in participant_series]
    
    # Flatten each participant's truncated time series into a single vector
    # Resulting shape: (n_participants, min_len * n_features)
    data_matrix = np.array([t.flatten() for t in truncated_series])
    exp_labels = np.array(exp_labels)
    
    # Compute Silhouette and Davies-Bouldin Scores
    # Note: These metrics are not DTW-based, but a simple Euclidean flattening.
    # If you want a DTW-based approach, you'd need a custom scoring function.
    try:
        sil = silhouette_score(data_matrix, exp_labels)
    except ValueError:
        sil = np.nan
    try:
        db = davies_bouldin_score(data_matrix, exp_labels)
    except ValueError:
        db = np.nan
    
    print(f"Experiment {exp}: Silhouette Score = {sil}, Davies-Bouldin Score = {db}")


# Interpretation
# Once clustered, interpret clusters by looking back at ILR trajectories or even inversely mapping ILR residuals back to original compositional space if needed.
