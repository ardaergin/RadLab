import pandas as pd
import numpy as np


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

##### Feature Engineering #####
# Calculate mean and standard deviation per participant (ID) for each ilr column
for col in ilr_columns:
    df[f'{col}_mean'] = df.groupby('ID')[col].transform('mean')
    df[f'{col}_std'] = df.groupby('ID')[col].transform('std')
    df[f'{col}_min'] = df.groupby('ID')[col].transform('min')
    df[f'{col}_max'] = df.groupby('ID')[col].transform('max')

# Additional time-based feature: time point relative to the total duration for each participant
df['relative_time'] = df.groupby('ID')['time'].transform(lambda x: x / x.max())

# Deviation from group
for col in ilr_columns:
    df[f'{col}_deviation'] = df[col] - df.groupby(['Experiment', 'time'])[col].transform('mean')

# Interaction Terms: Control Variables with Time
for col in control_columns:
    df[f'{col}_time_interaction'] = df[col] * df['time']

# Residualizing ilr variables
from statsmodels.regression.mixed_linear_model import MixedLM

for col in ilr_columns:
    model = MixedLM.from_formula(
        f"{col} ~ relative_time + excluded + injustice + personal + violence",
        groups="Experiment", 
        data=df
    )
    result = model.fit()
    df[f'{col}_residual'] = df[col] - result.fittedvalues
ilr_resid_columns = ['ilr1_residual', 'ilr2_residual', 'ilr3_residual']

# Moving Average or Exponential Moving Average (EMA)
window_size = 2 
for col in ilr_resid_columns:
    df[f'{col}_moving_avg'] = df.groupby('ID')[col].transform(lambda x: x.rolling(window=window_size, min_periods=1).mean())
    df[f'{col}_ema'] = df.groupby('ID')[col].transform(lambda x: x.ewm(span=window_size, adjust=False).mean())
ilr_moving_avg_cols = ['ilr1_residual_moving_avg', 'ilr2_residual_moving_avg', 'ilr3_residual_moving_avg']
ilr_ema_cols = ['ilr1_residual_ema', 'ilr2_residual_ema', 'ilr3_residual_ema']

# Fourier transformation
fourier_components = 1
# Compute Fourier transform for each 'ilr' variable per participant
for column_group in [ilr_moving_avg_cols, ilr_ema_cols]:
    for col in column_group:
        def compute_fourier(x):
            fft_vals = np.fft.fft(x)
            # Retain only the first N components (real and imaginary parts)
            return np.hstack([fft_vals.real[:fourier_components], fft_vals.imag[:fourier_components]])

        # Apply Fourier transformation and store as separate columns
        fourier_df = (
            df.groupby('ID')[col]
            .apply(lambda x: compute_fourier(x.values))
            .apply(pd.Series)
            .rename(columns=lambda i: f'{col}_fourier_{i+1}')
        )

        # Add Fourier features back to the dataframe
        df = df.join(fourier_df, on='ID')

##### Prepare Time Series Data for Clustering #####
time_series_data = []
grouped = df.groupby("ID")
feature_columns = [
    # 'ilr1', 'ilr2', 'ilr3', 
    # 'condition',
    'gender', 'age',
    # 'excluded', 'injustice', 'personal', 'violence',
    'ilr1_mean', 'ilr2_mean', 'ilr3_mean',
    'ilr1_std', 'ilr2_std', 'ilr3_std',
    'ilr1_min', 'ilr2_min', 'ilr3_min',
    'ilr1_max', 'ilr2_max', 'ilr3_max',
    'relative_time',
    'ilr1_deviation', 'ilr2_deviation', 'ilr3_deviation',
    # 'excluded_time_interaction', 'injustice_time_interaction', 'personal_time_interaction', 'violence_time_interaction',
    # 'ilr1_moving_avg', 'ilr2_moving_avg', 'ilr3_moving_avg', 
    # 'ilr1_ema', 'ilr2_ema', 'ilr3_ema',
    #'ilr1_residual', 'ilr2_residual', 'ilr3_residual',
    #'ilr1_fourier_1', 'ilr2_fourier_1', 'ilr3_fourier_1',
    # 'ilr1_fourier_2', 'ilr2_fourier_2', 'ilr3_fourier_2'
    # 'ilr1_moving_avg_fourier_1', 'ilr2_moving_avg_fourier_1', 'ilr3_moving_avg_fourier_1',
    # 'ilr1_moving_avg_fourier_2', 'ilr2_moving_avg_fourier_2', 'ilr3_moving_avg_fourier_2',
    # 'ilr1_residual_fourier_1', 'ilr2_residual_fourier_1', 'ilr3_residual_fourier_1',
    # 'ilr1_moving_avg_residual_fourier_1', 'ilr2_moving_avg_residual_fourier_1', 'ilr3_moving_avg_residual_fourier_1',
    #'ilr1_moving_avg_residual_fourier_2', 'ilr2_moving_avg_residual_fourier_2', 'ilr3_moving_avg_residual_fourier_2'
    #'ilr1_residual_moving_avg_fourier_1', 'ilr2_residual_moving_avg_fourier_1', 'ilr3_residual_moving_avg_fourier_1',
    'ilr1_residual_ema_fourier_1', 'ilr2_residual_ema_fourier_1', 'ilr3_residual_ema_fourier_1'
]
print(f"{len(feature_columns)} features used:")
print(feature_columns)



# Build time_series_data
time_series_data = []
participant_ids = []  # Track the order of participants
grouped = df.groupby("ID")

max_length = 15
for participant, group in grouped:
    group_sorted = group.sort_values("time")
    time_series = group_sorted[feature_columns].values
    
    if len(time_series) < max_length:
        time_series = np.pad(time_series, ((0, max_length - len(time_series)), (0, 0)), 'constant', constant_values=np.nan)
    else:
        time_series = time_series[:max_length]
    
    time_series_data.append(time_series)
    participant_ids.append(participant)

time_series_data = np.array(time_series_data)

# Clustering
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import dtw

n_clusters = 3
model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", random_state=0)
labels = model.fit_predict(time_series_data)
df["cluster"] = df["ID"].map(dict(zip(participant_ids, labels)))

print(df.groupby(['cluster', 'Experiment']).size().unstack(fill_value=0))
cluster_summary = df.groupby("cluster")[["ilr1", "ilr2", "ilr3", "excluded", "injustice", "personal", "violence"]].mean()
print(cluster_summary)

##### Evaluation (per experiment) #####
def dtw_distance_ignore_nans(ts1, ts2):
    mask = ~np.isnan(ts1).any(axis=1) & ~np.isnan(ts2).any(axis=1)
    if np.sum(mask) == 0:
        return 1e6
    ts1_clean = ts1[mask]
    ts2_clean = ts2[mask]
    ts1_reduced = np.linalg.norm(ts1_clean, axis=1)
    ts2_reduced = np.linalg.norm(ts2_clean, axis=1)
    return dtw(ts1_reduced, ts2_reduced)

def compute_pairwise_dtw_distances(subset_data):
    n_samples = subset_data.shape[0]
    dist_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            dist = dtw_distance_ignore_nans(subset_data[i], subset_data[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    return dist_matrix

def dunn_index(distance_matrix, subset_labels):
    unique_clusters = np.unique(subset_labels)
    cluster_diameters = []
    for c in unique_clusters:
        cluster_points = np.where(subset_labels == c)[0]
        if len(cluster_points) > 1:
            dist_within = distance_matrix[np.ix_(cluster_points, cluster_points)]
            cluster_diameters.append(dist_within.max())
        else:
            cluster_diameters.append(0.0)

    inter_cluster_distances = []
    for i in range(len(unique_clusters)):
        for j in range(i+1, len(unique_clusters)):
            c1 = unique_clusters[i]
            c2 = unique_clusters[j]
            points_c1 = np.where(subset_labels == c1)[0]
            points_c2 = np.where(subset_labels == c2)[0]
            dist_between = distance_matrix[np.ix_(points_c1, points_c2)].min()
            inter_cluster_distances.append(dist_between)

    if not inter_cluster_distances:
        return np.nan

    min_inter = np.min(inter_cluster_distances)
    max_intra = np.max(cluster_diameters)
    if max_intra == 0:
        return float('inf')
    return min_inter / max_intra

# Per-experiment evaluation
experiments = df['Experiment'].unique()
id_to_index = {pid: i for i, pid in enumerate(participant_ids)}

for exp in experiments:
    exp_ids = df.loc[df['Experiment'] == exp, 'ID'].unique()
    # Extract subset of time_series_data and labels
    exp_indices = [id_to_index[pid] for pid in exp_ids if pid in id_to_index]
    if len(exp_indices) < 2:
        print(f"Experiment {exp}: Not enough participants for evaluation.")
        continue

    subset_data = time_series_data[exp_indices]
    subset_labels = labels[exp_indices]

    exp_distance_matrix = compute_pairwise_dtw_distances(subset_data)
    exp_dunn = dunn_index(exp_distance_matrix, subset_labels)
    print(f"Experiment {exp}: Dunn Index = {exp_dunn}")


##### Plotting #####
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

n_clusters = 3
alpha = 0.1  # Smoothing factor for EMA
savgol_window = 3  # Window length for Savitzky-Golay filter
savgol_polyorder = 1  # Polynomial order for Savitzky-Golay filter

for cluster in range(n_clusters):
    cluster_data = df[df['cluster'] == cluster]
    
    # mean_relative = cluster_data.groupby('relative_time')[['ilr1', 'ilr2', 'ilr3']].mean()
    mean_relative = cluster_data.groupby('relative_time')[['ina', 'na', 'nna', 'enna']].mean()
    
    # Apply Exponential Moving Average (EMA) smoothing
    mean_ilr_smooth_ema = mean_relative.apply(lambda x: x.ewm(alpha=alpha).mean())
    
    # Plot the original and smoothed series for each ILR variable in the cluster
    plt.figure(figsize=(12, 6))
    
    # Plot EMA smoothed series3
    # plt.plot(mean_relative.index, mean_ilr_smooth_ema["ilr1"], label="ILR1 (EMA)", linestyle="--", color="blue")
    # plt.plot(mean_relative.index, mean_ilr_smooth_ema["ilr2"], label="ILR2 (EMA)", linestyle="--", color="orange")
    # plt.plot(mean_relative.index, mean_ilr_smooth_ema["ilr3"], label="ILR3 (EMA)", linestyle="--", color="green")

    plt.plot(mean_relative.index, mean_ilr_smooth_ema["ina"], label="ina (EMA)", linestyle="--", color="blue")
    plt.plot(mean_relative.index, mean_ilr_smooth_ema["na"], label="na (EMA)", linestyle="--", color="orange")
    plt.plot(mean_relative.index, mean_ilr_smooth_ema["nna"], label="nna (EMA)", linestyle="--", color="green")
    plt.plot(mean_relative.index, mean_ilr_smooth_ema["enna"], label="enna (EMA)", linestyle="--", color="red")
    
    # Formatting the plot
    plt.xlabel("Relative Time")
    plt.ylabel("Action value")
    plt.title(f"Cluster {cluster} - Original and Smoothed Time Series")
    plt.legend()
    plt.show()