import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler
from tslearn.clustering import TimeSeriesKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

print("=========================")
print("Running DTW clustering...")
print("=========================")

##### Setup #####
df = pd.read_csv("../data/data_ilr_transformed/combined_data__resid_with_time.csv")
print(f"Dataset succesfully imported. Shape: {df.shape}")
print(" - Number of experiments:", len(df["Experiment"].unique()))
print(" - Number of unique time points:", len(df["time"].unique()))

# Variables
original_DVs = ["ina", "na", "nna", "enna"]
ilr_columns = ['ilr1', 'ilr2', 'ilr3']
ilr_resid_columns = ['ilr1_residual', 'ilr2_residual', 'ilr3_residual']
control_columns = ['excluded', 'injustice', 'personal', 'violence']



##### Feature Engineering #####
# Additional time-based feature: time point relative to the total duration for each participant
df['relative_time'] = df.groupby('ID')['time'].transform(lambda x: x / x.max())

# Calculate mean and standard deviation per participant (ID) for each ilr column
for col in ilr_resid_columns:
    df[f'{col}_mean'] = df.groupby('ID')[col].transform('mean')
    df[f'{col}_std'] = df.groupby('ID')[col].transform('std')
    df[f'{col}_min'] = df.groupby('ID')[col].transform('min')
    df[f'{col}_max'] = df.groupby('ID')[col].transform('max')

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

# Fourier transformation
fourier_components = 2
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
            .rename(columns=lambda i: f'{col}_fourier_2_{i+1}')
        )

        # Add Fourier features back to the dataframe
        df = df.join(fourier_df, on='ID')

# Selecting the feature columns
feature_columns = [
    'relative_time',
    'ilr1_residual_mean', 'ilr2_residual_mean', 'ilr3_residual_mean',
    'ilr1_residual_std', 'ilr2_residual_std', 'ilr3_residual_std',
    'ilr1_residual_min', 'ilr2_residual_min', 'ilr3_residual_min',
    'ilr1_residual_max', 'ilr2_residual_max', 'ilr3_residual_max',
    # 'ilr1_moving_avg_residual_fourier_1', 'ilr2_moving_avg_residual_fourier_1', 'ilr3_moving_avg_residual_fourier_1',
    'ilr1_residual_ema_fourier_1', 'ilr2_residual_ema_fourier_1', # 'ilr3_residual_ema_fourier_1',
    # 'ilr3_residual_ema_fourier_2_1', 'ilr3_residual_ema_fourier_2_3', 'ilr3_residual_ema_fourier_2_4'
]
from sklearn.preprocessing import RobustScaler
feature_columns_to_scale = [col for col in feature_columns if col != 'relative_time']
print("Scaling the columns:", feature_columns_to_scale)
df[feature_columns_to_scale] = RobustScaler().fit_transform(df[feature_columns_to_scale])

print("====================")
print(f"{len(feature_columns)} features used:")
print(feature_columns)



##### Prepare Time Series Data for Clustering #####
time_series_data = []
grouped = df.groupby("ID")
for participant, group in grouped:
    group_sorted = group.sort_values("time")
    time_series = group_sorted[feature_columns].values

    # Padding or truncating each participant's time series to the same length
    max_length = 15 # The 'time' in the Experiments ranges from 6 to 15
    if len(time_series) < max_length:
        time_series = np.pad(time_series, ((0, max_length - len(time_series)), (0, 0)), 'constant', constant_values=np.nan)
    else:
        time_series = time_series[:max_length]
    time_series_data.append(time_series)
time_series_data = np.array(time_series_data)



##### Clustering #####
n_clusters = 3
print("====================")
print("Number of clusters:", n_clusters)
model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", random_state=0)
labels = model.fit_predict(time_series_data)
# Adding labels to the original DataFrame for analysis:
df["cluster"] = df["ID"].map(dict(zip(grouped.groups.keys(), labels)))

# Create a summary of the number of participants from each experiment within each cluster
experiment_cluster_summary = df.groupby(['cluster', 'Experiment']).size().unstack(fill_value=0)
print("====================")
print("Clustering summaries:")
print("====================")
print(experiment_cluster_summary)
print("====================")

# Compute the cluster-wise percentages for each experiment
experiment_cluster_summary_percentages = experiment_cluster_summary.div(experiment_cluster_summary.sum(axis=0), axis=1) * 100
experiment_cluster_summary_percentages = experiment_cluster_summary_percentages.round(1)
print(experiment_cluster_summary_percentages)
print("====================")

cluster_counts = df["cluster"].value_counts()
print(cluster_counts)
print("====================")

cluster_summary = df.groupby("cluster")[["ina", "na", "nna", "enna", "ilr1", "ilr2", "ilr3"]].mean()
print(cluster_summary)
print("====================")


##### Evaluation (per study!) #####
from tslearn.metrics import dtw, dtw_path
import numpy as np

def dtw_silhouette_score(X, labels):
    n_samples = len(X)
    if n_samples <= 1:  # Handle cases with only one or zero samples
        return np.nan
    silhouettes = []
    for i in range(n_samples):
        label_i = labels[i]
        a_i = 0
        count_a = 0
        for j in range(n_samples):
            if i != j and labels[j] == label_i:  # Exclude self-comparison
                a_i += dtw(X[i], X[j])
                count_a += 1
        if count_a == 0:  # Avoid division by zero if there's only one element in the cluster
            a_i = 0
        else:
            a_i /= count_a

        b_i = np.inf
        for label_j in set(labels):
            if label_j != label_i:
                b_j = 0
                count_b = 0
                for k in range(n_samples):
                    if labels[k] == label_j:
                        b_j += dtw(X[i], X[k])
                        count_b += 1
                if count_b > 0:
                    b_j /= count_b
                    b_i = min(b_i, b_j)
        if b_i == np.inf:
            s_i = 0.0
        else:
            s_i = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) != 0 else 0.0 # Avoid division by zero
        silhouettes.append(s_i)
    return np.nanmean(silhouettes)

silhouette_scores = []
davies_bouldin_scores = []
dtw_silhouette_scores = []

# Assume you have a variable `participant_ids` which stores the participant IDs in the exact order they appear in `time_series_data` and `labels`.
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
        p_data = df[df['ID'] == pid].sort_values('time')[feature_columns].values
        
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
    
    dtw_sil = dtw_silhouette_score(truncated_series, exp_labels)
    print(f"Experiment {exp}: DTW Silhouette Score = {dtw_sil}")
    dtw_silhouette_scores.append(dtw_sil)

    print(f"Experiment {exp}: Silhouette Score = {sil}, Davies-Bouldin Score = {db}")
    silhouette_scores.append(sil)
    davies_bouldin_scores.append(db)

# Calculating average and range
silhouette_avg = sum(silhouette_scores) / len(silhouette_scores)
davies_bouldin_avg = sum(davies_bouldin_scores) / len(davies_bouldin_scores)

print("====================")
print("silhouette_scores")
print("average:", silhouette_avg)
print("max:", max(silhouette_scores))
print("min:", min(silhouette_scores))

print("====================")
print("davies_bouldin_scores")
print("average:", davies_bouldin_avg)
print("max:", max(davies_bouldin_scores))
print("min:", min(davies_bouldin_scores))
print("====================")

# Calculating average and range for DTW Silhouette
dtw_silhouette_avg = np.nanmean(dtw_silhouette_scores)
print("====================")
print("DTW silhouette_scores")
print("average:", dtw_silhouette_avg)
print("max:", np.nanmax(dtw_silhouette_scores))
print("min:", np.nanmin(dtw_silhouette_scores))
print("====================")


##### Plotting #####
import matplotlib.pyplot as plt
import seaborn as sns

# Define font properties
font_size = 19

# Set Seaborn style
sns.set(style="whitegrid")

# Parameters
n_clusters = 3
alpha = 0.05  # Smoothing factor for EMA

# Set up a grid for side-by-side plots
fig, axes = plt.subplots(1, n_clusters, figsize=(15, 5), sharey=True)

# Define colors for the action options
colors = {
    "ina": "#4E79A7",  # Muted Blue
    "na": "#F28E2B",  # Soft Orange
    "nna": "#76B7B2",  # Teal
    "enna": "#E15759",  # Soft Red
}

# Calculate the number of unique IDs per cluster
id_counts = df.groupby('cluster')['ID'].nunique()
total_ids = df['ID'].nunique()

# Loop through clusters
for cluster, ax in enumerate(axes):
    cluster_data = df[df['cluster'] == cluster]
    mean_relative = cluster_data.groupby('relative_time')[['ina', 'na', 'nna', 'enna']].mean()
    
    # Apply Exponential Moving Average (EMA) smoothing
    mean_smooth_ema = mean_relative.apply(lambda x: x.ewm(alpha=alpha).mean())
    
    # Plot EMA smoothed series for each action option
    for action, color in colors.items():
        ax.plot(
            mean_relative.index, mean_smooth_ema[action], 
            label=action, color=color, linestyle="-"
        )
    
    # Formatting for each subplot
    
    percentage = (id_counts[cluster] / total_ids) * 100

    # Formatting for each subplot
    ax.set_title(
        r"$\bf{Cluster\ " + str(cluster + 1) + r"}$" + "\n" + 
        r"($\it{N}$" + f" = {id_counts[cluster]}, {percentage:.1f}%)",
        fontsize=font_size
    )



    if cluster == 0:  # Add Y-axis label only to the first plot
        ax.set_ylabel("Action Value", fontsize=font_size)
    ax.tick_params(axis='both', labelsize=font_size - 2)
    ax.grid(False)

# Add a common x-axis label
fig.text(0.5, 0.01, "Relative Time", ha="center", fontsize=font_size)

# Add a legend
action_labels = {
    "ina": "Inaction",
    "na": "Normative Action",
    "nna": "Non-Normative Action",
    "enna": "Extreme Non-Normative Action",
}

fig.legend(
    handles=[plt.Line2D([0], [0], color=color, linestyle="-", lw=2) for color in colors.values()],
    labels=list(action_labels.values()),  # Use descriptive labels
    loc="upper center", ncol=4, frameon=False, fontsize=font_size
)

# Adjust layout
fig.tight_layout(rect=[0, 0.05, 1, 0.9])  # Leave space for the legend and the common x-label
plt.show()
