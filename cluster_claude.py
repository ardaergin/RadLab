import numpy as np
import pandas as pd
from tslearn.clustering import TimeSeriesKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from statsmodels.regression.mixed_linear_model import MixedLM
from tslearn.clustering import TimeSeriesKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt



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

## Interaction Terms: Control Variables with Time
for col in control_columns:
    df[f'{col}_time_interaction'] = df[col] * df['time']

# Moving Average or Exponential Moving Average (EMA)
window_size = 2 
for col in ilr_columns:
    df[f'{col}_moving_avg'] = df.groupby('ID')[col].transform(lambda x: x.rolling(window=window_size, min_periods=1).mean())
    df[f'{col}_ema'] = df.groupby('ID')[col].transform(lambda x: x.ewm(span=window_size, adjust=False).mean())
ilr_moving_avg_cols = ['ilr1_moving_avg', 'ilr2_moving_avg', 'ilr3_moving_avg']

# Residualize each ilr variable
for col in ilr_columns:
    model = MixedLM.from_formula(f"{col} ~ relative_time", groups="Experiment", data=df)
    result = model.fit()
    df[f'{col}_residual'] = df[col] - result.fittedvalues
ilr_resid_columns = ['ilr1_residual', 'ilr2_residual', 'ilr3_residual']

# Residualize each ilr variable moving average
for col in ilr_moving_avg_cols:
    model = MixedLM.from_formula(f"{col} ~ relative_time", groups="Experiment", data=df)
    result = model.fit()
    df[f'{col}_residual'] = df[col] - result.fittedvalues
ilr_moving_avg_residual_cols = ['ilr1_moving_avg_residual', 'ilr2_moving_avg_residual', 'ilr3_moving_avg_residual']

# Number of Fourier components to retain
fourier_components = 1
# Compute Fourier transform for each 'ilr' variable per participant
for column_group in [ilr_resid_columns, ilr_moving_avg_residual_cols]:
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
    'condition',
    'gender', 'age',
    'excluded', 'injustice', 'personal', 'violence',
    'ilr1_mean', 'ilr2_mean', 'ilr3_mean',
    'ilr1_std', 'ilr2_std', 'ilr3_std',
    'ilr1_min', 'ilr2_min', 'ilr3_min',
    'ilr1_max', 'ilr2_max', 'ilr3_max',
    'relative_time',
    'excluded_time_interaction', 'injustice_time_interaction', 'personal_time_interaction', 'violence_time_interaction',
    # 'ilr1_moving_avg', 'ilr2_moving_avg', 'ilr3_moving_avg', 
    # 'ilr1_ema', 'ilr2_ema', 'ilr3_ema',
    #'ilr1_residual', 'ilr2_residual', 'ilr3_residual',
    #'ilr1_fourier_1', 'ilr2_fourier_1', 'ilr3_fourier_1',
    #'ilr1_fourier_2', 'ilr2_fourier_2', 'ilr3_fourier_2'
    # 'ilr1_moving_avg_fourier_1', 'ilr2_moving_avg_fourier_1', 'ilr3_moving_avg_fourier_1',
    # 'ilr1_moving_avg_fourier_2', 'ilr2_moving_avg_fourier_2', 'ilr3_moving_avg_fourier_2',
    'ilr1_moving_avg_residual_fourier_1', 'ilr2_moving_avg_residual_fourier_1', 'ilr3_moving_avg_residual_fourier_1',
    #'ilr1_moving_avg_residual_fourier_2', 'ilr2_moving_avg_residual_fourier_2', 'ilr3_moving_avg_residual_fourier_2'
]


# Preprocess the time series data for each participant
for participant, group in grouped:
    group_sorted = group.sort_values("time")
    time_series = group_sorted[feature_columns].values
    
    # Pad or truncate each participant's time series to the same length
    max_length = 15 # The 'time' in the Experiments ranges from 6 to 15
    if len(time_series) < max_length:
        time_series = np.pad(time_series, ((0, max_length - len(time_series)), (0, 0)), 'constant', constant_values=np.nan)
    else:
        time_series = time_series[:max_length]
    
    time_series_data.append(time_series)

# Convert to numpy array for clustering
time_series_data = np.array(time_series_data)



##### Clustering #####
n_clusters = 3
model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", random_state=0)
labels = model.fit_predict(time_series_data)
# Adding labels to the original DataFrame for analysis:
df["cluster"] = df["ID"].map(dict(zip(grouped.groups.keys(), labels)))



def evaluate_time_series_clustering(time_series_data, n_clusters_range=range(2, 6), metric='dtw'):
    """
    Evaluate time series clustering across different numbers of clusters
    
    Parameters:
    - time_series_data: numpy array of time series
    - n_clusters_range: range of cluster numbers to evaluate
    - metric: distance metric for clustering
    
    Returns:
    - DataFrame with clustering evaluation metrics
    """
    results = []
    
    # Preprocess: Scale features across time series
    scaler = StandardScaler()
    scaled_data = np.zeros_like(time_series_data, dtype=float)
    for i in range(time_series_data.shape[0]):
        # Handle NaN values by filling with column mean
        series = time_series_data[i]
        mask = ~np.isnan(series).all(axis=1)
        if np.any(mask):
            scaled_data[i][mask] = scaler.fit_transform(series[mask])
    
    for n_clusters in n_clusters_range:
        # Perform clustering
        model = TimeSeriesKMeans(
            n_clusters=n_clusters, 
            metric=metric, 
            random_state=42
        )
        
        # Fit and predict
        labels = model.fit_predict(scaled_data)
        
        # Compute metrics
        try:
            # Silhouette score (requires at least 2 clusters)
            if n_clusters > 1:
                # Flatten the data for silhouette score
                flat_data = scaled_data.reshape(scaled_data.shape[0], -1)
                sil_score = silhouette_score(flat_data, labels)
            else:
                sil_score = np.nan
            
            # Davies-Bouldin score
            db_score = davies_bouldin_score(flat_data, labels)
            
            # Cluster sizes
            unique, counts = np.unique(labels, return_counts=True)
            cluster_sizes = dict(zip(unique, counts))
            
            results.append({
                'n_clusters': n_clusters,
                'silhouette_score': sil_score,
                'davies_bouldin_score': db_score,
                'cluster_sizes': cluster_sizes
            })
        except Exception as e:
            print(f"Error evaluating {n_clusters} clusters: {e}")
    
    return pd.DataFrame(results)

# Example usage
evaluation_results = evaluate_time_series_clustering(time_series_data)
print(evaluation_results)

# Visualize cluster distribution
plt.figure(figsize=(10, 6))
for index, row in evaluation_results.iterrows():
    print(f"\nClusters: {row['n_clusters']}")
    print("Cluster Sizes:", row['cluster_sizes'])
    print(f"Silhouette Score: {row['silhouette_score']:.4f}")
    print(f"Davies-Bouldin Score: {row['davies_bouldin_score']:.4f}")

# Optional: Experimental group balance check
def check_cluster_experimental_balance(df, labels):
    """
    Check if clusters are balanced across experiments
    """
    cluster_experiment_dist = pd.crosstab(df['Experiment'], labels)
    print("\nCluster Distribution Across Experiments:")
    print(cluster_experiment_dist)
    print("\nChi-square test for independence:")
    from scipy.stats import chi2_contingency
    chi2, p_value, dof, expected = chi2_contingency(cluster_experiment_dist)
    print(f"p-value: {p_value}")

# Call the balance check function
check_cluster_experimental_balance(df, labels)
