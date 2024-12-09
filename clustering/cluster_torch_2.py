import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tslearn.clustering import TimeSeriesKMeans
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

##### Setup #####
df = pd.read_csv("data/data_ilr_transformed/combined_data.csv")
print(f"Combined dataset shape: {df.shape}")

# Variables
ilr_columns = ['ilr1', 'ilr2', 'ilr3']
control_columns = ['excluded', 'injustice', 'personal', 'violence']
covariate_columns = ['gender', 'age', 'condition']
ilr_resid_columns = ['ilr1_residual', 'ilr2_residual', 'ilr3_residual']

# Handle NA values and standardize age
for covariate in covariate_columns:
    df = df[df[covariate].notna()]
df[['age']] = StandardScaler().fit_transform(df[['age']])

##### Feature Engineering #####
# Relative time
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
fourier_components = 2
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
        df = df.join(fourier_df, on='ID')



##### Prepare Time Series Data #####
# Define features
feature_columns = [
    'relative_time',
    'ilr1_residual_mean', 'ilr2_residual_mean', 'ilr3_residual_mean',
    'ilr1_residual_std', 'ilr2_residual_std', 'ilr3_residual_std',
    'ilr1_residual_min', 'ilr2_residual_min', 'ilr3_residual_min',
    'ilr1_residual_max', 'ilr2_residual_max', 'ilr3_residual_max',
    'ilr1_residual_moving_avg_fourier_1', 'ilr2_residual_moving_avg_fourier_1', 'ilr3_residual_moving_avg_fourier_1',
    'ilr1_residual_moving_avg_fourier_2', 'ilr2_residual_moving_avg_fourier_2', 'ilr3_residual_moving_avg_fourier_2',
]
print(f"{len(feature_columns)} features used:")
print(feature_columns)


# Group data by # Prepare tensor dataset
grouped = df.groupby("ID")
time_series_data = []
for participant, group in grouped:
    group_sorted = group.sort_values("time")
    time_series = group_sorted[feature_columns].values

    # Ensure consistent sequence length by padding or truncating
    max_length = 15  # Example maximum time points
    if len(time_series) < max_length:
        time_series = np.pad(time_series, ((0, max_length - len(time_series)), (0, 0)), mode='constant', constant_values=0)
    else:
        time_series = time_series[:max_length]

    time_series_data.append(time_series)

time_series_data = np.array(time_series_data)
print(f"Time-series tensor shape: {time_series_data.shape}")  # (num_samples, sequence_length, num_features)

# Convert to PyTorch tensors
time_series_tensor = torch.tensor(time_series_data, dtype=torch.float32)

# Create DataLoader
dataset = TensorDataset(time_series_tensor)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)


class TimeSeriesAutoencoder(nn.Module):
    def __init__(self, input_size, latent_size):
        super(TimeSeriesAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, latent_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 64),
            nn.ReLU(),
            nn.Linear(64, input_size)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


# Hyperparameters
input_size = len(feature_columns)  # Features per time step
latent_size = 16  # Size of latent space
sequence_length = 15  # Number of time steps

# Initialize model, loss, and optimizer
model = TimeSeriesAutoencoder(input_size, latent_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 50
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in data_loader:
        inputs = batch[0].view(-1, input_size)  # Flatten time-series data for autoencoder
        optimizer.zero_grad()
        _, outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(data_loader):.4f}")


model.eval()
latent_representations = []
with torch.no_grad():
    for batch in data_loader:
        inputs = batch[0].view(-1, input_size)
        encoded, _ = model(inputs)
        latent_representations.append(encoded.numpy())

latent_representations = np.concatenate(latent_representations, axis=0)
print(f"Latent representations shape: {latent_representations.shape}")


from sklearn.cluster import KMeans

n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
labels = kmeans.fit_predict(latent_representations)

# Map cluster labels back to participants
df['cluster'] = df['ID'].map(dict(zip(grouped.groups.keys(), labels)))

# Analyze clusters
experiment_cluster_summary = df.groupby(['cluster', 'Experiment']).size().unstack(fill_value=0)
print(experiment_cluster_summary)


from sklearn.metrics import silhouette_score, davies_bouldin_score

sil_score = silhouette_score(latent_representations, labels)
db_score = davies_bouldin_score(latent_representations, labels)

print(f"Silhouette Score: {sil_score:.4f}, Davies-Bouldin Score: {db_score:.4f}")



##### Plotting #####
import matplotlib.pyplot as plt

n_clusters = 3
alpha = 0.3  # Smoothing factor for EMA
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
