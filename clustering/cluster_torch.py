import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

class TimeSeriesAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super(TimeSeriesAutoencoder, self).__init__()
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(
            *encoder_layers,
            nn.Linear(prev_dim, latent_dim)
        )
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        self.decoder = nn.Sequential(
            *decoder_layers,
            nn.Linear(prev_dim, input_dim)
        )
    
    def forward(self, x):
        # Ensure x is a tensor and flatten
        if not isinstance(x, torch.Tensor):
            x = x[0] if isinstance(x, list) else torch.tensor(x)
        
        # Flatten the input if it's 3D (batch, time steps, features)
        original_shape = x.shape
        if len(x.shape) == 3:
            x = x.view(original_shape[0], -1)
        
        # Ensure float tensor
        x = x.float()
        
        # Encode
        latent = self.encoder(x)
        
        # Decode
        reconstructed = self.decoder(latent)
        
        return reconstructed, latent

def prepare_data(df, feature_columns, max_length=15):
    # Group data by ID
    grouped = df.groupby("ID")
    time_series_data = []
    
    for participant, group in grouped:
        group_sorted = group.sort_values("time")
        time_series = group_sorted[feature_columns].values
        
        # Ensure consistent sequence length
        if len(time_series) < max_length:
            time_series = np.pad(time_series, 
                                 ((0, max_length - len(time_series)), (0, 0)), 
                                 mode='constant', 
                                 constant_values=0)
        else:
            time_series = time_series[:max_length]
        
        time_series_data.append(time_series)
    
    return np.array(time_series_data)

def train_autoencoder(model, dataloader, criterion, optimizer, epochs=100):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            reconstructed, _ = model(batch)
            
            # Compute loss
            loss = criterion(reconstructed, batch[0].view(batch[0].size(0), -1))
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}')
    
    return model

def evaluate_clustering(latent_representations, cluster_labels):
    """
    Compute clustering evaluation metrics
    """
    metrics = {
        'Silhouette Score': silhouette_score(latent_representations, cluster_labels),
        'Calinski-Harabasz Index': calinski_harabasz_score(latent_representations, cluster_labels),
        'Davies-Bouldin Index': davies_bouldin_score(latent_representations, cluster_labels)
    }
    
    print("\nClustering Evaluation Metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    
    return metrics

def visualize_clusters(latent_representations, cluster_labels, method='tsne'):
    """
    Visualize clusters using dimensionality reduction
    """
    plt.figure(figsize=(10, 6))
    
    # T-SNE Visualization
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
        reduced_data = reducer.fit_transform(latent_representations)
        title = 't-SNE Visualization of Latent Representations'
    
    # PCA Visualization
    elif method == 'pca':
        reducer = PCA(n_components=2)
        reduced_data = reducer.fit_transform(latent_representations)
        title = 'PCA Visualization of Latent Representations'
    
    # Create scatter plot
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], 
                c=cluster_labels, cmap='viridis', alpha=0.7)
    plt.colorbar(label='Cluster')
    plt.title(title)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.tight_layout()
    plt.show()

def cluster_distribution_analysis(df, cluster_column):
    """
    Analyze distribution of key variables across clusters
    """
    # Numerical columns for analysis
    numerical_cols = [
        'ilr1_residual', 'ilr2_residual', 'ilr3_residual', 
        'age', 'time'
    ]
    
    # Categorical columns for analysis
    categorical_cols = ['gender', 'condition']
    
    # Numerical Analysis
    print("\nNumerical Variables Distribution Across Clusters:")
    numerical_summary = df.groupby(cluster_column)[numerical_cols].mean()
    print(numerical_summary)
    
    # Categorical Analysis
    print("\nCategorical Variables Distribution Across Clusters:")
    for col in categorical_cols:
        print(f"\n{col} Distribution:")
        print(df.groupby([cluster_column, col]).size().unstack(fill_value=0))
    
    # Visualization of numerical distributions
    plt.figure(figsize=(15, 5))
    for i, col in enumerate(numerical_cols, 1):
        plt.subplot(1, len(numerical_cols), i)
        sns.boxplot(x=cluster_column, y=col, data=df)
        plt.title(f'{col} by Cluster')
    plt.tight_layout()
    plt.show()

def main():
    # Setup
    df = pd.read_csv("data/data_ilr_transformed/combined_data.csv")
    print(f"Combined dataset shape: {df.shape}")

    # Variables
    control_columns = ['excluded', 'injustice', 'personal', 'violence']
    covariate_columns = ['gender', 'age', 'condition']
    ilr_resid_columns = ['ilr1_residual', 'ilr2_residual', 'ilr3_residual']

    # Handle NA values and standardize age for covariate
    for covariate in covariate_columns:
        df = df[df[covariate].notna()]
    df[['age']] = StandardScaler().fit_transform(df[['age']])

    # Feature Engineering
    df['relative_time'] = df.groupby('ID')['time'].transform(lambda x: x / x.max())

    # Calculate statistics per participant
    for col in ilr_resid_columns:
        df[f'{col}_mean'] = df.groupby('ID')[col].transform('mean')
        df[f'{col}_std'] = df.groupby('ID')[col].transform('std')
        df[f'{col}_min'] = df.groupby('ID')[col].transform('min')
        df[f'{col}_max'] = df.groupby('ID')[col].transform('max')

    # Moving Average
    window_size = 2
    for col in ilr_resid_columns:
        df[f'{col}_moving_avg'] = df.groupby('ID')[col].transform(
            lambda x: x.rolling(window=window_size, min_periods=1).mean()
        )

    # Define features
    feature_columns = [
        'relative_time',
        'ilr1_residual_mean', 'ilr2_residual_mean', 'ilr3_residual_mean',
        'ilr1_residual_std', 'ilr2_residual_std', 'ilr3_residual_std',
        'ilr1_residual_min', 'ilr2_residual_min', 'ilr3_residual_min',
        'ilr1_residual_max', 'ilr2_residual_max', 'ilr3_residual_max',
        'ilr1_residual_moving_avg', 'ilr2_residual_moving_avg', 'ilr3_residual_moving_avg'
    ]

    # Prepare time series data
    time_series_data = prepare_data(df, feature_columns)

    # Convert to PyTorch tensors
    X = torch.FloatTensor(time_series_data)

    # Create DataLoader
    dataset = TensorDataset(X)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Hyperparameters
    input_dim = X.shape[1] * X.shape[2]  # Flattened features
    hidden_dims = [64, 32]  # Hidden layer dimensions
    latent_dim = 10  # Dimensionality of latent space

    # Initialize model
    model = TimeSeriesAutoencoder(input_dim, hidden_dims, latent_dim)

    # Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train Autoencoder
    model = train_autoencoder(model, dataloader, criterion, optimizer)

    # Extract Latent Representations
    model.eval()
    with torch.no_grad():
        latent_representations = []
        for batch in dataloader:
            _, latent = model(batch)
            latent_representations.append(latent.numpy())
    
    latent_representations = np.concatenate(latent_representations)
    print(f"latent_representations shape: {latent_representations.shape}")

    # Clustering on Latent Space
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(latent_representations)

    # Map clusters back to original DataFrame
    cluster_mapping = dict(zip(df['ID'].unique(), cluster_labels))
    df['cluster'] = df['ID'].map(cluster_mapping)

    # Evaluation
    evaluate_clustering(latent_representations, cluster_labels)

    # Visualizations
    visualize_clusters(latent_representations, cluster_labels, method='tsne')
    visualize_clusters(latent_representations, cluster_labels, method='pca')

    # Cluster Distribution Analysis
    cluster_distribution_analysis(df, 'cluster')

    return df, model, latent_representations, cluster_labels

if __name__ == "__main__":
    df_clustered, autoencoder_model, latent_reps, cluster_labels = main()
