import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans

##############################
# Step 1: Load & Prepare Data
##############################

# Replace 'your_data.csv' with your actual CSV file path
df = pd.read_csv('../data/data_ilr_transformed/combined_data__resid_with_time.csv')

# Ensure data is sorted by ID and time
df = df.sort_values(by=['ID', 'time'])

# Group by ID and convert each group into a sequence array [T, 3]
grouped = df.groupby('ID')
sequences = []
lengths = []

for pid, group in grouped:
    # Extract ilr1, ilr2, ilr3 into a numpy array
    seq = group[['ilr1_residual', 'ilr2_residual', 'ilr3_residual']].values
    sequences.append(seq)
    lengths.append(len(seq))

# Find max length
max_length = max(lengths)

# Pad sequences to max_length with zeros
padded_sequences = []
for seq in sequences:
    T = len(seq)
    if T < max_length:
        pad_width = max_length - T
        # Pad at the end
        padded_seq = np.concatenate([seq, np.zeros((pad_width, 3))], axis=0)
    else:
        padded_seq = seq
    padded_sequences.append(padded_seq)

padded_sequences = np.stack(padded_sequences, axis=0)  # [N, max_length, 3]
lengths = np.array(lengths)

##############################
# Step 2: Create a Dataset
##############################

class RadicalizationDataset(Dataset):
    def __init__(self, data, lengths):
        # data: np array [N, T, 3]
        # lengths: np array [N]
        self.data = torch.from_numpy(data).float()
        self.lengths = lengths

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.lengths[idx]

dataset = RadicalizationDataset(padded_sequences, lengths)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

##############################
# Step 3: Define LSTM Autoencoder
##############################

class Encoder(nn.Module):
    def __init__(self, input_size=3, hidden_size=32, latent_size=16, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, latent_size)

    def forward(self, x, lengths):
        # x: [batch, T, input_size]
        # lengths: [batch]
        # For variable lengths, use pack_padded_sequence:
        # Sort by length (descending) to use pack_padded_sequence if needed
        sorted_lengths, indices = torch.sort(lengths, descending=True)
        x_sorted = x[indices]

        packed = nn.utils.rnn.pack_padded_sequence(x_sorted, sorted_lengths.cpu(), batch_first=True, enforce_sorted=True)
        _, (h, c) = self.lstm(packed)
        # h: [num_layers, batch, hidden_size]
        h = h[-1]  # take last layer
        # Unsort h back to original order
        _, desort_indices = torch.sort(indices)
        h = h[desort_indices]

        latent = self.fc(h) # [batch, latent_size]
        return latent

class Decoder(nn.Module):
    def __init__(self, latent_size=16, hidden_size=32, output_size=3, num_layers=1, max_length=15):
        super().__init__()
        self.latent_to_hidden = nn.Linear(latent_size, hidden_size)
        self.lstm = nn.LSTM(output_size, hidden_size, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.max_length = max_length

    def forward(self, latent):
        batch_size = latent.size(0)
        hidden = self.latent_to_hidden(latent).unsqueeze(0)  # [1, batch, hidden_size]
        cell = torch.zeros_like(hidden) # [1, batch, hidden_size]

        # Use zero as input at each timestep
        inputs = torch.zeros(batch_size, self.max_length, 3, device=latent.device)
        output, (h, c) = self.lstm(inputs, (hidden, cell))
        reconstruction = self.output_layer(output) # [batch, T, 3]
        return reconstruction

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size=3, hidden_size=32, latent_size=16, num_layers=1, max_length=15):
        super().__init__()
        self.encoder = Encoder(input_size, hidden_size, latent_size, num_layers)
        self.decoder = Decoder(latent_size, hidden_size, input_size, num_layers, max_length)

    def forward(self, x, lengths):
        latent = self.encoder(x, lengths)
        reconstruction = self.decoder(latent)
        return reconstruction, latent

##############################
# Step 4: Train the Model
##############################

def train_autoencoder(model, dataloader, epochs=20, lr=1e-3, device='cpu'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, batch_len in dataloader:
            batch_x, batch_len = batch_x.to(device), batch_len.to(device)
            reconstruction, _ = model(batch_x, batch_len)
            loss = criterion(reconstruction, batch_x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
        
        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    return model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = LSTMAutoencoder(input_size=3, hidden_size=32, latent_size=16, num_layers=1, max_length=max_length)
trained_model = train_autoencoder(model, dataloader, epochs=20, lr=1e-3, device=device)

##############################
# Step 5: Extract Embeddings & Cluster
##############################

embeddings = []
trained_model.eval()
with torch.no_grad():
    for batch_x, batch_len in DataLoader(dataset, batch_size=32):
        batch_x, batch_len = batch_x.to(device), batch_len.to(device)
        _, latent = trained_model(batch_x, batch_len)
        embeddings.append(latent.cpu().numpy())

embeddings = np.concatenate(embeddings, axis=0) # [N, latent_size]

# Perform K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
cluster_labels = kmeans.fit_predict(embeddings)

print("Cluster assignments:", cluster_labels)
