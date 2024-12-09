import torch
import torch.nn as nn

class TimeSeriesAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(TimeSeriesAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.latent = nn.Linear(hidden_dim, latent_dim)
        # Decoder
        self.decoder = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # Encoder
        _, (hidden, _) = self.encoder(x)
        z = self.latent(hidden[-1])
        # Decoder
        z_expanded = z.unsqueeze(1).repeat(1, x.size(1), 1)  # Repeat latent across time steps
        decoded, _ = self.decoder(z_expanded)
        reconstructed = self.output_layer(decoded)
        return reconstructed, z

model = TimeSeriesAutoencoder(input_dim=len(feature_columns), latent_dim=10, hidden_dim=64)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    for batch in train_loader:  # Assuming `time_series_data` is wrapped in a DataLoader
        optimizer.zero_grad()
        reconstructed, z = model(batch)
        loss = criterion(reconstructed, batch)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

model.eval()
latent_representations = []

with torch.no_grad():
    for batch in data_loader:  # Use all time-series data
        _, z = model(batch)
        latent_representations.append(z)

latent_representations = torch.cat(latent_representations, dim=0).numpy()
