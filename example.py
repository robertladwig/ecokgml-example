import numpy as np
import matplotlib.pyplot as plt

# Simulate a diurnal cycle with 3 regimes (e.g., day-night CH₄ variations)
np.random.seed(42)


import pandas as pd

# Load the data from the CSV file
data = pd.read_csv('/Users/au740615/Documents/projects/ecokgml-example/processed_data.csv')

# Convert the dates and time stamps to one datetime column
data['datetime'] = pd.to_datetime(data['sampledate_datetime'] + ' ' + data['sampletime_datetime'])

# Drop the original date and time columns
data.drop(columns=['sampledate_datetime', 'sampletime_datetime'], inplace=True)

# Sort the data by datetime
data.sort_values(by='datetime', inplace=True)

a = data.chlor_rfu.values  # Extract out relevant column from dataframe as array
m = np.concatenate(( [True], np.isnan(a), [True] ))  # Mask
ss = np.flatnonzero(m[1:] != m[:-1]).reshape(-1,2)   # Start-stop limits
start,stop = ss[(ss[:,1] - ss[:,0]).argmax()]  # Get max interval, interval limits

longest_streak_data = data.loc[start:stop]


# Save the filtered data to a new CSV file

longest_streak_data.to_csv('/Users/au740615/Documents/projects/ecokgml-example/filtered_data.csv', index=False)




plt.figure(figsize=(12, 4))
plt.plot(longest_streak_data.datetime, longest_streak_data.chlor_rfu)
plt.title("Chlorophyll-a time series")
plt.xlabel("Time")
plt.ylabel("Chl-a signal")
plt.tight_layout()
plt.show()

longest_streak_data['datetime'] = pd.to_datetime(longest_streak_data['datetime'])

diurnal = longest_streak_data.resample('1H', on = 'datetime')['chlor_rfu'].mean().interpolate()


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
diurnal_scaled = scaler.fit_transform(diurnal.values.reshape(-1, 1)).flatten()

# Create sliding windows (48-hour windows with 1-hour step)
import numpy as np

def create_windows(data, window_size=48, step=1):
    return np.array([data[i:i+window_size] for i in range(0, len(data) - window_size, step)])

window_size = 48
X = create_windows(diurnal_scaled, window_size=window_size)

X.shape  # Confirm shape: (num_windows, 48)


import torch
import torch.nn as nn
from sklearn.cluster import KMeans

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim=1, model_dim=32, num_heads=4, num_layers=2):
        super().__init__()
        self.model_dim = model_dim
        self.embedding = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pooling = nn.AdaptiveAvgPool1d(1)  # Global average pooling

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.permute(0, 2, 1)  # [B, D, T]
        pooled = self.pooling(x).squeeze(-1)  # [B, D]
        return pooled

# Prepare input: [batch, time, 1]
X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)

# Model
model = TransformerEncoder()
with torch.no_grad():
    embeddings = model(X_tensor).numpy()  # Latent representation of each window


kmeans = KMeans(n_clusters=3, random_state=0)
labels = kmeans.fit_predict(embeddings)

# Map labels back to time
label_series = np.zeros_like(diurnal_scaled)
for i, label in enumerate(labels):
    label_series[i:i+48] = label  # propagate label over window

plt.figure(figsize=(12, 4))
plt.plot(diurnal_scaled, label="Signal", alpha=0.4)
plt.scatter(range(len(label_series)), diurnal_scaled, c=label_series, cmap="Set1", s=5, label="Cluster")
plt.title("Transformer Latent Clustering of Diurnal Time Series")
plt.xlabel("Time Step")
plt.ylabel("Normalized Value")
plt.legend()
plt.tight_layout()
plt.show()

from sklearn.manifold import TSNE
import seaborn as sns

# Reduce to 2D with t-SNE
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=0)
embeddings_2d = tsne.fit_transform(embeddings)

# Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], hue=labels, palette='Set1', s=40)
plt.title("t-SNE of Transformer Latent Space")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend(title="Cluster")
plt.tight_layout()
plt.show()
