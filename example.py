import numpy as np
import matplotlib.pyplot as plt
from math import pi, exp, sqrt, log, atan, sin, radians, nan, isinf, ceil, floor

# Simulate a diurnal cycle with 3 regimes (e.g., day-night CH₄ variations)
np.random.seed(42)


import pandas as pd

def do_sat_calc(temp, baro, altitude = 0, salinity = 0):
    mgL_mlL = 1.42905
    
    mmHg_mb = 0.750061683 # conversion from mm Hg to millibars
    if baro is None:
        mmHg_inHg = 25.3970886 # conversion from inches Hg to mm Hg
        standard_pressure_sea_level = 29.92126 # Pb, inches Hg
        standard_temperature_sea_level = 15 + 273.15 # Tb, 15 C = 288.15 K
        gravitational_acceleration = 9.80665 # g0, m/s^2
        air_molar_mass = 0.0289644 # M, molar mass of Earth's air (kg/mol)
        universal_gas_constant = 8.31447 #8.31432 # R*, N*m/(mol*K)
        
        # estimate pressure by the barometric formula
        baro = (1/mmHg_mb) * mmHg_inHg * standard_pressure_sea_level * exp((-gravitational_acceleration * air_molar_mass * altitude) / (universal_gas_constant * standard_temperature_sea_level))
    
    u = 10 ** (8.10765 - 1750.286 / (235 + temp)) # u is vapor pressure of water; water temp is used as an approximation for water & air temp at the air-water boundary
    press_corr = (baro*mmHg_mb - u) / (760 - u) # pressure correction is ratio of current to standard pressure after correcting for vapor pressure of water. 0.750061683 mmHg/mb
    
    ts = np.log((298.15 - temp)/(273.15 + temp))
    o2_sat = 2.00907 + 3.22014*ts + 4.05010*ts**2 + 4.94457*ts**3 + -2.56847e-1*ts**4 + 3.88767*ts**5 - salinity*(6.24523e-3 + 7.37614e-3*ts + 1.03410e-2*ts**2 + 8.17083e-3*ts**3) - 4.88682e-7*salinity**2
    return np.exp(o2_sat) * mgL_mlL * press_corr

## functions for gas exchange
def k_vachon(wind, area, param1 = 2.51, param2 = 1.48, param3 = 0.39):
    #wnd: wind speed at 10m height (m/s)
    #area: lake area (m2)
    #params: optional parameter changes
    k600 = param1 + (param2 * wind) + (param3 * wind * log(area/1000000, 10)) # units in cm per hour
    k600 = k600 * 24 / 100 #units in m per d
    return(k600)

def get_schmidt(temperature, gas = "O2"):
    t_range	= [4,35] # supported temperature range
    #if temperature < t_range[0] or temperature > t_range[1]:
    #    print("temperature:", temperature)
    #    raise Exception("temperature outside of expected range for kGas calculation")
    
    schmidt = pd.DataFrame({"He":[368,-16.75,0.374,-0.0036],
                   "O2":[1568,-86.04,2.142,-0.0216],
                  "CO2":[1742,-91.24,2.208,-0.0219],
                  "CH4":[1824,-98.12,2.413,-0.0241],
                  "SF6":[3255,-217.13,6.837,-0.0861],
                  "N2O":[2105,-130.08,3.486,-0.0365],
                  "Ar":[1799,-106.96,2.797,-0.0289],
                  "N2":[1615,-92.15,2.349,-0.0240]})
    gas_params = schmidt[gas]
    a = gas_params[0]
    b = gas_params[1]
    c = gas_params[2]
    d = gas_params[3]
    
    sc = a+b*temperature+c*temperature**2+d*temperature**3
    
    # print("a:", a, "b:", b, "c:", c, "d:", d, "sc:", sc)
    
    return(sc)

def k600_to_kgas(k600, temperature, gas = "O2"):
    n = 0.5
    schmidt = get_schmidt(temperature = temperature, gas = gas)
    sc600 = schmidt/600 
    
    kgas = k600 * (sc600**-n)
    
    #print("k600:", k600, "kGas:", kgas)
    return(kgas)

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



plt.figure(figsize=(6, 3))
plt.plot(longest_streak_data.datetime, longest_streak_data.wind_speed)
plt.xlabel("")
plt.ylabel("Wind velocity")
plt.tight_layout()
plt.grid(True)
plt.savefig('figs/bc_wind.png')
plt.show()


plt.figure(figsize=(6, 3))
plt.plot(longest_streak_data.datetime, longest_streak_data.do_wtemp)
plt.xlabel("")
plt.ylabel("Water temperature")
plt.tight_layout()
plt.grid(True)
plt.savefig('figs/bc_wtemp.png')
plt.show()


plt.figure(figsize=(6, 3))
plt.plot(longest_streak_data.datetime, longest_streak_data.par)
plt.xlabel("")
plt.ylabel("PAR")
plt.tight_layout()
plt.grid(True)
plt.savefig('figs/bc_par.png')
plt.show()


plt.figure(figsize=(10, 4))
plt.plot(longest_streak_data.datetime, longest_streak_data.chlor_rfu)
plt.title("Monitored time series")
plt.xlabel("")
plt.ylabel("Chl-a signal")
plt.grid(True)
plt.tight_layout()
plt.savefig('figs/bc_chla.png')
plt.show()


# Create process-based model

delta_t = 60
z_dep = 1 
r_prod = 1/86400
resp = - 1E-5
kd = 0.7

longest_streak_data.do_wtemp = longest_streak_data.do_wtemp.interpolate()

k600 =  k_vachon(wind = longest_streak_data.wind_speed, area = 39.4 * 10000)
piston_velocity = k600_to_kgas(k600 = k600, temperature = longest_streak_data.do_wtemp, gas = "O2")/86400
do_sat = do_sat_calc(longest_streak_data.do_wtemp, baro = 1024, altitude = 258) 
light_lower = longest_streak_data['par'] * np.exp(-kd * z_dep)
light_limit = np.log((z_dep + longest_streak_data['par']) / (z_dep + light_lower))
converted_chla = 5.057946 +  0.2166575 * longest_streak_data['chlor_rfu']
chla_limit = (converted_chla - np.min(converted_chla)) / (np.max(converted_chla) - np.min(converted_chla))
conv = 0.6 # https://doi.org/10.1038/s41598-019-43008-w

longest_streak_data['do_sim'] = longest_streak_data['do_raw']

for iter in range(1,(len(longest_streak_data['do_sim'] ))):
    atm_exchange = piston_velocity[iter] * (do_sat[iter] - longest_streak_data['do_sim'][iter - 1]) 
    produc = r_prod/ (kd * z_dep) * light_limit[iter] * converted_chla[iter] * conv
    longest_streak_data['do_sim'][iter] = (produc + atm_exchange + resp) * delta_t + longest_streak_data['do_sim'][iter - 1]

plt.figure(figsize=(10, 4))
plt.plot(longest_streak_data.datetime, longest_streak_data.do_raw)
plt.plot(longest_streak_data.datetime, longest_streak_data.do_sim)
plt.title("Observed vs. modeled signal")
plt.xlabel("")
plt.ylabel("DO signal")
plt.tight_layout()
plt.grid(True)
plt.savefig('figs/process_do.png')
plt.show()


longest_streak_data['datetime'] = pd.to_datetime(longest_streak_data['datetime'])

longest_streak_data.to_csv('/Users/au740615/Documents/projects/ecokgml-example/filtered_data.csv', index=False)


# =============================================
# 1. Import Libraries
# =============================================
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
from tqdm import tqdm

# =============================================
# 2. Load and Preprocess Data
# =============================================
df = pd.read_csv("/Users/au740615/Documents/projects/ecokgml-example/filtered_data.csv")
df['datetime'] = pd.to_datetime(df['datetime'])


# =============================================
# 3. Self-Attention for multivariate input + KMeans
# =============================================
class SimpleSelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x):
        Q, K, V = self.query(x), self.key(x), self.value(x)
        attention = torch.softmax(torch.bmm(Q, K.transpose(1, 2)) * self.scale, dim=-1)
        return torch.bmm(attention, V)

# Select and scale all relevant features
features = ['wind_speed', 'chlor_rfu', 'do_wtemp', 'par', 'do_raw']
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df[features])  # shape: (T, 5)

# Reshape for self-attention: (batch_size, seq_len, feature_dim)
input_tensor = torch.tensor(scaled_features, dtype=torch.float32).unsqueeze(0)  # shape: (1, T, 5)

# Apply self-attention
attention = SimpleSelfAttention(dim=5)
attention_output = attention(input_tensor).squeeze(0).detach().numpy()  # shape: (T, 5)

# KMeans clustering on the attention output
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(attention_output)
df['cluster'] = cluster_labels
cluster_onehot = pd.get_dummies(df['cluster'], prefix='cluster')





features = ['wind_speed', 'chlor_rfu', 'do_wtemp', 'par']
target_sim = 'do_sim'
target_raw = 'do_raw'

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df[features])
scaled_target_sim = scaler.fit_transform(df[[target_sim]])
scaled_target_raw = scaler.fit_transform(df[[target_raw]])

# Combine features with cluster labels
X_all = np.concatenate([scaled_features, cluster_onehot.values], axis=1)

# =============================================
# 4. Create Time Series Dataset
# =============================================
def create_sequences(data, target, window=30):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(target[i+window])
    return np.array(X), np.array(y)

window_size = 30
X_seq, y_seq_sim = create_sequences(X_all, scaled_target_sim, window_size)
_, y_seq_raw = create_sequences(X_all, scaled_target_raw, window_size)

# =============================================
# 5. Define Encoder-Decoder LSTM
# =============================================
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        _, (h, c) = self.lstm(x)
        return h, c

class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(output_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, h, c, seq_len):
        input_step = torch.zeros((h.size(1), 1, 1)).to(h.device)
        outputs = []
        for _ in range(seq_len):
            out, (h, c) = self.lstm(input_step, (h, c))
            input_step = self.fc(out)
            outputs.append(input_step)
        return torch.cat(outputs, dim=1)

class Seq2Seq(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.decoder = Decoder(hidden_dim, output_dim)

    def forward(self, x):
        h, c = self.encoder(x)
        return self.decoder(h, c, seq_len=1)

# =============================================
# 6. Train Function
# =============================================
def train_model(model, dataloader, optimizer, criterion, epochs=10):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for xb, yb in dataloader:
            optimizer.zero_grad()
            out = model(xb).squeeze()
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {epoch_loss / len(dataloader):.4f}")

# =============================================
# 7. Pre-train and Fine-tune
# =============================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)
y_sim_tensor = torch.tensor(y_seq_sim, dtype=torch.float32).squeeze().to(device)
y_raw_tensor = torch.tensor(y_seq_raw, dtype=torch.float32).squeeze().to(device)

batch_size = 64
train_loader_sim = DataLoader(TensorDataset(X_tensor, y_sim_tensor), batch_size=batch_size, shuffle=True)
train_loader_raw = DataLoader(TensorDataset(X_tensor, y_raw_tensor), batch_size=batch_size, shuffle=True)

model = Seq2Seq(input_dim=X_seq.shape[2], hidden_dim=64, output_dim=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

print("Pre-training on do_sim...")
train_model(model, train_loader_sim, optimizer, criterion, epochs=10)

print("Fine-tuning on do_raw...")
train_model(model, train_loader_raw, optimizer, criterion, epochs=10)

# =============================================
# 8. Evaluate
# =============================================
model.eval()
with torch.no_grad():
    preds_raw = model(X_tensor).cpu().squeeze().numpy()
    true_raw = y_seq_raw.squeeze()
    mse = mean_squared_error(true_raw, preds_raw)
    mae = mean_absolute_error(true_raw, preds_raw)
    print(f"Evaluation on do_raw: MSE = {mse:.4f}, MAE = {mae:.4f}")
  
preds_raw_original = scaler.inverse_transform(preds_raw.reshape(-1, 1)).flatten()
true_raw_original = scaler.inverse_transform(true_raw.reshape(-1, 1)).flatten()
import matplotlib.pyplot as plt

df['datetime'] = pd.to_datetime(df['datetime'])

plt.figure(figsize=(14, 6))
for cluster in sorted(df['cluster'].unique()):
    subset = df[df['cluster'] == cluster]
    plt.plot(subset['datetime'], subset['do_raw'], '.', label=f'Cluster {cluster}', alpha=0.6)
plt.title("do_raw Time Series Colored by Cluster")
plt.xlabel("Date")
plt.ylabel("Dissolved Oxygen (do_raw)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('figs/do_cluster.png')
plt.show()


plt.figure(figsize=(14, 6))
plt.plot(df['datetime'].values[window_size:], true_raw_original, label="Observed DO (do_raw)", color='black')
plt.plot(df['datetime'].values[window_size:], preds_raw_original, label="Predicted DO", color='red', alpha=0.7)
plt.plot(df['datetime'].values[window_size:], df['do_sim'].values[window_size:], label="PB Predicted DO", color='blue', alpha=0.7)
plt.title("Model vs Observed Dissolved Oxygen")
plt.xlabel("Date")
plt.ylabel("Dissolved Oxygen")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('figs/lstm_do.png')
plt.show()


import shap
from sklearn.linear_model import LinearRegression

# Step 1: Flatten the LSTM sequence input for SHAP (average over time window)
X_flat = X_seq.mean(axis=1)  # shape: (n_samples, n_features)

# Step 2: Create a wrapper model for SHAP (surrogate model)
# We'll use the predicted output from the LSTM as target
model.eval()
with torch.no_grad():
    y_pred = model(torch.tensor(X_seq, dtype=torch.float32).to(device)).cpu().squeeze().numpy()

# Step 3: Train surrogate model
surrogate = LinearRegression()
surrogate.fit(X_flat, y_pred)

# Step 4: Run SHAP analysis
explainer = shap.Explainer(surrogate, X_flat)
shap_values = explainer(X_flat)

# Step 5: Plot SHAP summary
feature_names = df[features].columns.tolist() + ['cluster_0', 'cluster_1', 'cluster_2']

print("X_flat shape:", X_flat.shape)
print("Number of feature names:", len(feature_names))


shap.summary_plot(shap_values, features=X_flat, feature_names=feature_names, show=False)
plt.savefig('figs/shapley.png')
