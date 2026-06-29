import pandas as pd
import numpy as np



#load dataset
raw_df = pd.read_csv("csv/gait_phase_LH_features.csv")
raw_data = raw_df.values.astype(np.float32)
raw_tensor = torch.from_numpy(raw_data).float()
num_time_steps = raw_tensor.shape[0]
print(f"Number of time steps: {num_time_steps}")
step_size = 125
step_size = int(step_size)
print(f"Step size: {step_size}")
print(f"Input tensor shape: {raw_tensor.shape}")

scaler = StandardScaler()
normalized_tensor = scaler.fit_transform(raw_tensor)

#kmeans
kmeans = KMeans(n_clusters=4, random_state=42, init='k-means++')
hidden_states = kmeans.fit_predict(normalized_tensor)

window_size = 125 #quarter of a second
for start_idx in range(0, num_time_steps - window_size + 1, step_size):
    end_idx = start_idx + window_size
    window_features = normalized_tensor[start_idx:end_idx]
    print(hidden_states[start_idx:end_idx])
    latest_state = hidden_states[end_idx - 1]

    print(f"Window [{start_idx:03d}:{end_idx:03d}] -> "
          f"Latest Inferred State: {latest_state}")




