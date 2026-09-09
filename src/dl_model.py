import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class FlowCastLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super(FlowCastLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def create_sequences(features, targets, seq_length=8):
    xs, ys = [], []
    for i in range(len(features) - seq_length):
        xs.append(features[i:(i + seq_length)])
        ys.append(targets[i + seq_length])
    return np.array(xs), np.array(ys)

def train_dl_model(data_path='data/processed/processed_corridor_data.csv', output_dir='models'):
    print("[3/4] Building and Training PyTorch LSTM Sequence Model...")
    df = pd.read_csv(data_path)

    feature_cols = [
        'sin_hour', 'cos_hour', 'sin_dow', 'cos_dow', 'is_weekend',
        'vol_lag1', 'vol_lag2', 'vol_roll4_mean', 'speed_lag1',
        'temperature', 'rainfall', 'visibility'
    ]

    # Standardize features and targets for neural network stability
    feat_mean, feat_std = df[feature_cols].mean().values, df[feature_cols].std().values
    vol_mean, vol_std = df['traffic_volume'].mean(), df['traffic_volume'].std()

    norm_features = (df[feature_cols].values - feat_mean) / (feat_std + 1e-8)
    norm_targets = (df['traffic_volume'].values - vol_mean) / (vol_std + 1e-8)

    seq_length = 8  # 4 hours of 30-min history
    X_seq, y_seq = create_sequences(norm_features, norm_targets, seq_length=seq_length)

    split_idx = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FlowCastLSTM(input_dim=len(feature_cols), hidden_dim=64, num_layers=2).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    epochs = 5
    for epoch in range(epochs):
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            preds = model(bx).squeeze()
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()

    # Evaluation on Test Set
    model.eval()
    with torch.no_grad():
        test_x = torch.tensor(X_test, dtype=torch.float32).to(device)
        norm_preds = model(test_x).squeeze().cpu().numpy()

    # Inverse scale predictions back to original volume
    pred_volumes = norm_preds * vol_std + vol_mean
    true_volumes = y_test * vol_std + vol_mean

    dl_rmse = float(np.sqrt(np.mean((true_volumes - pred_volumes) ** 2)))
    dl_mae = float(np.mean(np.abs(true_volumes - pred_volumes)))
    dl_mape = float(np.mean(np.abs((true_volumes - pred_volumes) / np.maximum(true_volumes, 1))) * 100)

    metrics = {
        'LSTM_Volume_RMSE': dl_rmse,
        'LSTM_Volume_MAE': dl_mae,
        'LSTM_Volume_MAPE': dl_mape
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'dl_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    torch.save(model.state_dict(), os.path.join(output_dir, 'lstm_flowcast.pt'))
    print(f"-> Deep Learning Complete. PyTorch LSTM Test RMSE: {dl_rmse:.2f} (MAPE: {dl_mape:.2f}%)")
    return metrics

if __name__ == "__main__":
    train_dl_model()