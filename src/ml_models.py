import json
import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression as SklearnLR
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

class NumPyLinearRegression:
    """Explicit Linear Regression trained via Gradient Descent using NumPy."""
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.epochs):
            y_pred = np.dot(X, self.weights) + self.bias
            # Gradient computation: dW = (1/N) * X^T * (Xw + b - y)
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

def train_classical_models(data_path='data/processed/processed_corridor_data.csv', output_dir='models'):
    print("[2/4] Training Classical ML Suite & NumPy Gradient Descent...")
    df = pd.read_csv(data_path)

    feature_cols = [
        'sin_hour', 'cos_hour', 'sin_dow', 'cos_dow', 'is_weekend',
        'vol_lag1', 'vol_lag2', 'vol_roll4_mean', 'speed_lag1',
        'temperature', 'rainfall', 'visibility', 'public_holiday', 'event_flag'
    ]

    X = df[feature_cols].values
    y_vol = df['traffic_volume'].values
    
    cong_map = {'Free-flow': 0, 'Moderate': 1, 'Heavy': 2, 'Severe': 3}
    y_cong = df['congestion_level'].map(cong_map).values
    y_risk = df['accident_risk'].values

    # Time-based Train/Test Split (80% Train, 20% Test)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train_v, y_test_v = y_vol[:split_idx], y_vol[split_idx:]
    y_train_c, y_test_c = y_cong[:split_idx], y_cong[split_idx:]
    y_train_r, y_test_r = y_risk[:split_idx], y_risk[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. NumPy Custom Linear Regression
    numpy_lr = NumPyLinearRegression(lr=0.01, epochs=500)
    numpy_lr.fit(X_train_scaled, y_train_v)
    numpy_preds = numpy_lr.predict(X_test_scaled)
    numpy_rmse = float(np.sqrt(mean_squared_error(y_test_v, numpy_preds)))

    # 2. Sklearn Linear Regression
    sk_lr = SklearnLR()
    sk_lr.fit(X_train_scaled, y_train_v)
    sk_preds = sk_lr.predict(X_test_scaled)
    sk_rmse = float(np.sqrt(mean_squared_error(y_test_v, sk_preds)))

    # 3. Gradient Boosting Regressor (Volume Forecast)
    gbr = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    gbr.fit(X_train, y_train_v)
    gbr_preds = gbr.predict(X_test)
    gbr_rmse = float(np.sqrt(mean_squared_error(y_test_v, gbr_preds)))
    gbr_mae = float(mean_absolute_error(y_test_v, gbr_preds))
    gbr_mape = float(np.mean(np.abs((y_test_v - gbr_preds) / np.maximum(y_test_v, 1))) * 100)

    # 4. Random Forest Classifier (Congestion Level)
    rfc = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
    rfc.fit(X_train, y_train_c)
    rfc_preds = rfc.predict(X_test)
    macro_f1 = float(f1_score(y_test_c, rfc_preds, average='macro'))

    # 5. SVM Classifier (Accident Risk)
    svm = SVC(probability=True, random_state=42)
    # Downsample for faster SVM training on large datasets
    sample_size = min(15000, len(X_train_scaled))
    svm.fit(X_train_scaled[:sample_size], y_train_r[:sample_size])
    svm_probs = svm.predict_proba(X_test_scaled)[:, 1]
    roc_auc = float(roc_auc_score(y_test_r, svm_probs))

    metrics = {
        'NumPy_LR_RMSE': numpy_rmse,
        'Sklearn_LR_RMSE': sk_rmse,
        'GBR_Volume_RMSE': gbr_rmse,
        'GBR_Volume_MAE': gbr_mae,
        'GBR_Volume_MAPE': gbr_mape,
        'RF_Congestion_Macro_F1': macro_f1,
        'SVM_Accident_ROC_AUC': roc_auc,
        'Feature_Importances': dict(zip(feature_cols, gbr.feature_importances_.tolist()))
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'ml_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"-> ML Suite Complete. GBR Volume RMSE: {gbr_rmse:.2f} | RF Macro-F1: {macro_f1:.2f} | SVM ROC-AUC: {roc_auc:.2f}")
    return metrics

if __name__ == "__main__":
    train_classical_models()