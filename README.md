# FlowCast: Intelligent Traffic Flow Prediction Platform

FlowCast is an end-to-end AI traffic forecasting system designed for the Northline Corridor (25 road segments). It ingests sensor telemetry, weather, and calendar signals to predict traffic volume, congestion severity, travel times, and accident risk across 30- to 120-minute horizons.

## System Architecture & Modules
- **M1: Ingestion & Validation**: Schema validation and quarantine rules.
- **M2: Cleaning & Wrangling**: Deduplication, outlier capping, and temporal joins.
- **M3: Feature Engineering**: Cyclical time encodings, multi-window lag features, rolling statistics, V/C ratio, and weather flags.
- **M5: Classical ML Engine**: Scikit-Learn & XGBoost models across 4 targets.
- **M6: Deep Learning Engine**: PyTorch LSTM sequence forecaster built from scratch.
- **M7: Analytics Dashboard**: 9 interactive operational views in Streamlit.

## Model Benchmark Results
| Model Architecture | Task | Volume RMSE | MAPE |
| :--- | :--- | :--- | :--- |
| Gradient Boosting (XGB) | Classical Best | 59.84 | 10.95% |
| PyTorch LSTM Network | Deep Learning | 71.58 | 12.89% |

## Quickstart & Reproduction
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   