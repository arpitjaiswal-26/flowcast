import pandas as pd
import numpy as np

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates all engineered feature sets specified in PRD Section 9.2.
    Ensures temporal ordering within segment groups to prevent data leakage.
    """
    df = df.copy()
    
    # Ensure proper datetime sorting
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["road_id", "datetime"]).reset_index(drop=True)
    
    # ----------------------------------------------------
    # 1. Temporal & Cyclical Encodings
    # ----------------------------------------------------
    hour = df["datetime"].dt.hour
    dow = df["datetime"].dt.dayofweek
    
    df["sin_hour"] = np.sin(2 * np.pi * hour / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * hour / 24.0)
    df["sin_dow"] = np.sin(2 * np.pi * dow / 7.0)
    df["cos_dow"] = np.cos(2 * np.pi * dow / 7.0)
    
    df["is_weekend"] = dow.isin([5, 6]).astype(int)
    df["is_peak_period"] = hour.isin([7, 8, 9, 10, 17, 18, 19, 20]).astype(int)
    
    # ----------------------------------------------------
    # 2. Lag Features (per segment)
    # ----------------------------------------------------
    for lag in [1, 2, 4, 48]:
        df[f"vol_lag_{lag}"] = df.groupby("road_id")["traffic_volume"].shift(lag)
        if "avg_speed" in df.columns:
            df[f"speed_lag_{lag}"] = df.groupby("road_id")["avg_speed"].shift(lag)
            
    # ----------------------------------------------------
    # 3. Rolling Window Statistics (4 and 8 steps)
    # ----------------------------------------------------
    for window in [4, 8]:
        # Closed='left' guarantees shift to avoid using current timestep target
        df[f"vol_roll_mean_{window}"] = (
            df.groupby("road_id")["traffic_volume"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        df[f"vol_roll_std_{window}"] = (
            df.groupby("road_id")["traffic_volume"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).std().fillna(0))
        )
        
    # ----------------------------------------------------
    # 4. Capacity & Headroom Calculations
    # ----------------------------------------------------
    capacity = df["road_capacity"] if "road_capacity" in df.columns else 800
    df["vc_ratio"] = (df["traffic_volume"] / capacity).round(3)
    df["capacity_headroom"] = capacity - df["traffic_volume"]
    
    # ----------------------------------------------------
    # 5. Weather Flags & Interactions
    # ----------------------------------------------------
    if "rainfall" in df.columns:
        df["rain_flag"] = (df["rainfall"] > 0.0).astype(int)
    else:
        df["rain_flag"] = df["weather_condition"].isin(["Rain", "Rainy"]).astype(int)
        
    if "visibility" in df.columns:
        df["low_visibility_flag"] = (df["visibility"] < 1000.0).astype(int)
    else:
        df["low_visibility_flag"] = df["weather_condition"].isin(["Fog", "Foggy"]).astype(int)
        
    # ----------------------------------------------------
    # 6. Calendar & Interaction Terms
    # ----------------------------------------------------
    holiday = df["public_holiday"] if "public_holiday" in df.columns else 0
    event = df["event_flag"] if "event_flag" in df.columns else 0
    
    df["holiday_peak_interaction"] = (holiday.astype(int) * df["is_peak_period"]).astype(int)
    df["event_peak_interaction"] = (event.astype(int) * df["is_peak_period"]).astype(int)
    
    # Drop warm-up NA rows created by lag calculations
    df = df.dropna(subset=["vol_lag_1", "vol_lag_2"]).reset_index(drop=True)
    
    return df

if __name__ == "__main__":
    import os
    raw_path = os.path.join("data", "processed", "processed_corridor_data.csv")
    if os.path.exists(raw_path):
        data = pd.read_csv(raw_path)
        data_featured = build_features(data)
        data_featured.to_csv(raw_path, index=False)
        print("✅ Features engineered and saved to processed_corridor_data.csv!")