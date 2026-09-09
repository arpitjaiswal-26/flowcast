import os
import numpy as np
import pandas as pd

def load_and_clean_data(raw_dir='data/raw', output_dir='data/processed'):
    print("[1/4] Ingesting and cleaning raw data streams...")
    
    traffic_path = os.path.join(raw_dir, 'traffic_sensor_log.csv')
    weather_path = os.path.join(raw_dir, 'weather_observations.csv')
    cal_path = os.path.join(raw_dir, 'calendar_events.csv')
    
    df_t = pd.read_csv(traffic_path)
    df_w = pd.read_csv(weather_path)
    df_c = pd.read_csv(cal_path)

    # Harmonize weather dates & clean text fields
    if 'date' in df_w.columns:
        df_w['date'] = pd.to_datetime(df_w['date'], dayfirst=True, errors='coerce').dt.strftime('%Y-%m-%d')
    if 'weather_condition' in df_w.columns:
        df_w['weather_condition'] = df_w['weather_condition'].astype(str).str.title().str.strip()

    # Deduplicate traffic logs
    df_t = df_t.drop_duplicates(subset=['road_id', 'date', 'time']).reset_index(drop=True)

    # Temporal Alignment: Hourly weather to 30-min traffic grain
    df_t['hour'] = df_t['time'].str[:2] + ":00"
    
    # Merge sources
    df = pd.merge(df_t, df_w, left_on=['weather_station_id', 'date', 'hour'], 
                  right_on=['station_id', 'date', 'time'], how='left', suffixes=('', '_w'))
    df = pd.merge(df, df_c, on='date', how='left')

    # Quarantine sensor-fault outliers
    df.loc[df['traffic_volume'] < 0, 'traffic_volume'] = np.nan
    df.loc[df['avg_speed'] > 200, 'avg_speed'] = np.nan
    df.loc[df['occupancy'] > 100, 'occupancy'] = np.nan

    # Fill numeric traffic values per segment and fallbacks
    for col in ['traffic_volume', 'avg_speed', 'occupancy']:
        if col in df.columns:
            df[col] = df.groupby(['road_id', 'time'])[col].transform(lambda x: x.fillna(x.median()))
            df[col] = df.groupby('road_id')[col].transform(lambda x: x.ffill().bfill())
            df[col] = df[col].fillna(df[col].median())

    # Derive missing congestion level via V/C ratio
    vc_ratio_raw = df['traffic_volume'] / (df['road_capacity'] / 2.0)
    derived_cong = pd.cut(
        vc_ratio_raw,
        bins=[-np.inf, 0.4999, 0.7999, 0.9999, np.inf],
        labels=['Free-flow', 'Moderate', 'Heavy', 'Severe']
    )
    if 'congestion_level' in df.columns:
        df['congestion_level'] = df['congestion_level'].fillna(derived_cong)
    else:
        df['congestion_level'] = derived_cong

    # Weather & Calendar defaults
    if 'temperature' in df.columns:
        df['temperature'] = df['temperature'].fillna(df['temperature'].median())
    if 'rainfall' in df.columns:
        df['rainfall'] = df['rainfall'].fillna(0.0)
    if 'visibility' in df.columns:
        df['visibility'] = df['visibility'].fillna(df['visibility'].median())
    if 'public_holiday' in df.columns:
        df['public_holiday'] = df['public_holiday'].fillna(0).astype(int)
    if 'event_flag' in df.columns:
        df['event_flag'] = df['event_flag'].fillna(0).astype(int)

    # Fill non-numeric text columns (e.g. event names/types) so dropna doesn't purge rows
    for col in df.select_dtypes(include=['object', 'category']).columns:
        df[col] = df[col].fillna('None')

    # Build Datetime Index & Sort
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    df = df.sort_values(['road_id', 'datetime']).reset_index(drop=True)

    # Cyclical & Temporal Features
    df['hour_int'] = df['datetime'].dt.hour
    df['minute_int'] = df['datetime'].dt.minute
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    time_float = df['hour_int'] + df['minute_int'] / 60.0
    df['sin_hour'] = np.sin(2 * np.pi * time_float / 24.0)
    df['cos_hour'] = np.cos(2 * np.pi * time_float / 24.0)
    df['sin_dow'] = np.sin(2 * np.pi * df['day_of_week'] / 7.0)
    df['cos_dow'] = np.cos(2 * np.pi * df['day_of_week'] / 7.0)

    # Lag & Rolling Features
    df['vol_lag1'] = df.groupby('road_id')['traffic_volume'].shift(1)
    df['vol_lag2'] = df.groupby('road_id')['traffic_volume'].shift(2)
    df['vol_roll4_mean'] = df.groupby('road_id')['traffic_volume'].transform(lambda x: x.shift(1).rolling(4).mean())
    df['speed_lag1'] = df.groupby('road_id')['avg_speed'].shift(1)

    # Backfill initial window lag NaNs per group to preserve full row count
    lag_cols = ['vol_lag1', 'vol_lag2', 'vol_roll4_mean', 'speed_lag1']
    for col in lag_cols:
        df[col] = df.groupby('road_id')[col].transform(lambda x: x.bfill())

    # Derived Engineering Targets & Ratios
    df['vc_ratio'] = df['traffic_volume'] / (df['road_capacity'] / 2.0)
    if 'accident_count' in df.columns:
        df['accident_risk'] = (df['accident_count'].fillna(0) > 0).astype(int)

    # Targeted drop on critical modeling features only
    feature_cols = ['traffic_volume', 'avg_speed', 'occupancy'] + lag_cols
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, 'processed_corridor_data.csv')
    df.to_csv(out_file, index=False)
    print(f"-> Processed dataset saved to {out_file} ({df.shape[0]} rows, {df.shape[1]} columns)")
    return df

if __name__ == "__main__":
    load_and_clean_data()