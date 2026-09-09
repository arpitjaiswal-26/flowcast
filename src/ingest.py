import os
import logging
import pandas as pd
import numpy as np

# Ensure report output directory exists
os.makedirs("reports", exist_ok=True)
os.makedirs("data/interim", exist_ok=True)

# Configure detailed quarantine log
logging.basicConfig(
    filename='reports/ingestion_quarantine.log',
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    force=True
)

def log_quarantined_records(df: pd.DataFrame, mask: pd.Series, table_name: str, reason: str):
    """Logs individual quarantined records with full context to preserve audit trail."""
    quarantined = df[~mask]
    if not quarantined.empty:
        for idx, row in quarantined.iterrows():
            identifier = row.get('road_id', row.get('weather_station_id', f'Row_{idx}'))
            timestamp = row.get('time', row.get('timestamp', 'N/A'))
            logging.info(
                f"QUARANTINE [{table_name}] | Key: {identifier} | Time: {timestamp} | Reason: {reason} | Raw Row: {row.to_dict()}"
            )
    return quarantined

def validate_and_ingest_sensor_log(file_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ingests, validates, and logs rejected traffic sensor records (FR-01, FR-02)."""
    df = pd.read_csv(file_path)
    
    # 1. Check for missing critical timestamps
    df['parsed_timestamp'] = pd.to_datetime(df['time'], errors='coerce')
    valid_time_mask = df['parsed_timestamp'].notnull()
    log_quarantined_records(df, valid_time_mask, 'traffic_sensor_log', 'Unparseable Timestamp')
    
    # 2. Check physical numeric boundaries
    valid_range_mask = (
        (df['traffic_volume'] >= 0) & (df['traffic_volume'] <= 5000) &
        (df['avg_speed'] >= 0) & (df['avg_speed'] <= 200) &
        (df['occupancy'] >= 0.0) & (df['occupancy'] <= 100.0)
    )
    log_quarantined_records(df, valid_range_mask, 'traffic_sensor_log', 'Out-of-Bounds Sensor Telemetry')
    
    # Combine validity criteria
    overall_mask = valid_time_mask & valid_range_mask
    valid_df = df[overall_mask].copy().drop(columns=['parsed_timestamp'])
    quarantined_df = df[~overall_mask].copy()
    
    return valid_df, quarantined_df

def validate_and_ingest_weather(file_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ingests, validates, and logs rejected weather records (FR-01, FR-02)."""
    df = pd.read_csv(file_path)
    
    # Handle mixed timestamp parsing
    df['parsed_timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
    valid_time_mask = df['parsed_timestamp'].notnull()
    log_quarantined_records(df, valid_time_mask, 'weather_observations', 'Unparseable Timestamp')
    
    valid_range_mask = (
        (df['temperature'] >= -20) & (df['temperature'] <= 55) &
        (df['rainfall'] >= 0) &
        (df['visibility'] >= 0)
    )
    log_quarantined_records(df, valid_range_mask, 'weather_observations', 'Out-of-Bounds Weather Measurements')
    
    overall_mask = valid_time_mask & valid_range_mask
    valid_df = df[overall_mask].copy().drop(columns=['parsed_timestamp'])
    quarantined_df = df[~overall_mask].copy()
    
    return valid_df, quarantined_df

def run_ingestion_pipeline(data_dir: str = "data/raw"):
    """Runs ingestion across all source tables and saves quarantine snapshots."""
    sensors_valid, sensors_quarantine = validate_and_ingest_sensor_log(os.path.join(data_dir, "traffic_sensor_log.csv"))
    weather_valid, weather_quarantine = validate_and_ingest_weather(os.path.join(data_dir, "weather_observations.csv"))
    
    # Save isolated invalid records for inspection
    sensors_quarantine.to_csv("data/interim/quarantined_sensors.csv", index=False)
    weather_quarantine.to_csv("data/interim/quarantined_weather.csv", index=False)
    
    print(f"Ingestion complete.")
    print(f" - Sensors: {len(sensors_valid)} valid, {len(sensors_quarantine)} quarantined.")
    print(f" - Weather: {len(weather_valid)} valid, {len(weather_quarantine)} quarantined.")
    print(f" - Detailed log generated at: reports/ingestion_quarantine.log")

if __name__ == "__main__":
    run_ingestion_pipeline()