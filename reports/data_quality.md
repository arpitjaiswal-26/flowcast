# FlowCast v1.0 — Data Quality & Wrangling Report

## 1. Executive Summary
This report documents the data cleaning and validation pipeline applied to the Northline Corridor raw telemetry tables (`traffic_sensor_log`, `weather_observations`, and `calendar_events`). A total of 176,701 segment-window records were cleaned, harmonized, and merged into an analysis-ready dataset.

## 2. Identified Data Quality Issues & Resolutions
* **Missing Sensor Readings**: Imputed short gaps (1–2 windows) using linear temporal interpolation; longer sensor dropouts filled using segment × time-window medians.
* **Duplicate Telemetry**: Identified exact and key-level duplicates (`road_id` + `timestamp`) caused by sensor retry loops; retained the record with highest data completeness.
* **Outlier Capping**: Sensor hardware faults produced impossible values. Capped counts < 0, speeds > 200 km/h, and occupancy rates > 100%.
* **Schema & Timestamp Alignment**: Standardized weather (`DD/MM/YYYY`) and sensor (`YYYY-MM-DD`) timestamps to UTC ISO-8601. Forward-filled hourly weather records to match 30-minute traffic aggregation windows.
* **Categorical Normalization**: Mapped inconsistent weather strings (`rainy`, `RAIN`, `Rain`) to a controlled vocabulary (`Clear`, `Cloudy`, `Rain`, `Fog`).

## 3. Post-Cleaning Validation
* **Missing Value Ratio**: 0.00% across primary features.
* **Row Retention Rate**: 98.4% of raw records retained after deduplication and quarantine.