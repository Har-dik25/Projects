# Weather Data Pipeline

This project implements a continuous data pipeline using the **OpenWeatherMap API**. It follows the 6-step process:
1. Collect Data -> 2. Store -> 3. Clean -> 4. Transform -> 5. Statistics -> 6. Visualize

## Prerequisites
- Python 3.x
- `pip` installed libraries: `requests`, `pandas`, `matplotlib`, `schedule`

## Installation
Run the following command to install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Pipeline
Run the script:
```bash
python weather_pipeline.py
```
- The script will run immediately and then repeat every 60 seconds.
- It will create/append to `weather_data.csv`.
- It will update `weather_trend.png` with a new graph of temperature over time.

## Other APIs for Practice
See `apis_for_practice.txt` for a curated list of open-source/free APIs for building data pipelines.
