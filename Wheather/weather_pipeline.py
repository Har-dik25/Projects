import requests
import pandas as pd
import matplotlib.pyplot as plt
import schedule
import time
import os
from datetime import datetime

# Configuration
# Configuration
DEFAULT_API_KEY = "4bea72ed1cb92c57be60f214f9d69f87"
DEFAULT_CITY = "London"
DATA_FILE = "weather_data.csv"
VISUALIZATION_FILE = "weather_trend.png"

def collect_data(city=DEFAULT_CITY, lat=None, lon=None, api_key=DEFAULT_API_KEY):
    """Step 1: Collect Data from OpenWeatherMap API"""
    if lat is not None and lon is not None:
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}"
    else:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            print(f"Error fetching data: {response.status_code}")
            return None
    except Exception as e:
        print(f"Exception during collection: {e}")
        return None

def transform_data(raw_data):
    """Step 3 & 4: Clean and Transform Data
       - Extract relevant fields
       - Convert Kelvin to Celsius
       - Add Timestamp
    """
    if not raw_data:
        return None
    
    # Transformation: Extract only necessary fields
    main_data = raw_data.get('main', {})
    weather_desc = raw_data.get('weather', [{}])[0].get('description', 'unknown')
    
    # Transformation: Kelvin to Celsius
    temp_k = main_data.get('temp')
    temp_c = temp_k - 273.15 if temp_k else None
    
    # Transformation: Coordinates
    coord = raw_data.get('coord', {})
    
    # Transformation: Wind
    wind = raw_data.get('wind', {})

    transformed = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'city': raw_data.get('name'),
        'temperature_c': round(temp_c, 2) if temp_c is not None else None,
        'humidity': main_data.get('humidity'),
        'pressure': main_data.get('pressure'),
        'description': weather_desc,
        'lat': coord.get('lat'),
        'lon': coord.get('lon'),
        'wind_speed': wind.get('speed'),
        'wind_deg': wind.get('deg')
    }
    return transformed

def store_data(cleaned_data):
    """Step 2: Store Data (CSV)"""
    if not cleaned_data:
        return
    
    df = pd.DataFrame([cleaned_data])
    
    # If file exists, append without header; else write with header
    mode = 'a' if os.path.exists(DATA_FILE) else 'w'
    header = not os.path.exists(DATA_FILE)
    
    df.to_csv(DATA_FILE, mode=mode, header=header, index=False)
    print(f"Data stored: {cleaned_data['timestamp']} - {cleaned_data['temperature_c']}°C")

def calculate_and_visualize(city=DEFAULT_CITY):
    """Step 5 & 6: Calculate Statistics and Visualize"""
    if not os.path.exists(DATA_FILE):
        return

    # Load Data
    try:
        df = pd.read_csv(DATA_FILE)
        
        # Step 3 (Cleaning redux): Drop any rows with missing values
        df.dropna(inplace=True)
        
        # Filter by city
        df_city = df[df['city'] == city]

        # Step 5: Statistics
        if len(df_city) < 2:
            return # Need at least 2 points to visualize trends
            
        avg_temp = df_city['temperature_c'].mean()
        max_temp = df_city['temperature_c'].max()
        min_temp = df_city['temperature_c'].min()
        
        print(f"Stats -> Avg: {avg_temp:.2f}°C, Max: {max_temp:.2f}°C, Min: {min_temp:.2f}°C")
        
        # Step 6: Visualization
        plt.figure(figsize=(10, 5))
        plt.plot(pd.to_datetime(df_city['timestamp']), df_city['temperature_c'], marker='o', linestyle='-', color='b', label='Temperature (°C)')
        plt.title(f"Real-time Temperature Trend in {city}")
        plt.xlabel("Time")
        plt.ylabel("Temperature (°C)")
        plt.grid(True)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(VISUALIZATION_FILE)
        plt.close() # Close to free memory
        print(f"Visualization updated: {VISUALIZATION_FILE}")
        
    except Exception as e:
        print(f"Error in visualization: {e}")

def run_pipeline(city=DEFAULT_CITY, lat=None, lon=None):
    if lat and lon:
        print(f"Running pipeline step for coordinates {lat}, {lon}...")
    else:
        print(f"Running pipeline step for {city}...")
        
    raw = collect_data(city=city, lat=lat, lon=lon)
    processed = transform_data(raw)
    
    # If we searched by coords, the city name in processed data might be different/more accurate
    # Ensure we use that city name for visualization consistency if needed
    if processed:
        actual_city = processed.get('city', city)
        store_data(processed)
        calculate_and_visualize(city=actual_city)
        return actual_city # Return the city name for dashboard use
    return city

def main():
    print("Starting Weather Data Pipeline...")
    print("Press Ctrl+C to stop.")
    
    # Run immediately once
    run_pipeline()
    
    # Schedule to run every 10 seconds for demonstration purposes
    schedule.every(10).seconds.do(run_pipeline)
    
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()
