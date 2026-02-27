"""
Prepare 3 Clean F1 Datasets for the Dynamic ML Platform
========================================================
1. f1_lap_time_prediction.csv   → Regression
2. f1_podium_prediction.csv     → Classification
3. f1_driver_segmentation.csv   → Clustering
"""
import pandas as pd
import numpy as np
import os, shutil

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, 'datasets')
WC = os.path.join(RAW, 'f1_world_championship')
CLS = os.path.join(RAW, 'f1_classification')
CLU = os.path.join(RAW, 'f1_clustering')

# First, let's inspect what we have
print("=" * 60)
print("  INSPECTING RAW DATA")
print("=" * 60)

for folder in [WC, CLS, CLU]:
    if os.path.exists(folder):
        for f in sorted(os.listdir(folder)):
            if f.endswith('.csv'):
                fp = os.path.join(folder, f)
                df = pd.read_csv(fp, nrows=3)
                rows = len(pd.read_csv(fp))
                print(f"  {os.path.relpath(fp, RAW):50s} | {rows:>7,} rows | cols: {list(df.columns)}")
    else:
        print(f"  MISSING: {folder}")

print("\n" + "=" * 60)
print("  BUILDING DATASET 1: Lap Time Prediction (Regression)")
print("=" * 60)

# Merge lap_times + results + races + circuits + constructors
lap_times = pd.read_csv(os.path.join(WC, 'lap_times.csv'))
results = pd.read_csv(os.path.join(WC, 'results.csv'))
races = pd.read_csv(os.path.join(WC, 'races.csv'))
circuits = pd.read_csv(os.path.join(WC, 'circuits.csv'))
constructors = pd.read_csv(os.path.join(WC, 'constructors.csv'))
drivers = pd.read_csv(os.path.join(WC, 'drivers.csv'))

# Merge lap times with race info
df1 = lap_times.merge(races[['raceId', 'year', 'round', 'circuitId', 'name']], on='raceId', how='left')
df1 = df1.merge(circuits[['circuitId', 'circuitRef', 'country']], on='circuitId', how='left')

# Get grid position and constructor from results
results_subset = results[['raceId', 'driverId', 'constructorId', 'grid', 'positionOrder']].drop_duplicates(subset=['raceId', 'driverId'])
df1 = df1.merge(results_subset, on=['raceId', 'driverId'], how='left')

# Get constructor name
df1 = df1.merge(constructors[['constructorId', 'constructorRef']], on='constructorId', how='left')

# Clean up - keep useful columns and rename
df1 = df1.rename(columns={
    'name': 'race_name',
    'milliseconds': 'lap_time_ms',
    'circuitRef': 'circuit',
    'constructorRef': 'constructor',
    'positionOrder': 'finish_position',
    'position': 'current_position',
})

# Select final columns
df1 = df1[['year', 'round', 'circuit', 'country', 'constructor', 'grid',
           'lap', 'current_position', 'lap_time_ms']].copy()

# Drop rows with missing lap_time_ms
df1 = df1.dropna(subset=['lap_time_ms'])
df1['lap_time_ms'] = df1['lap_time_ms'].astype(int)

# Sample to keep manageable (max 50K rows)
if len(df1) > 50000:
    df1 = df1.sample(50000, random_state=42).reset_index(drop=True)

df1.to_csv(os.path.join(RAW, 'f1_lap_time_prediction.csv'), index=False)
print(f"  ✅ Saved: f1_lap_time_prediction.csv | {df1.shape[0]:,} rows × {df1.shape[1]} cols")
print(f"     Columns: {list(df1.columns)}")
print(f"     Target: lap_time_ms (continuous → Regression)")

print("\n" + "=" * 60)
print("  BUILDING DATASET 2: Podium Finish Prediction (Classification)")
print("=" * 60)

# Use results + races + circuits + constructors
df2 = results.merge(races[['raceId', 'year', 'round', 'circuitId']], on='raceId', how='left')
df2 = df2.merge(circuits[['circuitId', 'circuitRef', 'country']], on='circuitId', how='left')
df2 = df2.merge(constructors[['constructorId', 'constructorRef']], on='constructorId', how='left')

# Create podium target (top 3 finish = 1, else = 0)
df2['podium'] = (df2['positionOrder'] <= 3).astype(int)

# Engineer features
df2 = df2.rename(columns={'circuitRef': 'circuit', 'constructorRef': 'constructor'})

# Select features
df2 = df2[['year', 'round', 'circuit', 'country', 'constructor', 'grid',
           'laps', 'podium']].copy()

# Clean
df2 = df2.dropna(subset=['grid', 'laps'])
df2 = df2[df2['grid'] > 0]  # Remove DNS

df2.to_csv(os.path.join(RAW, 'f1_podium_prediction.csv'), index=False)
print(f"  ✅ Saved: f1_podium_prediction.csv | {df2.shape[0]:,} rows × {df2.shape[1]} cols")
print(f"     Columns: {list(df2.columns)}")
print(f"     Target: podium (0/1 → Classification)")
print(f"     Class balance: {df2['podium'].value_counts().to_dict()}")

print("\n" + "=" * 60)
print("  BUILDING DATASET 3: Driver Segmentation (Clustering)")
print("=" * 60)

# Build driver career stats from results
driver_stats = results.merge(races[['raceId', 'year']], on='raceId', how='left')
driver_stats = driver_stats.merge(drivers[['driverId', 'forename', 'surname']], on='driverId', how='left')
driver_stats['driver_name'] = driver_stats['forename'] + ' ' + driver_stats['surname']

# Aggregate per driver
agg = driver_stats.groupby(['driverId', 'driver_name']).agg(
    total_races=('raceId', 'nunique'),
    avg_grid=('grid', 'mean'),
    avg_finish=('positionOrder', 'mean'),
    best_finish=('positionOrder', 'min'),
    total_points=('points', 'sum'),
    avg_points_per_race=('points', 'mean'),
    podiums=('positionOrder', lambda x: (x <= 3).sum()),
    wins=('positionOrder', lambda x: (x == 1).sum()),
    dnf_count=('statusId', lambda x: (x != 1).sum()),
    career_years=('year', lambda x: x.max() - x.min() + 1),
).reset_index()

# Derived features
agg['podium_rate'] = (agg['podiums'] / agg['total_races'] * 100).round(2)
agg['win_rate'] = (agg['wins'] / agg['total_races'] * 100).round(2)
agg['dnf_rate'] = (agg['dnf_count'] / agg['total_races'] * 100).round(2)
agg['consistency'] = (100 - (agg['avg_finish'] - agg['best_finish']) / agg['avg_finish'] * 100).round(2)

# Only keep drivers with at least 5 races for meaningful clustering
df3 = agg[agg['total_races'] >= 5].copy()

# Select final columns
df3 = df3[['driver_name', 'total_races', 'avg_grid', 'avg_finish', 'best_finish',
           'total_points', 'avg_points_per_race', 'podiums', 'wins',
           'podium_rate', 'win_rate', 'dnf_rate', 'consistency', 'career_years']].copy()

df3 = df3.round(2)
df3.to_csv(os.path.join(RAW, 'f1_driver_segmentation.csv'), index=False)
print(f"  ✅ Saved: f1_driver_segmentation.csv | {df3.shape[0]:,} rows × {df3.shape[1]} cols")
print(f"     Columns: {list(df3.columns)}")
print(f"     No target → Clustering")

# Now clean up: remove old subfolders, keep only the 3 new CSVs
print("\n" + "=" * 60)
print("  CLEANING UP — Removing old subfolders")
print("=" * 60)

for folder in ['f1_world_championship', 'f1_classification', 'f1_clustering']:
    fp = os.path.join(RAW, folder)
    if os.path.exists(fp):
        shutil.rmtree(fp)
        print(f"  🗑️  Removed: {folder}/")

# List final datasets folder
print("\n" + "=" * 60)
print("  FINAL DATASETS FOLDER")
print("=" * 60)
for f in sorted(os.listdir(RAW)):
    if f.endswith('.csv'):
        fp = os.path.join(RAW, f)
        df = pd.read_csv(fp)
        print(f"  📄 {f:40s} | {df.shape[0]:>6,} rows × {df.shape[1]:>2} cols")

print("\n✅ Done! 3 datasets ready for the Dynamic ML Platform.")
