import csv

input_file = 'weather_data.csv'
output_file = 'weather_data_fixed.csv'

expected_cols = 10
header = ['timestamp', 'city', 'temperature_c', 'humidity', 'pressure', 'description', 'lat', 'lon', 'wind_speed', 'wind_deg']

with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
     open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    # Write new header
    writer.writerow(header)
    
    # Skip original header if it exists
    try:
        next(reader) 
    except StopIteration:
        pass

    for row in reader:
        # Pad row to expected length
        if len(row) < expected_cols:
            row += [''] * (expected_cols - len(row))
        # Trim if too long (unlikely here but safe)
        elif len(row) > expected_cols:
             row = row[:expected_cols]
             
        writer.writerow(row)

import os
# Replace original file
if os.path.exists(output_file):
    os.replace(output_file, input_file)
    print("Fixed CSV file schema.")
