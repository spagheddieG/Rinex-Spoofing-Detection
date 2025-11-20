import json
import os

file_path = r"C:\Users\egome\OneDrive\Documents\KBR\Shoelace\Rinex-Spoofing-Detection\test.json"

print(f"Reading {file_path}...")
with open(file_path, 'r') as f:
    data = json.load(f)

target_file = "bamf314a15.25n"
target_sv = "G07"
# The key in by_time is the file header time, verified in previous steps
by_time_key = "2025-11-10T00:31:16" 

print(f"Modifying {target_file}...")

# Modify 'by_time'
by_time = data[target_file].get("indexed_records", {}).get("by_time", {})
if by_time_key in by_time:
    satellites = by_time[by_time_key].get("satellites", {})
    if target_sv in satellites:
        print(f"Found record in 'by_time' for {target_sv} at {by_time_key}")
        values = satellites[target_sv].get("values", {})
        
        print(f"Original SVclockBias: {values.get('SVclockBias')}")
        print(f"Original IODE: {values.get('IODE')}")
        
        # Inject spoofing: Change Clock Bias, keep IODE same as previous file (61.0)
        # The previous file (bamf314a00.25n) had IODE 61.0 for G07.
        values["SVclockBias"] = -0.00005
        values["IODE"] = 61.0 
        values["IODC"] = 61.0
        
        print(f"New SVclockBias: {values.get('SVclockBias')}")
        print(f"New IODE: {values.get('IODE')}")
    else:
        print(f"Satellite {target_sv} not found in {by_time_key}")
else:
    print(f"Key {by_time_key} not found in by_time. Available keys: {list(by_time.keys())}")

# Also update flat just in case (using the TOC time which is 02:00:00)
flat_time = "2025-11-10T02:00:00"
records = data[target_file].get("indexed_records", {}).get("flat", [])
for record in records:
    if record.get("sv") == target_sv and record.get("time") == flat_time:
        print(f"Updating flat record for {flat_time}")
        record["values"]["SVclockBias"] = -0.00005
        record["values"]["IODE"] = 61.0
        record["values"]["IODC"] = 61.0

print("Writing changes...")
with open(file_path, 'w') as f:
    json.dump(data, f, indent=4)
print("Done.")
