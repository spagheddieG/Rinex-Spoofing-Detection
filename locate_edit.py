import json

def find_edit_location():
    with open("test.json", "r", encoding="utf-8") as f:
        content = f.read()
        data = json.loads(content)

    # Just look at the first file in the dictionary
    filename = list(data.keys())[0]
    file_data = data[filename]
    
    if "indexed_records" not in file_data:
        print("No indexed_records found")
        return

    by_time = file_data["indexed_records"]["by_time"]
    sorted_epochs = sorted(by_time.keys())
    
    if len(sorted_epochs) < 2:
        print("Not enough epochs")
        return

    epoch1 = sorted_epochs[0]
    epoch2 = sorted_epochs[1]
    
    # Pick a satellite present in both
    sats1 = set(by_time[epoch1]["satellites"].keys())
    sats2 = set(by_time[epoch2]["satellites"].keys())
    common = list(sats1.intersection(sats2))
    
    if not common:
        print("No common satellites")
        return
        
    sat = common[0]
    
    print(f"Target Satellite: {sat}")
    print(f"Epoch 1: {epoch1}")
    print(f"Epoch 2: {epoch2}")
    
    val1 = by_time[epoch1]["satellites"][sat]["values"]
    val2 = by_time[epoch2]["satellites"][sat]["values"]
    
    print("\n--- Epoch 2 Values (Target for Edit) ---")
    print(json.dumps(val2, indent=None)) # Compact print to search for
    
    # Also print a snippet of the file content around these values to help locate them
    # This is a bit tricky with just json.load, so I'll just rely on the unique values
    
    print("\n--- Current IODE ---")
    print(f"Epoch 1 IODE: {val1.get('IODE')}")
    print(f"Epoch 2 IODE: {val2.get('IODE')}")

if __name__ == "__main__":
    find_edit_location()
