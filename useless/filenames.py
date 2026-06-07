import json
import random
from pathlib import Path

input_json = Path("patch_metadata.json")
output_json = Path("patch_metadata_shuffled.json")

# Optional: set seed for reproducible shuffle
# random.seed(42)

# Load JSON
with open(input_json, "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert entries to list and shuffle only entry order
items = list(data.items())
random.shuffle(items)

# Convert back to dictionary
shuffled_data = dict(items)

# Save shuffled JSON
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(shuffled_data, f, indent=4, ensure_ascii=False)

print(f"Shuffled JSON saved to: {output_json}")
print(f"Total entries: {len(shuffled_data)}")