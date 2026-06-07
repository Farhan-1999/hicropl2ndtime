import csv
import json
from pathlib import Path

input_csv = Path("image_to_admin.csv")          # change this
output_json = Path("output.json")      # change this

# Columns to exclude from the values
exclude_columns = {"image_name", "province", "prefecture", "county"}

data = {}

with open(input_csv, "r", encoding="utf-8-sig", newline="") as f:
    # Auto-detect delimiter: comma, tab, semicolon, etc.
    sample = f.read(2048)
    f.seek(0)

    dialect = csv.Sniffer().sniff(sample)
    reader = csv.DictReader(f, dialect=dialect)

    for row in reader:
        image_name = row["image_name"].strip()

        data[image_name] = {
            col: value.strip() if isinstance(value, str) else value
            for col, value in row.items()
            if col not in exclude_columns
        }

with open(output_json, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f"JSON saved to: {output_json}")
print(f"Total entries: {len(data)}")