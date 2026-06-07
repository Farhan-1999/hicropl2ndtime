import json
import re
from pathlib import Path
from copy import deepcopy

# Input files
filenames_txt = Path("sorted_filenames.txt")
original_metadata_json = Path("original_metadata.json")

# Output file
output_json = Path("patch_metadata.json")

# Pattern example:
# GF2_PMS1__L1A0000564539-MSS1_y00000_x00000.tif
patch_pattern = re.compile(r"^(?P<original_name>.+)_y\d+_x\d+\.[^.]+$")

# Load original metadata
with open(original_metadata_json, "r", encoding="utf-8") as f:
    original_metadata = json.load(f)

# Read patch filenames
with open(filenames_txt, "r", encoding="utf-8") as f:
    patch_filenames = [line.strip() for line in f if line.strip()]

patch_metadata = {}
missing_originals = []

for patch_name in patch_filenames:
    match = patch_pattern.match(patch_name)

    if not match:
        print(f"Skipping invalid patch filename format: {patch_name}")
        continue

    original_base = match.group("original_name")

    # Try possible original image keys
    possible_keys = [
        original_base,
        original_base + ".tif",
        original_base + ".tiff",
    ]

    found_key = None
    for key in possible_keys:
        if key in original_metadata:
            found_key = key
            break

    if found_key is None:
        missing_originals.append(patch_name)
        continue

    # Assign original image metadata to patch
    patch_metadata[patch_name] = deepcopy(original_metadata[found_key])

# Save output JSON
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(patch_metadata, f, indent=4, ensure_ascii=False)

print(f"Saved patch metadata to: {output_json}")
print(f"Total patch filenames: {len(patch_filenames)}")
print(f"Successfully matched patches: {len(patch_metadata)}")
print(f"Missing matches: {len(missing_originals)}")

if missing_originals:
    with open("missing_originals.txt", "w", encoding="utf-8") as f:
        for name in missing_originals:
            f.write(name + "\n")

    print("Missing patch names saved to: missing_originals.txt")