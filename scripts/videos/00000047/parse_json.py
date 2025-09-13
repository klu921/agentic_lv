import json
import os


with open('00000047_answers_reformatted.json', 'r') as f:
    msg = f.read()

# The file contains a JSON string that itself contains JSON
# So we need to decode it twice
first_decode = json.loads(msg)  # This gives us a string
actual_json = json.loads(first_decode)  # This gives us the actual data

print(f"Loaded {len(actual_json)} entries")
print("First entry:", json.dumps(actual_json[0], indent=2)[:500] if actual_json else "No entries")

with open('00000047_answers_dumped.json', 'w') as f:
    json.dump(actual_json, f, indent=2)
    