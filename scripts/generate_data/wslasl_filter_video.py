import json
import os

# Make sure output folder exists
os.makedirs("../data/WLASL/output", exist_ok=True)

# Load WLASL metadata
with open("../data/WLASL/archive/WLASL_v0.3.json", "r", encoding="utf-8") as f:
    wlasl_data = json.load(f)

# Map each target word to possible gloss variants in WLASL
target_map = {
    "yes": ["yes", "yeah"],
    "no": ["no", "nope"],
    "thank you": ["thank you", "thank-you", "thankyou", "thanks"],
    "pay": ["pay", "payment"],
    "goodbye": ["goodbye", "bye"],
    "hello": ["hello", "hi", "hey"],
    "price": ["price"],
    "cost": ["cost"],
    "money": ["money", "cash"]
}

filtered_dataset = {key: [] for key in target_map}
matched_glosses = {key: [] for key in target_map}

for entry in wlasl_data:
    gloss = entry["gloss"].lower().strip()
    for target_word, variants in target_map.items():
        variants_lower = [v.lower() for v in variants]
        if gloss in variants_lower:
            filtered_dataset[target_word].extend(
                [instance["video_id"] for instance in entry["instances"]]
            )
            matched_glosses[target_word].append(gloss)

# Remove duplicates
for key in filtered_dataset:
    filtered_dataset[key] = list(set(filtered_dataset[key]))
    matched_glosses[key] = list(set(matched_glosses[key]))

# Save filtered video IDs
with open("../data/WLASL/output/asl_videos.json", "w", encoding="utf-8") as f:
    json.dump(filtered_dataset, f, indent=4)

# Save matched glosses
with open("../data/WLASL/output/asl_matched_glosses.json", "w", encoding="utf-8") as f:
    json.dump(matched_glosses, f, indent=4)

# Print summary
print("Filtered WLASL results:\n")
for word in target_map:
    print(f"{word}: {len(filtered_dataset[word])} videos | matched glosses: {matched_glosses[word]}")
