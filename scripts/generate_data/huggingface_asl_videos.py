import os
import shutil
import pandas as pd
from collections import defaultdict
from huggingface_hub import list_repo_files, hf_hub_download

# config
repo_id = "akasheroor/American-Sign-Language-Dataset"
csv_path = "../data/Aslense Dataset.csv"
output_root = "../data/raw/asl_hf"

target_words = ["yes", "no", "thankyou", "pay", "bye", "hello", "price", "cost", "money"]

# set true to actually download
DOWNLOAD_FILES = True

# create output folders
os.makedirs(output_root, exist_ok=True)

for word in target_words:
    os.makedirs(os.path.join(output_root, word), exist_ok=True)

print(f"Created base folder: {output_root}")
print("Created class folders:")
for word in target_words:
    print(f" - {word}")

# load CSV
df = pd.read_csv(csv_path)
df.columns = [c.strip().lower() for c in df.columns]

word_col = "word"
video_col = "videos"

df[word_col] = df[word_col].astype(str).str.strip().str.lower()
df[video_col] = df[video_col].astype(str).str.strip()

filtered = df[df[word_col].isin(target_words)].copy()

print("\nCounts from CSV:")
for word in target_words:
    count = (filtered[word_col] == word).sum()
    print(f"{word}: {count}")

# list repo files
print("\nListing repo files from Hugging Face...")
repo_files = list_repo_files(repo_id=repo_id, repo_type="dataset")

# build filename -> full repo path map
file_map = {}
for path in repo_files:
    filename = os.path.basename(path)
    file_map[filename] = path

# match csv filenames to repo files
found_counts = defaultdict(int)
missing_counts = defaultdict(int)
matched_rows = []

for _, row in filtered.iterrows():
    word = row[word_col]
    filename = row[video_col]

    if filename in file_map:
        found_counts[word] += 1
        matched_rows.append((word, filename, file_map[filename]))
    else:
        missing_counts[word] += 1

print("\nMatched files in Hugging Face repo:")
for word in target_words:
    print(f"{word}: found={found_counts[word]}, missing={missing_counts[word]}")

# download matched files
if DOWNLOAD_FILES:
    print("\nDownloading matched files...")

    for word, filename, remote_path in matched_rows:
        word_dir = os.path.join(output_root, word)
        out_path = os.path.join(word_dir, filename)

        # Skip if already downloaded
        if os.path.exists(out_path):
            continue

        try:
            print(f"Downloading {word}: {filename}")
            cached_file = hf_hub_download(
                repo_id=repo_id,
                filename=remote_path,
                repo_type="dataset"
            )
            shutil.copy2(cached_file, out_path)

        except Exception as e:
            print(f"Failed to download {filename}: {e}")

    print("\nDownload complete.")

# video count in folders
print("\nFinal file counts in output folders:")
for word in target_words:
    word_dir = os.path.join(output_root, word)
    file_count = len([
        f for f in os.listdir(word_dir)
        if os.path.isfile(os.path.join(word_dir, f))
    ])
    print(f"{word}: {file_count}")
