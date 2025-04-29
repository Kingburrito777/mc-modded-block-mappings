import ollama
import csv
import re
import numpy as np
import json
from typing import Dict, List
from sklearn.metrics.pairwise import cosine_similarity

# File paths
# BLOCKS_CSV = "block_mapping/buildpaste_blocks.csv"  # Vanilla Minecraft blocks
ERROR_LOG = "error_log_unknown_blocks.txt"  # File with "Unknown block" errors
OUTPUT_CSV = "block_mapping/modded_block_map_full.csv"  # Output mappings

OLLAMA_HOST = "http://localhost:11434" # point to your Ollama server
MODEL_NAME = "nomic-embed-text"  # Nomic is the best!

# Step 1: Load vanilla blocks from blocks.csv
def load_vanilla_blocks(file_path: str) -> List[str]:
    vanilla_blocks = []
    try:
        with open(file_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                block = row[0].strip().replace('_', ' ')
                vanilla_blocks.append(block)
        print(f"Loaded {len(vanilla_blocks)} vanilla blocks.")
        return vanilla_blocks
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

# Step 2: Extract and clean modded blocks from error log
def extract_modded_blocks(file_path: str) -> List[str]:
    modded_blocks = {}
    try:
        with open(file_path, "r") as f:
            for line in f:
                match = re.search(r"Unknown block: (.+)", line.strip())
                if match:
                    block = match.group(1)
                    if ":" in block:
                        mod_name, block_name = block.split(":", 1)
                    else:
                        block_name = block
                    if mod_name == "minecraft":
                        continue
                    # Remove underscores, replace with spaces
                    block_name = block_name.replace("_", " ").replace(".", " ")
                    # remove numbers
                    block_name = re.sub(r'\d+', '', block_name).strip()
                    # remove minecraft/ in block name for diagonal mod
                    block_name = block_name.replace("minecraft/", "").strip()
                    # remove submod paths (e.g. ch/biomesoplenty/)
                    if "/" in block_name:
                        block_name = block_name.split("/")[-1].strip()
                    if block_name != '':
                        modded_blocks[match.group(1)] = block_name
        print(f"Found {len(modded_blocks)} unique modded blocks.")
        return modded_blocks
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return {}

# Step 3: Generate embeddings using Ollama
def get_embedding(text: str, client: ollama.Client) -> np.ndarray:
    try:
        response = client.embeddings(model=MODEL_NAME, prompt=text)
        return np.array(response["embedding"])
    except Exception as e:
        print(f"Error generating embedding for '{text}': {e}")
        return None

# Step 4: Compute embeddings for all blocks
def compute_embeddings(blocks: List[str], client: ollama.Client) -> Dict[str, np.ndarray]:
    embeddings = {}
    for block in blocks:
        embedding = get_embedding(block, client)
        if embedding is not None:
            embeddings[block] = embedding
    return embeddings

# Step 5: Find nearest neighbor using cosine similarity
def find_nearest_neighbor(modded_embedding: np.ndarray, vanilla_embeddings: Dict[str, np.ndarray]) -> str:
    max_similarity = -1
    nearest_block = None
    modded_vector = modded_embedding.reshape(1, -1)
    
    for vanilla_block, vanilla_embedding in vanilla_embeddings.items():
        vanilla_vector = vanilla_embedding.reshape(1, -1)
        similarity = cosine_similarity(modded_vector, vanilla_vector)[0][0]
        if similarity > max_similarity:
            max_similarity = similarity
            nearest_block = vanilla_block
    return nearest_block

# Step 6: Map modded blocks to vanilla blocks
def map_blocks(vanilla_blocks: Dict[str, str], modded_blocks: Dict[str, str]) -> Dict[str, str]:
    client = ollama.Client(host=OLLAMA_HOST)
    
    # Compute embeddings for vanilla blocks
    print("Computing embeddings for vanilla blocks...")
    vanilla_blocks_list = list(vanilla_blocks.keys())
    vanilla_embeddings = compute_embeddings(vanilla_blocks_list, client)
    if not vanilla_embeddings:
        print("Failed to compute vanilla embeddings.")
        return {}
    
    # Map modded blocks
    mappings = {}
    print("Mapping modded blocks to nearest vanilla blocks...")
    for i, modded_block in enumerate(modded_blocks):
        modded_embedding = get_embedding(modded_blocks[modded_block], client)
        if modded_embedding is not None:
            nearest_vanilla = find_nearest_neighbor(modded_embedding, vanilla_embeddings)
            if nearest_vanilla:
                mappings[modded_block] = vanilla_blocks[nearest_vanilla]
        else:
            print(f"Skipping '{modded_block}' due to embedding failure.")
            
        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1} blocks... Most recent mapping: {modded_block} -> {nearest_vanilla}")
    return mappings

# Step 7: Save mappings to CSV
def save_mappings(mappings: Dict[str, str], output_file: str):
    try:
        # First read existing mappings if file exists
        existing_mappings = {}
        # try:
        #     with open(output_file, "r", newline='') as csvfile:
        #         reader = csv.DictReader(csvfile)
        #         for row in reader:
        #             existing_mappings[row["Modded Block"]] = row["Vanilla Block"]
        # except FileNotFoundError:
        #     pass

        # # Merge existing and new mappings, keeping existing ones
        # for modded, vanilla in mappings.items():
        #     modded = modded.replace(' ', '_')
        #     if modded not in existing_mappings:
        #         existing_mappings[modded] = vanilla

        # Write merged mappings back to file
        with open(output_file, "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Modded Block", "Vanilla Block"])
            for modded, vanilla in mappings.items():
                writer.writerow([modded, vanilla])
        print(f"Saved mappings to {output_file}")
    except Exception as e:
        print(f"Error saving mappings: {e}")

# Main execution
def main():
    # Load data
    # vanilla_blocks = load_vanilla_blocks(BLOCKS_CSV)
    vanilla_blocks = {}
    with open("block_mapping/blocks.json", 'r') as f:
        for block in json.load(f):
            vanilla_blocks[block['displayName'].lower()] = block['name']

    print(f"Loaded {len(vanilla_blocks)} vanilla blocks.")

    modded_blocks = extract_modded_blocks(ERROR_LOG)

    if not vanilla_blocks or not modded_blocks:
        print("Failed to load blocks. Exiting.")
        return

    # Map blocks using nearest neighbor in embedding space
    mappings = map_blocks(vanilla_blocks, modded_blocks)
    
    if mappings:
        print("Mappings generated successfully:")
        for modded, vanilla in mappings.items():
            print(f"{modded} -> {vanilla}")
        save_mappings(mappings, OUTPUT_CSV)
    else:
        print("No mappings were generated.")

if __name__ == "__main__":
    main()