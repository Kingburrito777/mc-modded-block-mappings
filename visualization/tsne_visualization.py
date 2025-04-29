import pandas as pd
import numpy as np
import ollama
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Tuple
import uuid
import re

# Configuration
CSV_FILE = "block_mapping/modded_block_map_full.csv"
OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "nomic-embed-text"
OUTPUT_HTML = "tsne_block_visualization.html"

# Step 1: Load the CSV file
def load_block_mappings(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} block mappings from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return pd.DataFrame()

# Step 2: Preprocess block names for embedding
def preprocess_block_name(block: str) -> str:
    # Handle modded block prefix (e.g., fantasyfurniture:decorations/muffins_blueberry)
    if ":" in block:
        block = block.split(":", 1)[1]  # Take part after mod prefix
    # Remove underscores, replace with spaces
    block_name = block.replace("_", " ").replace(".", " ")
    # Remove numbers
    block_name = re.sub(r'\d+', '', block_name).strip()
    # Remove minecraft/ prefix
    block_name = block_name.replace("minecraft/", "").strip()
    # Remove submod paths (e.g., decorations/)
    if "/" in block_name:
        block_name = block_name.split("/")[-1].strip()
    return block_name

# Step 3: Generate embeddings using Ollama
def get_embedding(text: str, client: ollama.Client) -> np.ndarray:
    try:
        response = client.embeddings(model=MODEL_NAME, prompt=text)
        return np.array(response["embedding"])
    except Exception as e:
        print(f"Error generating embedding for '{text}' (original: {text}): {e}")
        return None

# Step 4: Compute embeddings for all blocks
def compute_embeddings(blocks: List[str], original_blocks: List[str], client: ollama.Client) -> Tuple[np.ndarray, List[str], List[str]]:
    embeddings = []
    valid_blocks = []
    valid_originals = []
    for block, orig_block in zip(blocks, original_blocks):
        embedding = get_embedding(block, client)
        if embedding is not None:
            embeddings.append(embedding)
            valid_blocks.append(block)
            valid_originals.append(orig_block)
        else:
            print(f"Skipping '{orig_block}' due to embedding failure.")
    return np.array(embeddings), valid_blocks, valid_originals

# Step 5: Create t-SNE visualization
def create_tsne_visualization(df: pd.DataFrame):
    client = ollama.Client(host=OLLAMA_HOST)
    
    # Get block names and preprocess them
    modded_blocks = df["Modded Block"].tolist()
    vanilla_blocks = df["Vanilla Block"].tolist()
    modded_blocks_clean = [preprocess_block_name(block) for block in modded_blocks]
    vanilla_blocks_clean = [preprocess_block_name(block) for block in vanilla_blocks]
    all_blocks_clean = modded_blocks_clean + vanilla_blocks_clean
    all_blocks_original = modded_blocks + vanilla_blocks
    labels = ["Modded"] * len(modded_blocks) + ["Vanilla"] * len(vanilla_blocks)
    
    # Compute embeddings
    print("Computing embeddings for all blocks...")
    embeddings, valid_blocks, valid_originals = compute_embeddings(all_blocks_clean, all_blocks_original, client)
    valid_labels = [labels[i] for i in range(len(all_blocks_clean)) if all_blocks_clean[i] in valid_blocks]
    
    if len(embeddings) == 0:
        print("No valid embeddings generated. Exiting.")
        return
    
    # Apply t-SNE
    print("Applying t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    tsne_results = tsne.fit_transform(embeddings)
    
    # Create DataFrame for Plotly
    plot_df = pd.DataFrame({
        "x": tsne_results[:, 0],
        "y": tsne_results[:, 1],
        "Block": valid_originals,  # Use original names for hover
        "Type": valid_labels
    })
    
    # Create Plotly scatter plot
    print("Generating Plotly visualization...")
    fig = px.scatter(
        plot_df,
        x="x",
        y="y",
        color="Type",
        hover_data=["Block"],
        title="t-SNE Visualization of Modded and Vanilla Block Embeddings",
        color_discrete_map={"Modded": "#FF6692", "Vanilla": "#1F77B4"}
    )
    
    # Update layout
    fig.update_traces(marker=dict(size=8, opacity=0.8))
    fig.update_layout(
        xaxis_title="t-SNE Dimension 1",
        yaxis_title="t-SNE Dimension 2",
        legend_title="Block Type",
        hovermode="closest",
        template="plotly_white",
        width=800,
        height=600
    )
    
    # Save to HTML
    fig.write_html(OUTPUT_HTML)
    print(f"Visualization saved to {OUTPUT_HTML}")

# Main execution
def main():
    # Load data
    df = load_block_mappings(CSV_FILE)
    if df.empty:
        print("No data loaded. Exiting.")
        return
    
    # Create visualization
    create_tsne_visualization(df)

if __name__ == "__main__":
    main()
