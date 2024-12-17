import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.stats import norm
import seaborn as sns
import logging
import sys
import os
from dotenv import load_dotenv

# Setup logging to both console and file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("stage4validate_bert_embeddings_test.log", mode='w'),
        logging.StreamHandler()
                    ])

def load_embeddings(file_path):
    """
    Load embeddings and metadata from CSV file.

    Args:
        file_path (str): Path to the input CSV file, received through stage 3
    
    Returns:
        DataFrame: Full data.
        np.ndarray: Embedding values.
        np.ndarray: Processed text.
    """
    logging.info("Loading embeddings and metadata...")
    df = pd.read_csv(file_path)

    # Verify these columns exist
    required_columns = ['Record ID', 'DOI', 'File Name', 'Processed Text']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # The first 4 columns are 'Record ID', 'DOI', 'File Name', 'Processed Text', then 'embedding_0'
    embedding_columns = df.columns[4:]
    embeddings = df[embedding_columns].values
    texts = df['Processed Text'].values

    logging.info(f"Loaded {len(embeddings)} embeddings and metadata and processed text.")
    return df, embeddings, texts

def compute_statistics(embeddings):
    """
    Compute mean, std deviation, and return Gaussian-like curve data (if the result looks like a bell, it's good).
    
    Arg:
        embeddings (np.ndarray): Embedding array.

    Returns:
        tuple: Mean and standard deviation.
    """
    mean = np.mean(embeddings)
    std_dev = np.std(embeddings)
    logging.info(f"Embedding Statistics - Mean: {mean:.6f}, Std Dev: {std_dev:.6f}")
    return mean, std_dev

def plot_gaussian(embeddings, output_path="embedding_distribution.png"):
    """
    Plot the distribution of embedding values and save to file.
    """
    logging.info("Generating Gaussian-like curve plot...")
    flat_embeddings = embeddings.flatten()
    sns.histplot(flat_embeddings, kde=True, stat="density", bins=50, color='blue')
    x = np.linspace(min(flat_embeddings), max(flat_embeddings), 100)
    plt.plot(x, norm.pdf(x, np.mean(flat_embeddings), np.std(flat_embeddings)), color='red')
    plt.title("Distribution of Embedding Values")
    plt.xlabel("Embedding Value")
    plt.ylabel("Density")
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Distribution plot saved to {output_path}")

def text_similarity_test(embeddings, texts, output_csv="text_similarity_report.csv"):
    """
    Compute pairwise cosine similarity for selected texts and save results.
    
    Args:
        embeddings (np.ndarray): Embedding array.
        texts (np.ndarray): Processed text.
        output_csv (str): Path to save similarity report.
    """
    logging.info("Performing text similarity tests...")
    selected_pairs = [(0, 1), (2, 3), (4, 5)]  # Adjust indices based on your data
    results = []

    for idx1, idx2 in selected_pairs:
        sim_score = cosine_similarity([embeddings[idx1]], [embeddings[idx2]])[0][0]
        results.append({
            "Text 1": texts[idx1][:100],
            "Text 2": texts[idx2][:100],
            "Cosine Similarity": sim_score
        })
        logging.info(f"Cosine Similarity ({idx1}, {idx2}): {sim_score:.4f}")

    report_df = pd.DataFrame(results)
    report_df.to_csv(output_csv, index=False)
    logging.info(f"Text similarity report saved to {output_csv}")

def clustering_analysis(embeddings, output_plot="kmeans_clusters.png"):
    """
    Perform KMeans clustering and visualize results.
    
    Args:
        embeddings (np.ndarray): Embedding array.
        output_plot (str): Path to save clustering plot.
    """
    logging.info("Performing KMeans clustering...")
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(embeddings)

    plt.figure(figsize=(10, 6))
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=clusters, cmap='viridis', s=10)
    plt.title("KMeans Clustering of BERT Embeddings")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.colorbar(label="Cluster")
    plt.savefig(output_plot)
    plt.close()
    logging.info(f"Clustering plot saved to {output_plot}")

def nearest_neighbor_test(embeddings, texts, k=3):
    """
    Perform nearest neighbor search for embeddings.

    Args:
        embeddings (np.ndarray): Embedding array.
        texts (np.ndarray): Processed text.
        k (int): Number of nearest neighbors.
    """
    logging.info("Performing nearest neighbor test...")
    nbrs = NearestNeighbors(n_neighbors=k, metric='cosine').fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)

    for i in range(5):  # Display results for first 5 texts
        logging.info(f"\nText {i}: {texts[i][:100]}...")
        for j in range(1, k):
            neighbor_idx = indices[i][j]
            logging.info(f"  Nearest Neighbor {j}: {texts[neighbor_idx][:100]}... (Distance: {distances[i][j]:.4f})")

def main():
    load_dotenv()

    # File paths
    input_file = os.getenv("BERT_EMBED_SUBSET_OUTPUT_PATH", "bert_embeddings_subset.csv")
    stats_output_csv = os.getenv("BERT_EMBED_SUBSET_STATS_PATH", "embedding_subset_statistics_report.csv")

    # Load embeddings and texts
    df, embeddings, texts = load_embeddings(input_file)

    # 1. Compute statistics and save
    mean, std_dev = compute_statistics(embeddings)
    plot_gaussian(embeddings, output_path="embedding_distribution.png")

    # 2. Perform text similarity tests
    text_similarity_test(embeddings, texts, output_csv="text_similarity_report.csv")

    # 3. Clustering analysis
    clustering_analysis(embeddings, output_plot="kmeans_clusters.png")

    # 4. Nearest neighbor analysis
    nearest_neighbor_test(embeddings, texts, k=4)

    # Save summary statistics
    stats_report = pd.DataFrame([{"Mean": mean, "Standard Deviation": std_dev}])
    stats_report.to_csv(stats_output_csv, index=False)
    logging.info(f"Embedding statistics report saved to {stats_output_csv}")

if __name__ == "__main__":
    main()