import pandas as pd
import mlflow
import os
import networkx as nx
import pickle
import logging

# Setup logging
logger = logging.getLogger(__name__)
logger.handlers = []
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('/app/models/training_random_walk.log', mode='w')
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# Load data
customer_file = "/app/data/customer_clean.xlsx"
product_file = "/app/data/product_clean.xlsx"
logger.info(f"Loading data from {customer_file} and {product_file}...")
try:
    customer_df = pd.read_excel(customer_file)
    product_df = pd.read_excel(product_file)
    logger.info(f"Loaded customer_df: {customer_df.shape}, product_df: {product_df.shape}")
except Exception as e:
    logger.error(f"Error loading data: {e}")
    raise

# Main function
def main():
    logger.info("Starting Random Walk training script")

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    with mlflow.start_run(run_name="Random_Walk_Training"):
        # Log data metrics
        mlflow.log_metric("num_rows", customer_df.shape[0])
        mlflow.log_metric("num_users", customer_df["user_id"].nunique())
        mlflow.log_metric("num_items", customer_df["parent_asin"].nunique())

        # Prepare data
        logger.info("Preparing data for Random Walk...")
        G = nx.Graph()
        G.add_nodes_from(customer_df["user_id"].unique(), bipartite=0)
        G.add_nodes_from(customer_df["parent_asin"].unique(), bipartite=1)
        edges = [(row["user_id"], row["parent_asin"], {"weight": row["rating"] if pd.notna(row["rating"]) else 1.0})
                 for _, row in customer_df.iterrows()]
        G.add_edges_from(edges)

        # Compute centrality (simplified recommendation)
        logger.info("Computing centrality...")
        centrality = nx.degree_centrality(G)
        mlflow.log_metric("num_nodes", G.number_of_nodes())
        mlflow.log_metric("num_edges", G.number_of_edges())

        # Save graph
        model_path = "/app/models/random_walk_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({"graph": G, "centrality": centrality}, f)
        mlflow.log_artifact(model_path)

        logger.info("Random Walk training completed")
        print("Random Walk training completed")

if __name__ == "__main__":
    logger.info("Script started")
    main()
    logger.info("Script completed")