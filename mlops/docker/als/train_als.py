import pandas as pd
import mlflow
import os
from implicit.als import AlternatingLeastSquares
from scipy.sparse import coo_matrix
import pickle
import logging

# Setup logging
logger = logging.getLogger(__name__)
logger.handlers = []
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('/app/models/training_als.log', mode='w')
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# Load data
customer_file = "/app/data/customer_clean.xlsx"
logger.info(f"Loading customer data from {customer_file}...")
try:
    customer_df = pd.read_excel(customer_file)
    logger.info(f"Loaded customer_df with shape: {customer_df.shape}")
except Exception as e:
    logger.error(f"Error loading customer data: {e}")
    raise

# Main function
def main():
    logger.info("Starting ALS training script")

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    with mlflow.start_run(run_name="ALS_Training"):
        # Log data metrics
        mlflow.log_metric("num_rows", customer_df.shape[0])
        mlflow.log_metric("num_columns", customer_df.shape[1])
        mlflow.log_metric("num_users", customer_df["user_id"].nunique())
        mlflow.log_metric("num_items", customer_df["parent_asin"].nunique())

        # Prepare data for ALS
        logger.info("Preparing data for ALS...")
        user_ids = customer_df["user_id"].unique()
        item_ids = customer_df["parent_asin"].unique()
        user_map = {uid: idx for idx, uid in enumerate(user_ids)}
        item_map = {iid: idx for idx, iid in enumerate(item_ids)}

        rows = customer_df["user_id"].map(user_map)
        cols = customer_df["parent_asin"].map(item_map)
        values = customer_df["rating"].fillna(1.0).astype(float)  # Assume implicit feedback if no rating
        user_item_matrix = coo_matrix((values, (rows, cols)), shape=(len(user_ids), len(item_ids)))

        # Train ALS model
        logger.info("Starting ALS training...")
        model = AlternatingLeastSquares(factors=100, regularization=0.01, iterations=15)
        model.fit(user_item_matrix)

        # Save model
        model_path = "/app/models/als_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({"model": model, "user_map": user_map, "item_map": item_map}, f)
        mlflow.log_artifact(model_path)

        # Log hyperparameters
        mlflow.log_param("factors", 100)
        mlflow.log_param("regularization", 0.01)
        mlflow.log_param("iterations", 15)

        logger.info("ALS training completed")
        print("ALS training completed")

if __name__ == "__main__":
    logger.info("Script started")
    main()
    logger.info("Script completed")