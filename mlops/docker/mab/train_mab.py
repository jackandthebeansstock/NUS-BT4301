import pandas as pd
import mlflow
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import pickle
import logging
import joblib 


# Setup logging
logger = logging.getLogger(__name__)
logger.handlers = []
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('/app/models/training_mab.log', mode='w')
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
    logger.info("Starting MAB training script")

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    with mlflow.start_run(run_name="MAB_Training"):
        # Log data metrics
        mlflow.log_metric("num_rows", customer_df.shape[0])
        mlflow.log_metric("num_users", customer_df["user_id"].nunique())
        mlflow.log_metric("num_items", customer_df["parent_asin"].nunique())

        # Prepare data
        logger.info("Preparing data for MAB...")
        # Encode categories
        category_encoder = LabelEncoder()
        all_categories = set()
        for cat_str in product_df['categories']:
            cats = eval(cat_str) if isinstance(cat_str, str) and cat_str.startswith('[') else [cat_str]
            all_categories.update(cats)
        category_encoder.fit(list(all_categories))
        product_df['category_id'] = product_df['categories'].apply(
            lambda x: category_encoder.transform(eval(x) if isinstance(x, str) and x.startswith('[') else [x])[0]
        )
        asin_to_category = dict(zip(product_df['parent_asin'], product_df['category_id']))

        # Merge category info
        customer_df['category_id'] = customer_df['parent_asin'].map(asin_to_category).fillna(0).astype(int)
        
        # Encode user and item IDs
        user_encoder = LabelEncoder()
        item_encoder = LabelEncoder()
        customer_df["user_idx"] = user_encoder.fit_transform(customer_df["user_id"])
        customer_df["item_idx"] = item_encoder.fit_transform(customer_df["parent_asin"])

        # Features: user_idx, item_idx, category_id
        X = customer_df[["user_idx", "item_idx", "category_id"]].values
        y = customer_df["rating"].fillna(1.0).values  # Assume implicit feedback

        # Train model
        logger.info("Starting MAB training...")
        model = LinearRegression()
        model.fit(X, y)
        score = model.score(X, y)
        mlflow.log_metric("r2_score", score)
        logger.info(f"MAB model R2 score: {score}")

        # Save model and encoders
        model_path = "/app/models/mab_model.pkl"
        user_encoder_path = "/app/models/mab_user_encoder.pkl"
        item_encoder_path = "/app/models/mab_item_encoder.pkl"
        category_encoder_path = "/app/models/mab_category_encoder.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        joblib.dump(user_encoder, user_encoder_path)
        joblib.dump(item_encoder, item_encoder_path)
        joblib.dump(category_encoder, category_encoder_path)

        mlflow.log_artifact(model_path)
        mlflow.log_artifact(user_encoder_path)
        mlflow.log_artifact(item_encoder_path)
        mlflow.log_artifact(category_encoder_path)
        logger.info("Model and artifacts logged to MLflow")

        logger.info("MAB training completed")
        print("MAB training completed")

if __name__ == "__main__":
    logger.info("Script started")
    main()
    logger.info("Script completed")