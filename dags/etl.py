from airflow import DAG
from airflow.decorators import task, dag
from datetime import datetime
from datasets import load_dataset
import pandas as pd
import os
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
import numpy as np

# Directory to save extracted and transformed data
OUTPUT_DIR = os.path.join(os.getcwd(), "dataset")
os.makedirs(OUTPUT_DIR, exist_ok=True)

@dag(
    dag_id='kindle_store_etl_taskflow',
    start_date=datetime(2025, 3, 14),
    schedule_interval='@weekly',
    catchup=False,
)
def kindle_store_etl():
    """DAG for extracting, transforming, and loading Kindle Store data."""
    
    @task
    def extract_data():
        """Extract Kindle Store reviews and metadata from Hugging Face."""
        # Load Kindle Store reviews
        reviews_dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_Kindle_Store", trust_remote_code=True,num_proc=8,split = 'full')
        metadata_dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_meta_Kindle_Store", trust_remote_code=True,num_proc=8,split = 'full')        

        # Since data is already in Parquet format, we just need the paths
        reviews_path = os.path.join(OUTPUT_DIR, "kindle_reviews")
        metadata_path = os.path.join(OUTPUT_DIR, "kindle_metadata")
        
        reviews_dataset.to_parquet(reviews_path)
        metadata_dataset.to_parquet(metadata_path)
        
        print(f"Extracted data saved to {reviews_path} and {metadata_path}")
        
        # Return file paths for downstream tasks
        return {"reviews_path": reviews_path, "metadata_path": metadata_path}
    
    @task
    def transform_data(file_paths: dict):
        """Transform the extracted data by selecting specific columns and performing optimizations."""
        
        # Define columns to keep
        reviews_columns = ["title", "text", "ID", "productID", "customerID", "timestamp", "verified_purchase"]
        metadata_columns = ["title", "author", "average_rating", "rating_number", "price", 
                           "publisherstore", "categories", "details", "ProductID"]
        
        # Process reviews data in chunks to handle large datasets
        transformed_reviews_path = os.path.join(OUTPUT_DIR, "transformed_kindle_reviews.parquet")
        
        # Read and process reviews in chunks with pyarrow for better memory efficiency
        reviews_table = pq.read_table(file_paths["reviews_path"])
        
        # Select only needed columns (case-insensitive matching)
        available_columns = [col.lower() for col in reviews_table.column_names]
        selected_columns = []
        column_mapping = {}
        
        for col in reviews_columns:
            if col.lower() in available_columns:
                idx = available_columns.index(col.lower())
                actual_col = reviews_table.column_names[idx]
                selected_columns.append(actual_col)
                column_mapping[actual_col] = col  # For renaming if needed
        
        # Select columns and perform transformations
        reviews_table = reviews_table.select(selected_columns)
        
        # Convert to pandas for complex transformations if needed
        reviews_df = reviews_table.to_pandas()
        
        # Data cleaning and transformations
        # Convert timestamp to datetime if it's not already
        if 'timestamp' in reviews_df.columns:
            reviews_df['timestamp'] = pd.to_datetime(reviews_df['timestamp'], errors='coerce')
        
        # Convert verified_purchase to boolean if needed
        if 'verified_purchase' in reviews_df.columns:
            reviews_df['verified_purchase'] = reviews_df['verified_purchase'].astype(bool)
            
        # Drop duplicates and nulls in critical columns
        reviews_df = reviews_df.drop_duplicates(subset=['ID'])
        reviews_df = reviews_df.dropna(subset=['ID', 'productID'])
        
        # Process metadata with similar approach
        transformed_metadata_path = os.path.join(OUTPUT_DIR, "transformed_kindle_metadata.parquet")
        
        metadata_table = pq.read_table(file_paths["metadata_path"])
        
        # Select metadata columns (case-insensitive matching)
        available_meta_columns = [col.lower() for col in metadata_table.column_names]
        selected_meta_columns = []
        meta_column_mapping = {}
        
        for col in metadata_columns:
            if col.lower() in available_meta_columns:
                idx = available_meta_columns.index(col.lower())
                actual_col = metadata_table.column_names[idx]
                selected_meta_columns.append(actual_col)
                meta_column_mapping[actual_col] = col  # For renaming if needed
        
        metadata_table = metadata_table.select(selected_meta_columns)
        metadata_df = metadata_table.to_pandas()
        
        # Clean up metadata
        # Normalize product IDs to ensure consistency
        if 'ProductID' in metadata_df.columns:
            metadata_df['ProductID'] = metadata_df['ProductID'].str.strip()
        
        # Convert price to numeric
        if 'price' in metadata_df.columns:
            metadata_df['price'] = pd.to_numeric(metadata_df['price'].str.replace('[$,]', '', regex=True), errors='coerce')
        
        # Convert ratings to numeric
        if 'average_rating' in metadata_df.columns:
            metadata_df['average_rating'] = pd.to_numeric(metadata_df['average_rating'], errors='coerce')
        
        if 'rating_number' in metadata_df.columns:
            metadata_df['rating_number'] = pd.to_numeric(metadata_df['rating_number'], errors='coerce')
        
        # Drop duplicates in metadata
        metadata_df = metadata_df.drop_duplicates(subset=['ProductID'])
        
        # Save transformed data with optimized compression
        reviews_df.to_parquet(
            transformed_reviews_path, 
            compression='snappy',  # Good balance of speed and compression
            index=False
        )
        
        metadata_df.to_parquet(
            transformed_metadata_path,
            compression='snappy',
            index=False
        )
        
        print(f"Transformed data saved to {transformed_reviews_path} and {transformed_metadata_path}")
        
        # Return transformed file paths
        return {"reviews_path": transformed_reviews_path, "metadata_path": transformed_metadata_path}
    
    @task
    def load_data(file_paths: dict):
        """Load the transformed data (example: log completion)."""
        reviews_path = file_paths["reviews_path"]
        metadata_path = file_paths["metadata_path"]
        
        print(f"Data loaded from {reviews_path} and {metadata_path}")
        # Add logic here to load into a database or other destination if needed
    
    # Define task flow
    extracted_files = extract_data()
    transformed_files = transform_data(extracted_files)
    load_data(transformed_files)

# Instantiate the DAG
kindle_dag = kindle_store_etl()