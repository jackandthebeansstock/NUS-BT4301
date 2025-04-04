from airflow import DAG
from airflow.decorators import task, dag
from datasets import load_dataset, load_from_disk
import pandas as pd
import os
import datetime as dt
import numpy as np
import time
import json
from ast import literal_eval
from transformers import pipeline
import torch
from google.cloud import bigquery
from collections import Counter
import random
from google.oauth2 import service_account

# Directory to save extracted and transformed data
OUTPUT_DIR = os.path.join(os.getcwd(), "dataset")
os.makedirs(OUTPUT_DIR, exist_ok=True)
TRAINING_CUTOFF_DATE = dt.datetime(2019,1,1)
TRAINING_CUTOFF_REVIEWS_LOWER = 10
TRAINING_CUTOFF_REVIEWS_UPPER = 50

@dag(
    dag_id='kindle_store_etl_taskflow',
    start_date=dt.datetime(2025, 3, 18),
    schedule_interval='@weekly',
    catchup=False,
)
def kindle_store_etl():
    """DAG for extracting, transforming, and loading Kindle Store data."""
    
    @task
    def extract_review_data():
        """Extract Kindle Store reviews from Hugging Face."""
        print(torch.backends.mps.is_available()) 
        # Load Kindle Store reviews
        reviews_dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_Kindle_Store", trust_remote_code=True,num_proc=64,split = 'full')
        training_yr = time.mktime(TRAINING_CUTOFF_DATE.timetuple())*1000        

        reviews_path = os.path.join(OUTPUT_DIR, "kindle_reviews")
        
        # filtered_review = reviews_dataset.filter(lambda example: example['timestamp'] > training_yr and example['verified_purchase'])

        # Print some info about the filtered dataset
        # print(f"Number of users after filtering: {len(filtered_review)}")
        # print("First few entries:", filtered_review[:5])

        columns_to_keep = ['rating', 'title', 'text', 'parent_asin', 'user_id', 'timestamp','verified_purchase']
        filtered_review = reviews_dataset.select_columns(columns_to_keep)

        filtered_review.save_to_disk(reviews_path)
        
        print(f"Extracted data saved to {reviews_path}")
        
        # Return file paths for downstream tasks
        return {"reviews_path": reviews_path}
    
    @task
    def extract_book_data():
        """Extract Kindle Store metadata from Hugging Face."""
        
        # Load Kindle Store reviews
        metadata_dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_meta_Kindle_Store", trust_remote_code=True,num_proc=64,split = 'full')
        
        columns_to_keep = ['title', 'average_rating', 'rating_number', 'price', 'store', 'categories', 'details', 'parent_asin']
        metadata_dataset = metadata_dataset.select_columns(columns_to_keep)
        metadata_path = os.path.join(OUTPUT_DIR, "kindle_metadata")

        metadata_dataset.save_to_disk(metadata_path)
        
        print(f"Extracted data saved to {metadata_path}")
        
        # Return file paths for downstream tasks
        return {"metadata_path": metadata_path}

    @task
    def transform_review_data(file_path: dict, batch_size=128, device=None):
        """Loads a dataset from disk, applies sentiment analysis in batches, and saves the transformed dataset efficiently."""
        reviews_path = file_path.get('reviews_path')
        transformed_review_path = os.path.join(OUTPUT_DIR, "transformed_reviews")

        # Load dataset directly in Arrow format
        reviews_dataset = load_from_disk(reviews_path)
        training_yr = time.mktime(TRAINING_CUTOFF_DATE.timetuple())*1000
        transformed_dataset = reviews_dataset.filter(lambda example: (example['timestamp'] > training_yr) and example['verified_purchase'])

        user_counts = Counter(transformed_dataset['user_id'])
        eligible_users = [user for user, count in user_counts.items() if (count >= TRAINING_CUTOFF_REVIEWS_LOWER and count < TRAINING_CUTOFF_REVIEWS_UPPER)]
        eligible_users = set(eligible_users)
        transformed_dataset = transformed_dataset.filter(lambda example: example['user_id'] in eligible_users)

        columns_to_keep = ['rating', 'title', 'text', 'parent_asin', 'user_id', 'timestamp']
        transformed_dataset = transformed_dataset.select_columns(columns_to_keep)
        num_users = len(set(transformed_dataset['user_id']))
        num_reviews = len(transformed_dataset)
        print(f"Final dataset: {num_reviews} reviews from {num_users} unique users")
        print(f"Average reviews per user: {num_reviews / num_users:.2f}")

        # # Auto-detect device
        # if device is None:
        #     device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        # device_id =  0 if device.type == 'mps' else -1
        # print(f"Using device: {device}")

        # # Load sentiment analysis model
        # model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        # sentiment_analyzer = pipeline(
        #     "sentiment-analysis",
        #     model=model_name,
        #     tokenizer=model_name,
        #     device=device_id,
        #     batch_size=batch_size
        # )

        # # Define function to process text in batches
        # def compute_sentiment(batch):
        # # Combine title and text, ensuring missing values don't break the pipeline
        #     combined_texts = [
        #         ((t if t is not None else "") + " " + (r if r is not None else ""))[:500]
        #         for t, r in zip(batch["title"], batch["text"])
        #     ]
        #     results = sentiment_analyzer(combined_texts)

        #     # Convert POSITIVE/NEGATIVE to a numeric score
        #     batch["sentiment_score"] = [
        #         res["score"] if res["label"] == "POSITIVE" else (1 - res["score"]) for res in results
        #     ]

        #     # Remove title and text while keeping other columns
        #     batch.pop("title", None)
        #     batch.pop("text", None)
        #     return batch

        # # Apply transformation efficiently using map() with batched=True
        # transformed_dataset = transformed_dataset.map(compute_sentiment, batched=True, batch_size=batch_size)

        transformed_dataset.save_to_disk(transformed_review_path)

        print(f"Transformed dataset saved at: {transformed_review_path}")
        return {"reviews_path": transformed_review_path}

    @task
    def transform_book_data(file_path: dict):
        metadata_path = file_path.get('metadata_path')
        metadata_dataset = load_from_disk(metadata_path)
        transformed_book_path = os.path.join(OUTPUT_DIR, "transformed_books")

        def process_store(store_str):
            if isinstance(store_str, str):
                # Remove 'Format' part and keep only the author
                if 'Format:' in store_str:
                    return store_str.split('Format:')[0].replace('(Author)', '').strip()
                return store_str
            return store_str

        # Define a function to extract the last value from 'categories' as 'genre'
        def extract_genre(categories):
            if isinstance(categories, list) and len(categories) > 0:
                return categories[-1]
            return None

        # Define a function to process the 'details' string and convert to dict
        def process_details(details_str):
            if isinstance(details_str, str):
                try:
                    # Try parsing as JSON
                    return json.loads(details_str)
                except json.JSONDecodeError:
                    try:
                        # Fallback to literal_eval if it's a Python-style string dict
                        return literal_eval(details_str)
                    except (ValueError, SyntaxError):
                        return {}  # Return empty dict if parsing fails
            return {}  # Return empty dict if not a string

        # Transform the dataset and keep only the specified columns
        def transform_example(example):
            details_dict = process_details(example['details'])
            return {
                'title': example['title'],
                'average_rating': example['average_rating'],
                'rating_number': example['rating_number'],
                'price': example['price'],
                'store': process_store(example['store']),
                'genre': extract_genre(example['categories']),
                'parent_asin': example['parent_asin'],
                'publisher': details_dict.get('Publisher', ''),
                'publication_date': details_dict.get('Publication date',''),
                'language': details_dict.get('Language','')
            }

        # Apply the transformation and remove all original columns
        metadata_dataset = metadata_dataset.map(transform_example, remove_columns=metadata_dataset.column_names,num_proc = 32)
        
        metadata_dataset.save_to_disk(transformed_book_path)
        print(f"Transformed data saved to {transformed_book_path}")
        
        # Return file paths for downstream tasks
        return {"metadata_path": transformed_book_path}

    @task
    def load_data(review_file_path: dict, books_file_path: dict):
        """Load transformed Kindle Store data into Google BigQuery."""
        credentials = service_account.Credentials.from_service_account_file("/opt/airflow/secrets/big-query-key.json")
        client = bigquery.Client(credentials=credentials, project="bt4301-454516")

        # Define table schemas
        review_schema = [
            bigquery.SchemaField("rating", "FLOAT"),
            bigquery.SchemaField("title", "STRING"),
            bigquery.SchemaField("text", "STRING"),
            bigquery.SchemaField("parent_asin", "STRING"),
            bigquery.SchemaField("user_id", "STRING"),
            bigquery.SchemaField("timestamp", "TIMESTAMP")
        ]

        metadata_schema = [
            bigquery.SchemaField("title", "STRING"),
            bigquery.SchemaField("average_rating", "FLOAT"),
            bigquery.SchemaField("rating_number", "INTEGER"),
            bigquery.SchemaField("price", "FLOAT"),
            bigquery.SchemaField("store", "STRING"),
            bigquery.SchemaField("genre", "STRING"),
            bigquery.SchemaField("parent_asin", "STRING"),
            bigquery.SchemaField("publisher", "STRING"),
            bigquery.SchemaField("publication_date", "STRING"),
            bigquery.SchemaField("language", "STRING"),
        ]

        # Load datasets from disk
        reviews_dataset = load_from_disk(review_file_path["reviews_path"])
        metadata_dataset = load_from_disk(books_file_path["metadata_path"])

        # Convert to pandas DataFrames
        reviews_df = reviews_dataset.to_pandas()
        metadata_df = metadata_dataset.to_pandas()

        # Convert timestamp to datetime (assuming it's in milliseconds)
        reviews_df["timestamp"] = pd.to_datetime(reviews_df["timestamp"], unit="ms")
        metadata_df['price'] = pd.to_numeric(metadata_df['price'], errors='coerce')
        metadata_df['price'].fillna(0.0, inplace=True)
        # Define BigQuery table references
        project_id = "bt4301-454516"  # Replace with your GCP project ID
        dataset_id = "kindle_store"
        reviews_table_id = f"{project_id}.{dataset_id}.reviews"
        metadata_table_id = f"{project_id}.{dataset_id}.metadata"

        # Load reviews data
        reviews_job_config = bigquery.LoadJobConfig(
            schema=review_schema,
            write_disposition="WRITE_TRUNCATE",  # Overwrite table if it exists
        )
        reviews_load_job = client.load_table_from_dataframe(reviews_df, reviews_table_id, job_config=reviews_job_config)
        reviews_load_job.result()  # Wait for the job to complete
        print(f"Loaded {reviews_load_job.output_rows} rows into {reviews_table_id}")

        # Load metadata data
        metadata_job_config = bigquery.LoadJobConfig(
            schema=metadata_schema,
            write_disposition="WRITE_TRUNCATE",
        )
        metadata_load_job = client.load_table_from_dataframe(metadata_df, metadata_table_id, job_config=metadata_job_config)
        metadata_load_job.result()
        print(f"Loaded {metadata_load_job.output_rows} rows into {metadata_table_id}")

    # @task
    # def load_data(file_path_review: dict, file_path_book: dict):
    #     """Load the transformed data (example: log completion)."""
    #     reviews_path = file_path_review["reviews_path"]
    #     metadata_path = file_path_book["metadata_path"]
        
    #     print(f"Data loaded from {reviews_path} and {metadata_path}")
    #     # Add logic here to load into a database or other destination if needed
    
    # Define task flow
    extracted_book_path = extract_book_data()
    extracted_review_path = extract_review_data()
    transformed_book_path = transform_book_data(extracted_book_path)
    transformed_review_path = transform_review_data(extracted_review_path)
    load_data(transformed_review_path, transformed_book_path)

    
# Instantiate the DAG
kindle_dag = kindle_store_etl()