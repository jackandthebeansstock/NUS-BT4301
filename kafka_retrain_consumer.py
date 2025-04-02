# kafka_retrain_consumer.py
from kafka import KafkaConsumer
import json
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import threading
import time
import mlflow
import torch
from your_model import BERT4Rec  # Import your model class
import joblib
import os

# Configuration
KAFKA_TOPIC = 'user_interactions'
BATCH_SIZE = 1000  # Number of interactions to collect before retraining
RETRAIN_INTERVAL = timedelta(hours=24)  # Minimum time between retrains
DB_PATH = 'books.db'
MODEL_ARTIFACTS = {
    'book_encoder': 'book_encoder.pkl',
    'category_encoder': 'category_encoder.pkl',
    'asin_to_category': 'asin_to_category.pkl'
}

class Retrainer:
    def __init__(self):
        self.consumer = KafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers=['localhost:9092'],
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            group_id='retrain-group',
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        
        self.last_retrain_time = datetime.min
        self.interaction_buffer = []
        self.lock = threading.Lock()
        
        # Load initial encoders
        self.book_encoder = joblib.load(MODEL_ARTIFACTS['book_encoder'])
        self.category_encoder = joblib.load(MODEL_ARTIFACTS['category_encoder'])
        with open(MODEL_ARTIFACTS['asin_to_category'], 'rb') as f:
            self.asin_to_category = pickle.load(f)
        
    def process_interactions(self):
        for message in self.consumer:
            interaction = message.value
            with self.lock:
                self.interaction_buffer.append(interaction)
                
                # Check if we should retrain
                if (len(self.interaction_buffer) >= BATCH_SIZE and 
                    datetime.now() - self.last_retrain_time >= RETRAIN_INTERVAL):
                    self.retrain_model()
    
    def prepare_training_data(self):
        """Convert interactions to training sequences"""
        conn = sqlite3.connect(DB_PATH)
        try:
            # Get all historical interactions in chronological order
            query = """
                SELECT book_id, timestamp 
                FROM user_interactions 
                ORDER BY timestamp ASC
            """
            history = pd.read_sql(query, conn)
            
            # Group by session (you might need a session ID in your data)
            # For simplicity, we'll treat all history as one sequence
            book_sequences = history['book_id'].tolist()
            
            # Encode books and categories
            encoded_books = self.book_encoder.transform(book_sequences)
            categories = [self.asin_to_category.get(asin, 0) for asin in book_sequences]
            encoded_categories = self.category_encoder.transform(categories)
            
            return encoded_books, encoded_categories
        finally:
            conn.close()
    
    def retrain_model(self):
        print("Starting model retraining...")
        self.last_retrain_time = datetime.now()
        
        # Prepare data
        book_sequences, category_sequences = self.prepare_training_data()
        
        # Initialize MLflow
        mlflow.set_tracking_uri("http://localhost:5001")
        mlflow.set_experiment("book-recommendations-retraining")
        
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("retrain_time", self.last_retrain_time.isoformat())
            mlflow.log_param("training_samples", len(book_sequences))
            
            # Initialize model
            num_books = len(self.book_encoder.classes_)
            num_categories = len(self.category_encoder.classes_)
            model = BERT4Rec(num_books, num_categories)
            
            # Train model (simplified - you'd use your actual training loop)
            optimizer = torch.optim.Adam(model.parameters())
            loss_fn = torch.nn.CrossEntropyLoss()
            
            # Convert to tensors and train
            # Note: You'd need to implement proper batching and masking
            input_tensor = torch.tensor(book_sequences[:-1], dtype=torch.long)
            target_tensor = torch.tensor(book_sequences[1:], dtype=torch.long)
            cat_tensor = torch.tensor(category_sequences[:-1], dtype=torch.long)
            
            output = model(input_tensor.unsqueeze(0), cat_tensor.unsqueeze(0))
            loss = loss_fn(output.squeeze(), target_tensor)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Log metrics
            mlflow.log_metric("loss", loss.item())
            
            # Save model
            mlflow.pytorch.log_model(model, "bert4rec_model")
            
            # Update the latest_run_id.txt
            run_id = mlflow.active_run().info.run_id
            with open('latest_run_id.txt', 'w') as f:
                f.write(run_id)
            
            print(f"Model retrained successfully. Run ID: {run_id}")
        
        # Clear buffer after successful retrain
        with self.lock:
            self.interaction_buffer = []

if __name__ == "__main__":
    retrainer = Retrainer()
    retrainer.process_interactions()