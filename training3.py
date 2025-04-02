import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import random
import numpy as np
import mlflow
import mlflow.pytorch
import joblib
import pickle
import logging
from kafka import KafkaProducer
import json
import os
import sys

# Setup logging - force handlers and clear any existing
logger = logging.getLogger(__name__)
logger.handlers = []  # Clear inherited handlers
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('training.log', mode='w')  # Overwrite each run
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# Device setup
device = torch.device('cpu')  # Forced CPU from previous fix
logger.info(f"Using device: {device}")

# Kafka producer setup
try:
    producer = KafkaProducer(
        bootstrap_servers=['localhost:9092'],
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        retries=3,
        retry_backoff_ms=500
    )
    logger.info("Kafka producer initialized successfully")
except Exception as e:
    logger.warning(f"Failed to initialize Kafka producer: {str(e)}. Will skip Kafka functionality.")
    producer = None

# KindleDataset class (unchanged)
class KindleDataset(Dataset):
    def __init__(self, customer_file, product_file, seq_len=20, mode='train', sample_fraction=0.2):
        logger.info("Initializing KindleDataset")
        try:
            self.customer_df = pd.read_excel(customer_file).sample(frac=sample_fraction, random_state=42)
            self.product_df = pd.read_excel(product_file)
            logger.info(f"Loaded customer_df: {self.customer_df.shape}, product_df: {self.product_df.shape}")
        except Exception as e:
            logger.error(f"Error loading data files: {str(e)}")
            raise

        self.book_encoder = LabelEncoder()
        all_books = list(self.customer_df['parent_asin'].unique()) + ['[PAD]', '[MASK]']
        self.book_encoder.fit(all_books)
        self.pad_id = self.book_encoder.transform(['[PAD]'])[0]
        self.mask_id = self.book_encoder.transform(['[MASK]'])[0]
        self.customer_df['parent_asin'] = self.book_encoder.transform(self.customer_df['parent_asin'])

        self.category_encoder = LabelEncoder()
        all_categories = set()
        for cat_str in self.product_df['categories']:
            cats = eval(cat_str) if isinstance(cat_str, str) and cat_str.startswith('[') else [cat_str]
            all_categories.update(cats)
        self.category_encoder.fit(list(all_categories))
        self.product_df['category_ids'] = self.product_df['categories'].apply(
            lambda x: self.category_encoder.transform(eval(x) if isinstance(x, str) and x.startswith('[') else [x])[0]
        )
        self.asin_to_category = dict(zip(self.product_df['parent_asin'], self.product_df['category_ids']))

        self.user_sequences = (
            self.customer_df.sort_values('timestamp')
            .groupby('user_id')['parent_asin']
            .apply(list)
            .to_dict()
        )
        self.user_ids = list(self.user_sequences.keys())

        if mode == 'eval':
            self.user_ids = [u for u in self.user_ids if len(self.user_sequences[u]) >= 2]

        self.seq_len = seq_len
        self.mode = mode
        self.num_books = len(self.book_encoder.classes_)
        self.num_categories = len(self.category_encoder.classes_)
        logger.info(f"Dataset initialized: {self.num_books} books, {self.num_categories} categories, mode={mode}")

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user = self.user_ids[idx]
        sequence = self.user_sequences[user]
        if self.mode == 'train':
            if len(sequence) < self.seq_len:
                seq = [self.pad_id] * (self.seq_len - len(sequence)) + sequence
            else:
                seq = sequence[-self.seq_len:]
            input_seq, labels = self.mask_random(seq)
        else:
            seq = sequence[:-1]
            if len(seq) < self.seq_len - 1:
                input_seq = [self.pad_id] * (self.seq_len - len(seq) - 1) + seq + [self.mask_id]
            else:
                input_seq = seq[-(self.seq_len - 1):] + [self.mask_id]
            labels = [0] * (self.seq_len - 1) + [sequence[-1]]

        category_ids = [self.asin_to_category.get(self.book_encoder.inverse_transform([x])[0], 0) for x in input_seq]
        return {
            'input': torch.tensor(input_seq, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'categories': torch.tensor(category_ids, dtype=torch.long)
        }

    def mask_random(self, seq):
        masked_seq = seq.copy()
        labels = [0] * len(seq)
        for i in range(len(seq)):
            if seq[i] == self.pad_id:
                continue
            if random.random() < 0.15:
                masked_seq[i] = self.mask_id
                labels[i] = seq[i]
        return masked_seq, labels

# BERT4Rec class (unchanged)
class BERT4Rec(nn.Module):
    def __init__(self, num_books, num_categories, hidden_size=32, num_layers=1, num_heads=2, max_seq_len=20, dropout=0.1):
        super(BERT4Rec, self).__init__()
        logger.info("Initializing BERT4Rec model")
        self.num_books = num_books
        self.book_embedding = nn.Embedding(num_books, hidden_size, padding_idx=0)
        self.category_embedding = nn.Embedding(num_categories, hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size * 2,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out = nn.Linear(hidden_size, num_books)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.register_buffer('pos_ids', torch.arange(max_seq_len))

    def forward(self, input_ids, category_ids):
        book_emb = self.book_embedding(input_ids)
        cat_emb = self.category_embedding(category_ids)
        pos_emb = self.position_embedding(self.pos_ids[:input_ids.size(1)])
        emb = book_emb + cat_emb + pos_emb
        mask = (input_ids != 0).float()
        emb = self.layer_norm(self.dropout(emb))
        output = self.transformer(emb, src_key_padding_mask=(mask == 0))
        logits = self.out(output)
        return logits

# Updated train_model
def train_model(train_dataset, num_epochs=2, use_mlflow=True):
    logger.info(f"Starting train_model with device {device}")
    model = BERT4Rec(
        num_books=train_dataset.num_books,
        num_categories=train_dataset.num_categories
    )
    model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    logger.info(f"DataLoader created with {len(train_loader)} batches")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    run_id = None
    if use_mlflow:
        try:
            mlflow_dir = os.path.abspath("./mlruns")
            os.makedirs(mlflow_dir, exist_ok=True)
            mlflow.set_tracking_uri("http://localhost:5001")
            logger.info("MLflow tracking URI set")
            
            with mlflow.start_run(run_name="BERT4Rec_Fast_Test") as run:
                run_id = run.info.run_id
                logger.info(f"MLflow run started with ID: {run_id}")
                mlflow.log_param("num_epochs", num_epochs)
                mlflow.log_param("hidden_size", 32)
                mlflow.log_param("num_layers", 1)
                mlflow.log_param("num_books", train_dataset.num_books)
                mlflow.log_param("num_categories", train_dataset.num_categories)

                for epoch in range(num_epochs):
                    model.train()
                    total_loss = 0
                    logger.info(f"Starting epoch {epoch+1}")
                    for i, batch in enumerate(train_loader):
                        input_ids = batch['input'].to(device)
                        labels = batch['labels'].to(device)
                        categories = batch['categories'].to(device)

                        optimizer.zero_grad()
                        logits = model(input_ids, categories)
                        logits = logits.view(-1, model.num_books)
                        labels = labels.view(-1)

                        loss = criterion(logits, labels)
                        loss.backward()
                        optimizer.step()

                        total_loss += loss.item()
                        if (i+1) % 10 == 0:
                            logger.info(f"Batch {i+1}/{len(train_loader)} loss = {loss.item():.4f}")

                    avg_loss = total_loss / len(train_loader)
                    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
                    mlflow.log_metric("train_loss", avg_loss, step=epoch)
                    logger.info(f"Epoch {epoch+1} completed, avg loss = {avg_loss}")

                torch.save(model.state_dict(), "bert4rec_model.pth")
                joblib.dump(train_dataset.book_encoder, "book_encoder.pkl")
                joblib.dump(train_dataset.category_encoder, "category_encoder.pkl")
                with open("asin_to_category.pkl", "wb") as f:
                    pickle.dump(train_dataset.asin_to_category, f)
                logger.info("Artifacts saved locally")

                mlflow.log_artifact("bert4rec_model.pth")
                mlflow.log_artifact("book_encoder.pkl")
                mlflow.log_artifact("category_encoder.pkl")
                mlflow.log_artifact("asin_to_category.pkl")
                mlflow.pytorch.log_model(model, "bert4rec_model")
                logger.info("Model and artifacts logged to MLflow")

                with open("latest_run_id.txt", "w") as f:
                    f.write(run_id)
                logger.info(f"Saved run ID {run_id} to latest_run_id.txt")
        except Exception as e:
            logger.error(f"MLflow error: {str(e)}. Continuing without MLflow.")
            run_id = None
    else:
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            logger.info(f"Starting epoch {epoch+1}")
            for i, batch in enumerate(train_loader):
                input_ids = batch['input'].to(device)
                labels = batch['labels'].to(device)
                categories = batch['categories'].to(device)

                optimizer.zero_grad()
                logits = model(input_ids, categories)
                logits = logits.view(-1, model.num_books)
                labels = labels.view(-1)

                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                if (i+1) % 10 == 0:
                    logger.info(f"Batch {i+1}/{len(train_loader)} loss = {loss.item():.4f}")

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
            logger.info(f"Epoch {epoch+1} completed, avg loss = {avg_loss}")
            
        torch.save(model.state_dict(), "bert4rec_model.pth")
        joblib.dump(train_dataset.book_encoder, "book_encoder.pkl")
        joblib.dump(train_dataset.category_encoder, "category_encoder.pkl")
        with open("asin_to_category.pkl", "wb") as f:
            pickle.dump(train_dataset.asin_to_category, f)
        logger.info("Artifacts saved locally")

    return model, run_id

# predict, evaluate_model, main functions (unchanged)
def predict(model, book_ids, train_dataset, top_k=5):
    model.eval()
    logger.info("Starting prediction")
    with torch.no_grad():
        max_seq_len = model.pos_ids.size(0)
        if len(book_ids) < max_seq_len - 1:
            input_seq = [train_dataset.pad_id] * (max_seq_len - len(book_ids) - 1) + book_ids + [train_dataset.mask_id]
        else:
            input_seq = book_ids[-(max_seq_len - 1):] + [train_dataset.mask_id]

        original_asins = [train_dataset.book_encoder.inverse_transform([x])[0] for x in input_seq]
        category_ids = [train_dataset.asin_to_category.get(asin, 0) for asin in original_asins]

        input_tensor = torch.tensor([input_seq], dtype=torch.long).to(device)
        category_tensor = torch.tensor([category_ids], dtype=torch.long).to(device)
        logits = model(input_tensor, category_tensor)
        last_logits = logits[0, -1, :]
        probs = torch.softmax(last_logits, dim=0)
        top_k_probs, top_k_ids = torch.topk(probs, top_k)

        preds = list(map(str, train_dataset.book_encoder.inverse_transform(top_k_ids.cpu().numpy())))
        logger.info(f"Prediction completed: {preds}")
        return preds

def evaluate_model(model, eval_loader, top_k=5, max_evals=10, run_id=None):
    model.eval()
    hits = 0
    total = 0
    ndcg_sum = 0
    logger.info("Starting evaluation")

    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            if i >= max_evals:
                break
                
            input_ids = batch['input'].to(device)
            labels = batch['labels'].to(device)
            categories = batch['categories'].to(device)

            logits = model(input_ids, categories)
            last_logits = logits[:, -1, :]
            last_labels = labels[:, -1]

            _, top_k_preds = torch.topk(last_logits, top_k, dim=1)

            for j in range(last_labels.size(0)):
                real_book = last_labels[j].item()
                guesses = top_k_preds[j].cpu().numpy()

                if real_book in guesses:
                    hits += 1
                    rank = list(guesses).index(real_book) + 1
                    ndcg = 1 / np.log2(rank + 1)
                    ndcg_sum += ndcg

                total += 1

    hit_rate = hits / total if total > 0 else 0
    ndcg = ndcg_sum / total if total > 0 else 0

    print(f"Hit Rate (Top {top_k}): {hit_rate:.4f}")
    print(f"NDCG (Top {top_k}): {ndcg:.4f}")
    
    if run_id:
        try:
            with mlflow.start_run(run_id=run_id):
                mlflow.log_metric("hit_rate", hit_rate)
                mlflow.log_metric("ndcg", ndcg)
        except Exception as e:
            logger.warning(f"Failed to log evaluation metrics to MLflow: {str(e)}")
    
    logger.info(f"Evaluation completed: Hit Rate={hit_rate}, NDCG={ndcg}")
    return hit_rate, ndcg

def main():
    logger.info("Starting main function")
    
    use_mlflow = "--no-mlflow" not in sys.argv
    test_kafka = "--test-kafka" in sys.argv
    
    train_dataset = KindleDataset('customer_clean.xlsx', 'product_clean.xlsx', seq_len=20, mode='train', sample_fraction=0.2)
    eval_dataset = KindleDataset('customer_clean.xlsx', 'product_clean.xlsx', seq_len=20, mode='eval', sample_fraction=0.2)
    logger.info("Datasets loaded")

    model, run_id = train_model(train_dataset, num_epochs=2, use_mlflow=use_mlflow)
    model.book_encoder = train_dataset.book_encoder

    eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False)
    logger.info("Starting evaluation")
    evaluate_model(model, eval_loader, max_evals=10, run_id=run_id)

    if (test_kafka or producer is not None) and len(eval_dataset.user_ids) > 0:
        sample_user = eval_dataset.user_ids[0]
        sample_books = eval_dataset.user_sequences[sample_user][:-1]
        guesses = predict(model, sample_books, train_dataset)
        real_next = eval_dataset.user_sequences[sample_user][-1]
        print(f"\nSample Prediction:")
        print(f"Input: {list(map(str, train_dataset.book_encoder.inverse_transform(sample_books)))}")
        print(f"Predicted: {guesses}")
        print(f"Actual: {train_dataset.book_encoder.inverse_transform([real_next])[0]}")

        if producer is not None:
            sample_interaction = {
                'book_id': train_dataset.book_encoder.inverse_transform([real_next])[0],
                'event_type': 'click',
                'timestamp': pd.Timestamp.now().isoformat()
            }
            try:
                producer.send('user_interactions', sample_interaction)
                producer.flush()
                logger.info(f"Sent sample interaction to Kafka: {sample_interaction}")
            except Exception as e:
                logger.error(f"Error sending to Kafka: {str(e)}")
        else:
            logger.warning("Kafka producer not available. Skipping Kafka test.")
    else:
        if len(eval_dataset.user_ids) == 0:
            logger.warning("No evaluation users available for prediction test")
        else:
            logger.info("Skipping Kafka test")

    logger.info("Script completed successfully")
    print("\nTraining and testing completed successfully!")
    if not use_mlflow:
        print("MLflow tracking was disabled")
    if producer is None:
        print("Kafka producer was not available")

if __name__ == "__main__":
    logger.info("Script started")
    main()
    logger.info("Script completed")