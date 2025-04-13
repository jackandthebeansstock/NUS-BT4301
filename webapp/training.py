import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score
import random
import numpy as np
import mlflow
import mlflow.pytorch
import joblib
import pickle
import logging
import os
import sys
import tempfile
import shutil

# Setup logging
logger = logging.getLogger(__name__)
logger.handlers = []  # Clear inherited handlers
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('training.log', mode='w')
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# Device setup
device = torch.device('cpu')  # Forced CPU as per your setup
logger.info(f"Using device: {device}")

# KindleDataset class
class KindleDataset(Dataset):
    def __init__(self, customer_file, product_file, seq_len=10, mode='train', sample_fraction=0.2):
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

# BERT4Rec class
class BERT4Rec(nn.Module):
    def __init__(self, num_books, num_categories, hidden_size=64, num_layers=2, num_heads=2, max_seq_len=10, dropout=0.1):
        super(BERT4Rec, self).__init__()
        logger.info("Initializing BERT4Rec model")
        self.num_books = num_books
        self.book_embedding = nn.Embedding(num_books, hidden_size, padding_idx=0)
        self.category_embedding = nn.Embedding(num_categories, hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size * 4,
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

# Train model function
def train_model(train_dataset, num_epochs=2, use_mlflow=True):
    logger.info(f"Starting training with {num_epochs} epochs")
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
    # Create a temporary directory for saving artifacts
    temp_dir = tempfile.mkdtemp()
    try:
        # Define file paths in the temporary directory
        model_path = os.path.join(temp_dir, "bert4rec_model.pth")
        book_encoder_path = os.path.join(temp_dir, "book_encoder.pkl")
        category_encoder_path = os.path.join(temp_dir, "category_encoder.pkl")
        asin_to_category_path = os.path.join(temp_dir, "asin_to_category.pkl")

        if use_mlflow:
            try:
                mlflow.set_tracking_uri("http://localhost:5001")
                logger.info("MLflow tracking URI set")
                
                with mlflow.start_run(run_name="BERT4Rec_Training") as run:
                    run_id = run.info.run_id
                    logger.info(f"MLflow run started with ID: {run_id}")
                    mlflow.log_param("num_epochs", num_epochs)
                    mlflow.log_param("hidden_size", 64)
                    mlflow.log_param("num_layers", 2)
                    mlflow.log_param("num_books", train_dataset.num_books)
                    mlflow.log_param("num_categories", train_dataset.num_categories)

                    for epoch in range(num_epochs):
                        model.train()
                        total_loss = 0
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

                        avg_loss = total_loss / len(train_loader)
                        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
                        mlflow.log_metric("train_loss", avg_loss, step=epoch)
                        logger.info(f"Epoch {epoch+1} completed, avg loss = {avg_loss:.4f}")

                    # Save artifacts to the temporary directory
                    torch.save(model.state_dict(), model_path)
                    joblib.dump(train_dataset.book_encoder, book_encoder_path)
                    joblib.dump(train_dataset.category_encoder, category_encoder_path)
                    with open(asin_to_category_path, "wb") as f:
                        pickle.dump(train_dataset.asin_to_category, f)

                    # Log artifacts to MLflow
                    mlflow.log_artifact(model_path)
                    mlflow.log_artifact(book_encoder_path)
                    mlflow.log_artifact(category_encoder_path)
                    mlflow.log_artifact(asin_to_category_path)
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

                avg_loss = total_loss / len(train_loader)
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
                logger.info(f"Epoch {epoch+1} completed, avg loss = {avg_loss:.4f}")

            # Save artifacts directly to the working directory when MLflow is disabled
            torch.save(model.state_dict(), "bert4rec_model.pth")
            joblib.dump(train_dataset.book_encoder, "book_encoder.pkl")
            joblib.dump(train_dataset.category_encoder, "category_encoder.pkl")
            with open("asin_to_category.pkl", "wb") as f:
                pickle.dump(train_dataset.asin_to_category, f)
            logger.info("Artifacts saved locally")
    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)

    return model, run_id

# Evaluate model function
def evaluate_model(model, eval_loader, top_k=5, max_evals=10, run_id=None):
    model.eval()
    hits = 0
    total = 0
    ndcg_sum = 0
    mrr_sum = 0
    all_preds = []
    all_labels = []
    all_probs = []
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

            probs = torch.softmax(last_logits, dim=1)
            top_k_probs, top_k_preds = torch.topk(probs, top_k, dim=1)
            top_k_preds = top_k_preds.cpu().numpy()
            top_k_probs = top_k_probs.cpu().numpy()
            last_labels_np = last_labels.cpu().numpy()

            for j in range(last_labels.size(0)):
                real_book = last_labels_np[j]
                guesses = top_k_preds[j]
                guess_probs = top_k_probs[j]

                if real_book in guesses:
                    hits += 1
                    rank = list(guesses).index(real_book) + 1
                    ndcg = 1 / np.log2(rank + 1)
                    ndcg_sum += ndcg
                    mrr_sum += 1 / rank

                total += 1

                pred = guesses[0]
                all_preds.append(pred)
                all_labels.append(real_book)
                all_probs.append(guess_probs.max())

    hit_rate = hits / total if total > 0 else 0
    ndcg = ndcg_sum / total if total > 0 else 0
    mrr = mrr_sum / total if total > 0 else 0
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='micro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    binary_labels = [1 if pred == label else 0 for pred, label in zip(all_preds, all_labels)]
    auc_roc = roc_auc_score(binary_labels, all_probs) if len(set(binary_labels)) > 1 else 0

    metrics = {
        "Hit Rate": hit_rate,
        "NDCG": ndcg,
        "MRR": mrr,
        "Accuracy": accuracy,
        "Precision": precision,
        "F1 Score": f1,
        "AUC-ROC": auc_roc
    }
    
    print(f"\nEvaluation Metrics (Top {top_k}):")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    if run_id:
        try:
            with mlflow.start_run(run_id=run_id):
                for metric, value in metrics.items():
                    mlflow.log_metric(metric.lower().replace(" ", "_"), value)
        except Exception as e:
            logger.warning(f"Failed to log evaluation metrics to MLflow: {str(e)}")
    
    logger.info(f"Evaluation completed: {', '.join([f'{k}={v:.4f}' for k, v in metrics.items()])}")
    return metrics

# Main function
def main():
    logger.info("Starting training script")
    
    use_mlflow = "--no-mlflow" not in sys.argv
    
    train_dataset = KindleDataset('customer_clean.xlsx', 'product_clean.xlsx', seq_len=10, mode='train', sample_fraction=0.2)
    eval_dataset = KindleDataset('customer_clean.xlsx', 'product_clean.xlsx', seq_len=10, mode='eval', sample_fraction=0.2)
    logger.info("Datasets loaded")

    model, run_id = train_model(train_dataset, num_epochs=2, use_mlflow=use_mlflow)

    eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False)
    logger.info("Starting evaluation")
    metrics = evaluate_model(model, eval_loader, max_evals=10, run_id=run_id)

    logger.info("Training and evaluation completed successfully")
    print("\nTraining and evaluation completed successfully!")
    if not use_mlflow:
        print("MLflow tracking was disabled")

if __name__ == "__main__":
    logger.info("Script started")
    main()
    logger.info("Script completed")