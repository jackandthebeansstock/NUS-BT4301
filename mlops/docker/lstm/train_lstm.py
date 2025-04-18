import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import mlflow
import os
import pickle
import logging
import joblib

# Setup logging
logger = logging.getLogger(__name__)
logger.handlers = []
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('/app/models/training_lstm.log', mode='w')
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# Device setup
device = torch.device('cpu')
logger.info(f"Using device: {device}")

# Dataset class
class SequenceDataset(Dataset):
    def __init__(self, customer_file, product_file, seq_len=10, mode='train', sample_fraction=0.2):
        logger.info("Initializing SequenceDataset")
        self.customer_df = pd.read_excel(customer_file).sample(frac=sample_fraction, random_state=42)
        logger.info(f"Loaded customer_df: {self.customer_df.shape}")

        self.item_encoder = LabelEncoder()
        all_items = list(self.customer_df['parent_asin'].unique()) + ['[PAD]']
        self.item_encoder.fit(all_items)
        self.pad_id = self.item_encoder.transform(['[PAD]'])[0]
        self.customer_df['parent_asin'] = self.item_encoder.transform(self.customer_df['parent_asin'])

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
        self.num_items = len(self.item_encoder.classes_)
        logger.info(f"Dataset initialized: {self.num_items} items, mode={mode}")

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
            input_seq = seq[:-1]
            label = seq[1:]
        else:
            seq = sequence[:-1]
            if len(seq) < self.seq_len - 1:
                input_seq = [self.pad_id] * (self.seq_len - len(seq) - 1) + seq
            else:
                input_seq = seq[-(self.seq_len - 1):]
            label = [sequence[-1]] * (self.seq_len - 1)

        return {
            'input': torch.tensor(input_seq, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# LSTM model
class LSTMPredictor(nn.Module):
    def __init__(self, num_items, embedding_dim=64, hidden_dim=128):
        super(LSTMPredictor, self).__init__()
        self.embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_items)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        out = self.dropout(lstm_out)
        logits = self.fc(out)
        return logits

# Train model
def train_model(dataset, num_epochs=2):
    logger.info(f"Starting training with {num_epochs} epochs")
    model = LSTMPredictor(num_items=dataset.num_items)
    model.to(device)

    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    model_path = "/app/models/lstm_model.pth"
    item_encoder_path = "/app/models/lstm_item_encoder.pkl"

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    with mlflow.start_run(run_name="LSTM_Training"):
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("embedding_dim", 64)
        mlflow.log_param("hidden_dim", 128)
        mlflow.log_param("num_items", dataset.num_items)

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for batch in loader:
                input_ids = batch['input'].to(device)
                labels = batch['labels'].to(device)

                optimizer.zero_grad()
                logits = model(input_ids)
                logits = logits.view(-1, dataset.num_items)
                labels = labels.view(-1)

                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            logger.info(f"Epoch {epoch+1} completed, avg loss = {avg_loss:.4f}")

        # Save artifacts
        torch.save(model.state_dict(), model_path)
        joblib.dump(dataset.item_encoder, item_encoder_path)

        mlflow.log_artifact(model_path)
        mlflow.log_artifact(item_encoder_path)
        mlflow.pytorch.log_model(model, "lstm_model")
        logger.info("Model and artifacts logged to MLflow")

    return model

# Main function
def main():
    logger.info("Starting LSTM training script")
    
    customer_file = "/app/data/customer_clean.xlsx"
    product_file = "/app/data/product_clean.xlsx"
    train_dataset = SequenceDataset(customer_file, product_file, seq_len=10, mode='train', sample_fraction=0.2)
    logger.info("Dataset loaded")

    model = train_model(train_dataset, num_epochs=2)

    logger.info("Training completed successfully")
    print("Training completed successfully!")

if __name__ == "__main__":
    logger.info("Script started")
    main()
    logger.info("Script completed")