from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO
from flask_cors import CORS
import sqlite3
from threading import Lock
import torch
import torch.nn as nn
import joblib
import pickle
import pandas as pd
import logging
import random
from kafka import KafkaProducer
import json
import mlflow
import mlflow.pytorch
from datetime import datetime
import os
import glob

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('app.log', mode='w'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config['SECRET_KEY'] = 'secret!'

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='eventlet',
    ping_timeout=60,
    ping_interval=30,
    max_http_buffer_size=1_000_000,
    engineio_logger=False,
    logger=False
)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

db_lock = Lock()

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
    logger.error(f"Failed to initialize Kafka producer: {str(e)}")
    exit(1)

# Metrics tracking (in-memory)
metrics = {
    'total_recommendations': 0,
    'total_clicks': 0,
    'clicks_on_recommended': 0,
    'recommended_books': set(),
    'has_recommended': False  # Track if recommendations have been shown
}

# Set to track unique clicks
unique_clicks = set()

def init_db():
    with db_lock:
        try:
            conn = sqlite3.connect('books.db')
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS books
                        (id TEXT PRIMARY KEY,
                         title TEXT,
                         author TEXT,
                         img TEXT,
                         hovers INTEGER,
                         clicks INTEGER)''')
            c.execute('''CREATE TABLE IF NOT EXISTS user_interactions
                        (book_id TEXT,
                         timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                         UNIQUE(book_id))''')  # Add UNIQUE constraint to prevent duplicate book_id entries
            
            c.execute("SELECT COUNT(*) FROM books")
            if c.fetchone()[0] == 0:
                product_df = pd.read_excel('product_clean.xlsx')
                sample_books = []
                for _, row in product_df.iterrows():
                    book_id = str(row['parent_asin'])
                    title = str(row['title'])
                    author = str(row['author']) if pd.notna(row['author']) else 'Unknown Author'
                    img = 'https://via.placeholder.com/150'
                    sample_books.append((book_id, title, author, img, 0, 0))
                
                c.executemany('INSERT INTO books VALUES (?,?,?,?,?,?)', sample_books)
                conn.commit()
                logger.info(f"Initialized database with {len(sample_books)} books from product_clean.xlsx")
            conn.close()
        except Exception as e:
            logger.error(f"Database initialization failed: {str(e)}")
            raise

try:
    init_db()
    logger.info("Database initialized successfully!")
except Exception as e:
    logger.error(f"Failed to initialize database: {str(e)}")
    exit(1)

def get_db():
    with db_lock:
        conn = sqlite3.connect('books.db', check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

class BERT4Rec(nn.Module):
    def __init__(self, num_books, num_categories, hidden_size=64, num_layers=2, num_heads=2, max_seq_len=10, dropout=0.1):
        super(BERT4Rec, self).__init__()
        self.num_books = num_books
        self.book_embedding = nn.Embedding(num_books, hidden_size, padding_idx=0)
        self.category_embedding = nn.Embedding(num_categories, hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
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

def load_from_mlflow_artifacts():
    try:
        # Define paths
        mlartifacts_dir = "mlartifacts"
        experiment_id = "0"  # Hardcode to default experiment ID since we're not using mlruns

        # Check if mlartifacts directory exists
        if not os.path.exists(mlartifacts_dir):
            logger.error(f"MLflow artifacts directory not found at {mlartifacts_dir}")
            raise FileNotFoundError(f"MLflow artifacts directory not found at {mlartifacts_dir}")

        # Check if experiment directory exists in mlartifacts
        experiment_dir = os.path.join(mlartifacts_dir, experiment_id)
        if not os.path.exists(experiment_dir):
            logger.error(f"Experiment directory not found at {experiment_dir}")
            raise FileNotFoundError(f"Experiment directory not found at {experiment_dir}")

        # Find all run directories in mlartifacts/0/ (only directories)
        run_dirs = [d for d in glob.glob(os.path.join(experiment_dir, "*")) if os.path.isdir(d)]
        if not run_dirs:
            logger.error(f"No runs found in mlartifacts for experiment {experiment_id}")
            raise Exception(f"No runs found in mlartifacts for experiment {experiment_id}")

        # Sort runs by modification time to get the latest
        run_dirs.sort(key=os.path.getmtime, reverse=True)
        latest_run_dir = run_dirs[0]
        run_id = os.path.basename(latest_run_dir)
        logger.info(f"Latest run ID from mlartifacts: {run_id}")

        # Construct the artifact path in mlartifacts
        artifact_base_path = os.path.join(mlartifacts_dir, experiment_id, run_id, "artifacts")
        if not os.path.exists(artifact_base_path):
            logger.error(f"Artifacts directory not found at {artifact_base_path}")
            raise FileNotFoundError(f"Artifacts directory not found at {artifact_base_path}")

        # Load the model weights directly from bert4rec_model.pth
        model_pth_path = os.path.join(artifact_base_path, "bert4rec_model.pth")
        if not os.path.exists(model_pth_path):
            logger.error(f"Model .pth file not found at {model_pth_path}")
            raise FileNotFoundError(f"Model .pth file not found at {model_pth_path}")

        # Load the encoders first to get num_books and num_categories
        book_encoder_path = os.path.join(artifact_base_path, "book_encoder.pkl")
        category_encoder_path = os.path.join(artifact_base_path, "category_encoder.pkl")
        asin_to_category_path = os.path.join(artifact_base_path, "asin_to_category.pkl")

        if not os.path.exists(book_encoder_path):
            logger.error(f"book_encoder.pkl not found at {book_encoder_path}")
            raise FileNotFoundError(f"book_encoder.pkl not found at {book_encoder_path}")
        if not os.path.exists(category_encoder_path):
            logger.error(f"category_encoder.pkl not found at {category_encoder_path}")
            raise FileNotFoundError(f"category_encoder.pkl not found at {category_encoder_path}")
        if not os.path.exists(asin_to_category_path):
            logger.error(f"asin_to_category.pkl not found at {asin_to_category_path}")
            raise FileNotFoundError(f"asin_to_category.pkl not found at {asin_to_category_path}")

        book_encoder = joblib.load(book_encoder_path)
        logger.info(f"Loaded book_encoder from {book_encoder_path}")

        category_encoder = joblib.load(category_encoder_path)
        logger.info(f"Loaded category_encoder from {category_encoder_path}")

        num_books = len(book_encoder.classes_)
        num_categories = len(category_encoder.classes_)

        # Instantiate the model
        model = BERT4Rec(num_books, num_categories)
        model.load_state_dict(torch.load(model_pth_path, map_location=device))
        model.to(device)
        model.eval()
        logger.info(f"Loaded model weights from {model_pth_path}")

        # Load asin_to_category
        with open(asin_to_category_path, 'rb') as f:
            asin_to_category = pickle.load(f)
        logger.info(f"Loaded asin_to_category from {asin_to_category_path}")

        return model, book_encoder, category_encoder, asin_to_category

    except Exception as e:
        logger.error(f"Failed to load from mlartifacts: {str(e)}")
        raise

# Load model and artifacts from mlartifacts
try:
    model, book_encoder, category_encoder, asin_to_category = load_from_mlflow_artifacts()
    num_books = len(book_encoder.classes_)
    num_categories = len(category_encoder.classes_)
    pad_id = book_encoder.transform(['[PAD]'])[0]
    mask_id = book_encoder.transform(['[MASK]'])[0]
    logger.info("Model and components loaded successfully from mlartifacts!")
except Exception as e:
    logger.error(f"Loading from mlartifacts failed: {str(e)}")
    exit(1)

def predict(encoded_book_ids, top_k=5):
    with torch.no_grad():
        max_seq_len = model.pos_ids.size(0)
        encoded_book_ids = encoded_book_ids.tolist()
        if len(encoded_book_ids) < max_seq_len - 1:
            input_seq = [pad_id] * (max_seq_len - len(encoded_book_ids) - 1) + encoded_book_ids + [mask_id]
        else:
            input_seq = encoded_book_ids[-(max_seq_len - 1):] + [mask_id]
        original_asins = [book_encoder.inverse_transform([x])[0] for x in input_seq]
        category_ids = [asin_to_category.get(asin, 0) for asin in original_asins]
        input_tensor = torch.tensor([input_seq], dtype=torch.long).to(device)
        category_tensor = torch.tensor([category_ids], dtype=torch.long).to(device)
        logits = model(input_tensor, category_tensor)
        last_logits = logits[0, -1, :]
        probs = torch.softmax(last_logits, dim=0)
        top_k_probs, top_k_ids = torch.topk(probs, top_k)
        top_k_asins = book_encoder.inverse_transform(top_k_ids.cpu().numpy())
        return list(map(str, top_k_asins)), top_k_probs.cpu().numpy()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    dates = []
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT DATE(timestamp) FROM user_interactions ORDER BY timestamp DESC")
        dates = [row[0] for row in cursor.fetchall()]
        conn.close()
    except Exception as e:
        logger.error(f"Error fetching dates for dashboard: {str(e)}")
    return render_template('dashboard.html', dates=dates)

@app.route('/api/books')
def get_books():
    try:
        conn = get_db()
        books = conn.execute('SELECT * FROM books').fetchall()
        logger.debug(f"Returning {len(books)} books")
        return jsonify([dict(book) for book in books])
    except Exception as e:
        logger.error(f"Database error in get_books: {str(e)}")
        return jsonify({"error": "Failed to load books"}), 500
    finally:
        conn.close()

@app.route('/api/has_history')
def has_history():
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM user_interactions")
        count = cursor.fetchone()[0]
        logger.debug(f"History count: {count}")
        return jsonify({'hasHistory': count > 0})
    except Exception as e:
        logger.error(f"Database error in has_history: {str(e)}")
        return jsonify({"error": "Failed to check history"}), 500
    finally:
        conn.close()

@app.route('/api/click_history')
def get_click_history():
    try:
        conn = get_db()
        query = '''
            SELECT ui.book_id, b.title, ui.timestamp
            FROM user_interactions ui
            JOIN books b ON ui.book_id = b.id
            ORDER BY ui.timestamp DESC
            LIMIT 50
        '''
        clicks = conn.execute(query).fetchall()
        click_history = [
            {
                'book_id': click['book_id'],
                'title': click['title'],
                'timestamp': click['timestamp']
            } for click in clicks
        ]
        logger.debug(f"Returning click history with {len(click_history)} entries")
        return jsonify({
            'clicks': click_history,
            'total_clicks': len(click_history)
        })
    except Exception as e:
        logger.error(f"Database error in get_click_history: {str(e)}")
        return jsonify({"error": "Failed to load click history"}), 500
    finally:
        conn.close()

@app.route('/api/book_click_history/<book_id>')
def get_book_click_history(book_id):
    try:
        conn = get_db()
        query = '''
            SELECT ui.book_id, b.title, ui.timestamp
            FROM user_interactions ui
            JOIN books b ON ui.book_id = b.id
            WHERE ui.book_id = ?
            ORDER BY ui.timestamp DESC
            LIMIT 50
        '''
        clicks = conn.execute(query, (book_id,)).fetchall()
        click_history = [
            {
                'book_id': click['book_id'],
                'title': click['title'],
                'timestamp': click['timestamp']
            } for click in clicks
        ]
        logger.debug(f"Returning click history for book {book_id} with {len(click_history)} entries")
        return jsonify({
            'clicks': click_history,
            'total_clicks': len(click_history)
        })
    except Exception as e:
        logger.error(f"Database error in get_book_click_history: {str(e)}")
        return jsonify({"error": "Failed to load book click history"}), 500
    finally:
        conn.close()

@app.route('/api/historical_metrics')
def get_historical_metrics():
    date = request.args.get('date')
    try:
        conn = get_db()
        query = '''
            SELECT ui.book_id, ui.timestamp
            FROM user_interactions ui
            WHERE DATE(ui.timestamp) = ?
        '''
        clicks = conn.execute(query, (date,)).fetchall()
        total_clicks = len(clicks)
        clicks_on_recommended = sum(1 for click in clicks if click['book_id'] in metrics['recommended_books'])
        total_recommendations = metrics['total_recommendations'] if metrics['total_recommendations'] > 0 else 1  # Avoid division by zero
        ctr = clicks_on_recommended / total_recommendations
        return jsonify({
            'total_recommendations': total_recommendations,
            'total_clicks': total_clicks,
            'clicks_on_recommended': clicks_on_recommended,
            'ctr': ctr
        })
    except Exception as e:
        logger.error(f"Error fetching historical metrics: {str(e)}")
        return jsonify({"error": "Failed to load historical metrics"}), 500
    finally:
        conn.close()

@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')

@socketio.on('update_interaction')
def handle_interaction(data):
    global metrics, unique_clicks
    book_id = data.get('bookId')
    event_type = data.get('eventType')
    if not book_id or not event_type:
        logger.warning("Invalid interaction data")
        return
    
    # Update books.db
    try:
        conn = get_db()
        cursor = conn.cursor()
        if event_type == 'click':
            # Check if this book has already been clicked
            if book_id not in unique_clicks:
                # Increment the clicks in the books table
                cursor.execute('UPDATE books SET clicks = clicks + 1 WHERE id = ?', (book_id,))
                # Insert into user_interactions (UNIQUE constraint will prevent duplicates)
                try:
                    cursor.execute('INSERT INTO user_interactions (book_id) VALUES (?)', (book_id,))
                    unique_clicks.add(book_id)
                    metrics['total_clicks'] += 1
                    if book_id in metrics['recommended_books']:
                        metrics['clicks_on_recommended'] += 1
                except sqlite3.IntegrityError:
                    # If the book_id already exists in user_interactions, skip incrementing
                    logger.debug(f"Book {book_id} already clicked, skipping duplicate.")
            else:
                logger.debug(f"Book {book_id} already clicked, skipping duplicate.")
        elif event_type == 'hover':
            cursor.execute('UPDATE books SET hovers = hovers + 1 WHERE id = ?', (book_id,))
        conn.commit()
        updated_book = conn.execute('SELECT * FROM books WHERE id = ?', (book_id,)).fetchone()
        socketio.emit('update_counts', dict(updated_book))
        logger.info(f"Updated book {book_id} with {event_type}")
    except Exception as e:
        logger.error(f"Interaction error: {str(e)}")
    finally:
        conn.close()

    # Send interaction to Kafka
    interaction = {
        'book_id': book_id,
        'event_type': event_type,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    try:
        producer.send('user_interactions', interaction)
        producer.flush()
        logger.info(f"Sent to Kafka (interactions): {interaction}")
    except Exception as e:
        logger.error(f"Kafka send error (interactions): {str(e)}")

    # Send metrics to Kafka
    send_metrics_update()

def send_metrics_update():
    global metrics
    metrics_data = {
        'total_recommendations': metrics['total_recommendations'],
        'total_clicks': metrics['total_clicks'],
        'clicks_on_recommended': metrics['clicks_on_recommended'],
        'ctr': metrics['clicks_on_recommended'] / metrics['total_recommendations'] if metrics['total_recommendations'] > 0 else 0,
        'timestamp': datetime.now().isoformat()
    }
    try:
        producer.send('metrics_topic', metrics_data)
        producer.flush()
        logger.info(f"Sent metrics to Kafka: {metrics_data}")
    except Exception as e:
        logger.error(f"Kafka send error (metrics): {str(e)}")

@socketio.on('get_recommendations')
def handle_get_recommendations(data):
    global metrics
    logger.debug(f"Received get_recommendations with data: {data}")
    session_clicks = data.get('sessionClicks', [])
    
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT book_id FROM user_interactions ORDER BY timestamp ASC')
    historical_clicks = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    def filter_valid_books(book_ids):
        return [bid for bid in book_ids if bid in book_encoder.classes_]
    
    valid_session_clicks = filter_valid_books(session_clicks)
    valid_historical_clicks = filter_valid_books(historical_clicks)
    
    recommended_books_1 = []
    probs_1 = []
    if valid_session_clicks:
        encoded_session = book_encoder.transform(valid_session_clicks)
        books_1, probs_1 = predict(encoded_session, top_k=20)
        recommended_books_1 = books_1
    
    combined_clicks = valid_historical_clicks + valid_session_clicks
    recommended_books_2 = []
    probs_2 = []
    if combined_clicks:
        encoded_combined = book_encoder.transform(combined_clicks)
        books_2, probs_2 = predict(encoded_combined, top_k=20)
        recommended_books_2 = books_2
    
    all_books = recommended_books_1 + recommended_books_2
    all_probs = list(probs_1) + list(probs_2)
    unique_books = []
    unique_probs = []
    seen = set()
    
    for book, prob in zip(all_books, all_probs):
        if book not in seen:
            seen.add(book)
            unique_books.append(book)
            unique_probs.append(prob)
    
    if unique_books:
        sorted_pairs = sorted(zip(unique_probs, unique_books), reverse=True)
        top_books = [book for _, book in sorted_pairs][:20]
        # Set initial count to 20 for the first recommendation, then increment
        if not metrics['has_recommended']:
            metrics['total_recommendations'] = 20
            metrics['has_recommended'] = True
        else:
            metrics['total_recommendations'] += len(top_books)
        metrics['recommended_books'].update(top_books)
        if len(top_books) < 20:
            conn = get_db()
            cursor = conn.cursor()
            placeholder = ','.join(['?'] * len(top_books))
            cursor.execute(f'SELECT id FROM books WHERE id NOT IN ({placeholder})', tuple(top_books))
            remaining_books = [row['id'] for row in cursor.fetchall()]
            conn.close()
            if remaining_books:
                additional_books = random.sample(remaining_books, min(20 - len(top_books), len(remaining_books)))
                top_books.extend(additional_books)
                if not metrics['has_recommended']:
                    metrics['total_recommendations'] = 20
                    metrics['has_recommended'] = True
                else:
                    metrics['total_recommendations'] += len(additional_books)
                metrics['recommended_books'].update(additional_books)
    else:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('SELECT id FROM books')
        all_book_ids = [row['id'] for row in cursor.fetchall()]
        conn.close()
        top_books = random.sample(all_book_ids, min(20, len(all_book_ids)))
        # Set initial count to 20 for the first recommendation, then increment
        if not metrics['has_recommended']:
            metrics['total_recommendations'] = 20
            metrics['has_recommended'] = True
        else:
            metrics['total_recommendations'] += len(top_books)
        metrics['recommended_books'].update(top_books)
    
    logger.debug(f"Emitting recommendations: {top_books}")
    socketio.emit('recommendations', {'books': top_books}, to=request.sid)
    send_metrics_update()

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5003, debug=True, use_reloader=False)