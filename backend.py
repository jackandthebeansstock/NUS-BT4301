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
import os

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
                         timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
            
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

# Load model from MLflow using the server on port 5001
def load_model_from_mlflow():
    try:
        # Set MLflow tracking URI to the server
        mlflow.set_tracking_uri("http://localhost:5001")
        logger.info("MLflow tracking URI set to http://localhost:5001")

        # Read the run ID from latest_run_id.txt
        with open('latest_run_id.txt', 'r') as f:
            run_id = f.read().strip()
        logger.info(f"Read run ID from latest_run_id.txt: {run_id}")

        # Load the model using the run ID
        model_uri = f"runs:/{run_id}/bert4rec_model"
        model = mlflow.pytorch.load_model(model_uri)
        model.to(device)
        model.eval()
        logger.info(f"Loaded model from MLflow run: {model_uri}")
        return model
    except FileNotFoundError as e:
        logger.error(f"latest_run_id.txt not found: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Failed to load model from MLflow: {str(e)}")
        # Fallback to local file if MLflow fails
        try:
            model = BERT4Rec(num_books, num_categories)
            model.load_state_dict(torch.load('bert4rec_model.pth', map_location=device))
            model.to(device)
            model.eval()
            logger.info("Loaded model from local file bert4rec_model.pth as fallback")
            return model
        except Exception as fallback_e:
            logger.error(f"Fallback loading failed: {str(fallback_e)}")
            raise

# Load model and artifacts
try:
    book_encoder = joblib.load('book_encoder.pkl')
    category_encoder = joblib.load('category_encoder.pkl')
    with open('asin_to_category.pkl', 'rb') as f:
        asin_to_category = pickle.load(f)
    num_books = len(book_encoder.classes_)
    num_categories = len(category_encoder.classes_)
    model = load_model_from_mlflow()  # Load from MLflow server
    pad_id = book_encoder.transform(['[PAD]'])[0]
    mask_id = book_encoder.transform(['[MASK]'])[0]
    logger.info("Model and components loaded successfully!")
except Exception as e:
    logger.error(f"Model loading failed: {str(e)}")
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

@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')

@socketio.on('update_interaction')
def handle_interaction(data):
    book_id = data.get('bookId')
    event_type = data.get('eventType')
    if not book_id or not event_type:
        logger.warning("Invalid interaction data")
        return
    
    # Send interaction to Kafka
    interaction = {
        'book_id': book_id,
        'event_type': event_type,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    try:
        producer.send('user_interactions', interaction)
        producer.flush()
        logger.info(f"Sent to Kafka: {interaction}")
    except Exception as e:
        logger.error(f"Kafka send error: {str(e)}")
        return
    
    # Update SQLite for real-time UI
    try:
        conn = get_db()
        cursor = conn.cursor()
        if event_type == 'click':
            cursor.execute('UPDATE books SET clicks = clicks + 1 WHERE id = ?', (book_id,))
            cursor.execute('INSERT INTO user_interactions (book_id) VALUES (?)', (book_id,))
        elif event_type == 'hover':
            cursor.execute('UPDATE books SET hovers = hovers + 1 WHERE id = ?', (book_id,))
        conn.commit()
        updated_book = conn.execute('SELECT * FROM books WHERE id = ?', (book_id,)).fetchone()
        socketio.emit('update_counts', dict(updated_book))
    except Exception as e:
        logger.error(f"Interaction error: {str(e)}")
    finally:
        conn.close()

@socketio.on('get_recommendations')
def handle_get_recommendations(data):
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
        books_1, probs_1 = predict(encoded_session, top_k=5)
        recommended_books_1 = books_1
        logger.debug(f"Session clicks recommendations: {recommended_books_1}")
    
    combined_clicks = valid_historical_clicks + valid_session_clicks
    recommended_books_2 = []
    probs_2 = []
    if combined_clicks:
        encoded_combined = book_encoder.transform(combined_clicks)
        books_2, probs_2 = predict(encoded_combined, top_k=5)
        recommended_books_2 = books_2
        logger.debug(f"Combined clicks recommendations: {recommended_books_2}")
    
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
        top_books = [book for _, book in sorted_pairs][:10]
    else:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('SELECT id FROM books')
        all_book_ids = [row['id'] for row in cursor.fetchall()]
        conn.close()
        top_books = random.sample(all_book_ids, min(10, len(all_book_ids)))
    
    logger.debug(f"Emitting recommendations: {top_books}")
    socketio.emit('recommendations', {'books': top_books}, to=request.sid)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5003, debug=True, use_reloader=False)