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
from collections import defaultdict

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
    async_mode='threading',
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
    'has_recommended': False,
    'unique_users': 0,
    'likes': 0,
    'dislikes': 0,
    'diversity': 0.0,
    'coverage': 0.0
}
unique_users = set()
user_recommendations = defaultdict(set)  # Track recommendations per user

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
                         clicks INTEGER,
                         genre TEXT)''')
            c.execute('''CREATE TABLE IF NOT EXISTS user_interactions
                        (book_id TEXT,
                         user_id TEXT,
                         timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                         PRIMARY KEY (book_id, user_id))''')
            c.execute('''CREATE TABLE IF NOT EXISTS user_feedback
                        (book_id TEXT,
                         user_id TEXT,
                         feedback_type TEXT,
                         timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                         PRIMARY KEY (book_id, user_id))''')
            conn.commit()
            logger.info("Created books, user_interactions, and user_feedback tables")
            
            c.execute("SELECT COUNT(*) FROM books")
            if c.fetchone()[0] == 0:
                product_df = pd.read_excel('product_clean.xlsx')
                sample_books = []
                for _, row in product_df.iterrows():
                    book_id = str(row['parent_asin'])
                    title = str(row['title'])
                    author = str(row['author']) if pd.notna(row['author']) else 'Unknown Author'
                    img = 'https://via.placeholder.com/150'
                    genre = None
                    categories = row['categories']
                    if categories and categories != 'nan':
                        try:
                            category_list = [cat.strip() for cat in categories.split("'") if cat.strip()]
                            logger.debug(f"Split categories: {category_list}")
                            if category_list:
                                genre = category_list[-2]
                        except (ValueError, SyntaxError):
                            logger.warning(f"Invalid categories format for book {book_id}: {categories}")
                    sample_books.append((book_id, title, author, img, 0, 0, genre))
                
                try:
                    c.executemany('INSERT INTO books VALUES (?,?,?, ?, ?, ?, ?)', sample_books)
                    conn.commit()
                    logger.info(f"Initialized database with {len(sample_books)} books from product_clean.xlsx")
                except Exception as e:
                    logger.error(f"Error inserting books: {str(e)}")
                    raise    
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
        mlartifacts_dir = "mlartifacts"
        experiment_id = "0"
        if not os.path.exists(mlartifacts_dir):
            logger.error(f"MLflow artifacts directory not found at {mlartifacts_dir}")
            raise FileNotFoundError(f"MLflow artifacts directory not found at {mlartifacts_dir}")
        experiment_dir = os.path.join(mlartifacts_dir, experiment_id)
        if not os.path.exists(experiment_dir):
            logger.error(f"Experiment directory not found at {experiment_dir}")
            raise FileNotFoundError(f"Experiment directory not found at {experiment_dir}")
        run_dirs = [d for d in glob.glob(os.path.join(experiment_dir, "*")) if os.path.isdir(d)]
        if not run_dirs:
            logger.error(f"No runs found in mlartifacts for experiment {experiment_id}")
            raise Exception(f"No runs found in mlartifacts for experiment {experiment_id}")
        run_dirs.sort(key=os.path.getmtime, reverse=True)
        latest_run_dir = run_dirs[0]
        run_id = os.path.basename(latest_run_dir)
        logger.info(f"Latest run ID from mlartifacts: {run_id}")
        artifact_base_path = os.path.join(mlartifacts_dir, experiment_id, run_id, "artifacts")
        if not os.path.exists(artifact_base_path):
            logger.error(f"Artifacts directory not found at {artifact_base_path}")
            raise FileNotFoundError(f"Artifacts directory not found at {artifact_base_path}")
        model_pth_path = os.path.join(artifact_base_path, "bert4rec_model.pth")
        if not os.path.exists(model_pth_path):
            logger.error(f"Model .pth file not found at {model_pth_path}")
            raise FileNotFoundError(f"Model .pth file not found at {model_pth_path}")
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
        model = BERT4Rec(num_books, num_categories)
        model.load_state_dict(torch.load(model_pth_path, map_location=device))
        model.to(device)
        model.eval()
        logger.info(f"Loaded model weights from {model_pth_path}")
        with open(asin_to_category_path, 'rb') as f:
            asin_to_category = pickle.load(f)
        logger.info(f"Loaded asin_to_category from {asin_to_category_path}")
        return model, book_encoder, category_encoder, asin_to_category
    except Exception as e:
        logger.error(f"Failed to load from mlartifacts: {str(e)}")
        raise

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

def get_genre_distribution(top_books):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT id, genre FROM books WHERE id IN ({})'.format(','.join(['?'] * len(top_books))), tuple(top_books))
    genres = [row['genre'] for row in cursor.fetchall() if row['genre']]
    conn.close()
    genre_counts = defaultdict(int)
    for genre in genres:
        genre_counts[genre] += 1
    return dict(genre_counts)

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
    user_id = request.args.get('user_id', 'anonymous')
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM user_interactions WHERE user_id = ?", (user_id,))
        count = cursor.fetchone()[0]
        logger.debug(f"History count for user {user_id}: {count}")
        return jsonify({'hasHistory': count > 0})
    except Exception as e:
        logger.error(f"Database error in has_history: {str(e)}")
        return jsonify({"error": "Failed to check history"}), 500
    finally:
        conn.close()

@app.route('/api/click_history')
def get_click_history():
    user_id = request.args.get('user_id', 'anonymous')
    try:
        conn = get_db()
        query = '''
            SELECT ui.book_id, b.title, b.author, b.genre, ui.timestamp
            FROM user_interactions ui
            JOIN books b ON ui.book_id = b.id
            WHERE ui.user_id = ?
            ORDER BY ui.timestamp DESC
            LIMIT 50
        '''
        clicks = conn.execute(query, (user_id,)).fetchall()
        click_history = [
            {
                'book_id': click['book_id'],
                'title': click['title'],
                'author': click['author'],
                'genre': click['genre'],
                'timestamp': click['timestamp']
            } for click in clicks
        ]
        logger.debug(f"Returning click history for user {user_id} with {len(click_history)} entries")
        return jsonify({
            'clicks': click_history,
            'total_clicks': len(click_history)
        })
    except Exception as e:
        logger.error(f"Database error in get_click_history: {str(e)}")
        return jsonify({"error": "Failed to load click history"}), 500
    finally:
        conn.close()

@app.route('/api/historical_metrics')
def get_historical_metrics():
    date = request.args.get('date')
    try:
        conn = get_db()
        query = '''
            SELECT ui.book_id, ui.user_id, ui.timestamp
            FROM user_interactions ui
            WHERE DATE(ui.timestamp) = ?
        '''
        clicks = conn.execute(query, (date,)).fetchall()
        total_clicks = len(clicks)
        clicks_on_recommended = sum(1 for click in clicks if click['book_id'] in user_recommendations.get(click['user_id'], set()))
        total_recommendations = metrics['total_recommendations'] if metrics['total_recommendations'] > 0 else 1
        ctr = clicks_on_recommended / total_recommendations
        query = '''
            SELECT feedback_type, COUNT(*) as count
            FROM user_feedback
            WHERE DATE(timestamp) = ?
            GROUP BY feedback_type
        '''
        feedback = conn.execute(query, (date,)).fetchall()
        likes = next((row['count'] for row in feedback if row['feedback_type'] == 'like'), 0)
        dislikes = next((row['count'] for row in feedback if row['feedback_type'] == 'dislike'), 0)
        query = '''
            SELECT COUNT(DISTINCT user_id) as unique_users
            FROM user_feedback
            WHERE DATE(timestamp) = ?
        '''
        cursor = conn.execute(query, (date,))
        unique_users = cursor.fetchone()['unique_users']
        
        # Genre distribution
        query = '''
            SELECT b.genre, COUNT(*) as count
            FROM user_interactions ui
            JOIN books b ON ui.book_id = b.id
            WHERE DATE(ui.timestamp) = ? AND b.genre IS NOT NULL
            GROUP BY b.genre
        '''
        genres = conn.execute(query, (date,)).fetchall()
        genre_distribution = {row['genre']: row['count'] for row in genres}
        
        metrics_data = {
            'total_recommendations': total_recommendations,
            'total_clicks': total_clicks,
            'clicks_on_recommended': clicks_on_recommended,
            'ctr': ctr,
            'unique_users': unique_users,
            'likes': likes,
            'dislikes': dislikes,
            'diversity': metrics['diversity'],
            'coverage': metrics['coverage']
        }
        return jsonify({
            'metrics': metrics_data,
            'genre_distribution': genre_distribution
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
    global metrics
    book_id = data.get('bookId')
    event_type = data.get('eventType')
    user_id = data.get('userId', 'anonymous')
    if not book_id or not event_type:
        logger.warning("Invalid interaction data")
        return
    
    unique_users.add(user_id)
    metrics['unique_users'] = len(unique_users)
    
    try:
        conn = get_db()
        cursor = conn.cursor()
        if event_type == 'click':
            try:
                cursor.execute('INSERT INTO user_interactions (book_id, user_id) VALUES (?, ?)', (book_id, user_id))
                cursor.execute('UPDATE books SET clicks = clicks + 1 WHERE id = ?', (book_id,))
                metrics['total_clicks'] += 1
            except sqlite3.IntegrityError:
                logger.debug(f"Click for book {book_id} by user {user_id} already exists.")
        elif event_type == 'hover':
            cursor.execute('UPDATE books SET hovers = hovers + 1 WHERE id = ?', (book_id,))
        elif event_type in ['like', 'dislike']:
            try:
                cursor.execute('INSERT INTO user_feedback (book_id, user_id, feedback_type) VALUES (?, ?, ?)',
                              (book_id, user_id, event_type))
                metrics[event_type + 's'] += 1
                if book_id in user_recommendations[user_id]:
                    metrics['clicks_on_recommended'] += 1
                logger.info(f"Recorded {event_type} for book {book_id} by user {user_id}")
            except sqlite3.IntegrityError:
                logger.debug(f"Feedback for book {book_id} by user {user_id} already exists.")
        conn.commit()
        if event_type in ['click', 'hover']:
            updated_book = conn.execute('SELECT * FROM books WHERE id = ?', (book_id,)).fetchone()
            socketio.emit('update_counts', dict(updated_book))
        logger.info(f"Updated book {book_id} with {event_type}")
    except Exception as e:
        logger.error(f"Interaction error: {str(e)}")
    finally:
        conn.close()

    interaction = {
        'book_id': book_id,
        'event_type': event_type,
        'user_id': user_id,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    try:
        producer.send('user_interactions', interaction)
        producer.flush()
        logger.info(f"Sent to Kafka (interactions): {interaction}")
    except Exception as e:
        logger.error(f"Kafka send error (interactions): {str(e)}")

    send_metrics_update()

def send_metrics_update():
    global metrics
    top_books = []
    for user_id, books in user_recommendations.items():
        top_books.extend(list(books)[:20])
    top_books = list(set(top_books))[:20]
    genre_distribution = get_genre_distribution(top_books)
    metrics_data = {
        'total_recommendations': metrics['total_recommendations'],
        'total_clicks': metrics['total_clicks'],
        'clicks_on_recommended': metrics['clicks_on_recommended'],
        'ctr': metrics['clicks_on_recommended'] / metrics['total_recommendations'] if metrics['total_recommendations'] > 0 else 0,
        'unique_users': metrics['unique_users'],
        'likes': metrics['likes'],
        'dislikes': metrics['dislikes'],
        'diversity': metrics['diversity'],
        'coverage': metrics['coverage'],
        'timestamp': datetime.now().isoformat()
    }
    try:
        producer.send('metrics_topic', {
            'metrics': metrics_data,
            'genre_distribution': genre_distribution
        })
        producer.flush()
        logger.info(f"Sent metrics to Kafka: {metrics_data}")
    except Exception as e:
        logger.error(f"Kafka send error (metrics): {str(e)}")

@socketio.on('get_recommendations')
def handle_get_recommendations(data):
    global metrics
    logger.debug(f"Received get_recommendations with data: {data}")
    session_clicks = data.get('sessionClicks', [])
    user_id = data.get('userId', 'anonymous')
    
    conn = get_db()
    cursor = conn.cursor()
    historical_clicks = []
    try:
        cursor.execute('SELECT book_id FROM user_interactions WHERE user_id = ? ORDER BY timestamp ASC', (user_id,))
        historical_clicks = [row[0] for row in cursor.fetchall()]
    except sqlite3.OperationalError as e:
        logger.warning(f"Could not query user_interactions: {str(e)}. Proceeding with empty history.")
        historical_clicks = []
    
    cursor.execute('SELECT id, genre FROM books')
    book_genres = {row['id']: row['genre'] for row in cursor.fetchall()}
    total_books = len(book_genres)
    
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
    
    top_books = []
    if unique_books:
        sorted_pairs = sorted(zip(unique_probs, unique_books), reverse=True)
        top_books = [book for _, book in sorted_pairs][:20]
        if not metrics['has_recommended']:
            metrics['total_recommendations'] = 20
            metrics['has_recommended'] = True
        else:
            metrics['total_recommendations'] += len(top_books)
        user_recommendations[user_id].update(top_books)
        genres = [book_genres.get(book, None) for book in top_books if book_genres.get(book, None)]
        unique_genres = len(set(genres) - {None})
        metrics['diversity'] = unique_genres
        metrics['coverage'] = (len(set().union(*user_recommendations.values())) / total_books * 100) if total_books > 0 else 0
        if len(top_books) < 20:
            cursor.execute('SELECT id FROM books WHERE id NOT IN ({})'.format(','.join(['?'] * len(top_books))), tuple(top_books))
            remaining_books = [row['id'] for row in cursor.fetchall()]
            if remaining_books:
                additional_books = random.sample(remaining_books, min(20 - len(top_books), len(remaining_books)))
                top_books.extend(additional_books)
                metrics['total_recommendations'] += len(additional_books)
                user_recommendations[user_id].update(additional_books)
                genres = [book_genres.get(book, None) for book in top_books if book_genres.get(book, None)]
                metrics['diversity'] = len(set(genres) - {None})
                metrics['coverage'] = (len(set().union(*user_recommendations.values())) / total_books * 100) if total_books > 0 else 0
    else:
        cursor.execute('SELECT id FROM books')
        all_book_ids = [row['id'] for row in cursor.fetchall()]
        top_books = random.sample(all_book_ids, min(20, len(all_book_ids)))
        if not metrics['has_recommended']:
            metrics['total_recommendations'] = 20
            metrics['has_recommended'] = True
        else:
            metrics['total_recommendations'] += len(top_books)
        user_recommendations[user_id].update(top_books)
        genres = [book_genres.get(book, None) for book in top_books if book_genres.get(book, None)]
        metrics['diversity'] = len(set(genres) - {None})
        metrics['coverage'] = (len(set().union(*user_recommendations.values())) / total_books * 100) if total_books > 0 else 0
    
    logger.debug(f"Emitting recommendations: {top_books}")
    socketio.emit('recommendations', {'books': top_books, 'user_id': user_id}, to=request.sid)
    send_metrics_update()
    
    conn.close()

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5003, debug=True, use_reloader=False)