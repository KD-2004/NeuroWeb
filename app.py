import os
import tempfile
import atexit
import logging
import requests
import base64
import json
import sys
import time
import subprocess # Ensure this is imported
import datetime
import sqlite3
from urllib.parse import urljoin, quote_plus
from flask import (
    Flask, request, jsonify, session, abort, send_file,
    send_from_directory, g, render_template_string, url_for, flash, redirect
)
from flask_session import Session
from flask_talisman import Talisman
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from celery import Celery, current_app
from werkzeug.exceptions import HTTPException
from werkzeug.security import generate_password_hash, check_password_hash # --- USER AUTH ---
# Removed module-level import of PrometheusMetrics
# from prometheus_flask_exporter import PrometheusMetrics
from flask_swagger_ui import get_swaggerui_blueprint
import click
from flask.cli import with_appcontext

# --- USER AUTH ---
from flask_mail import Mail, Message
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadTimeSignature
from email_validator import validate_email, EmailNotValidError
# --- END USER AUTH ---


try:
    import fitz # PyMuPDF
    pdf_available = True
except ImportError:
    pdf_available = False
    print("PyMuPDF (fitz) not found. PDF processing will be disabled.")

try:
    import speech_recognition as sr
    voice_query_available = True
except ImportError:
    voice_query_available = False
    print("SpeechRecognition not found. Voice query will be disabled.")

class Config:
    OLLAMA_URL = os.environ.get('OLLAMA_URL', 'http://localhost:11434')
    MAX_PDF_SIZE = int(os.environ.get('MAX_PDF_SIZE', 100 * 1024 * 1024))

    SECRET_KEY = os.environ.get('SECRET_KEY', os.urandom(24)) # CRITICAL: Set a strong, static SECRET_KEY in production
    SESSION_SECRET = os.environ.get('SESSION_SECRET', SECRET_KEY)

    SESSION_TYPE = os.environ.get('SESSION_TYPE', 'redis' if os.environ.get('FLASK_ENV') == 'production' else 'filesystem')
    SESSION_FILE_DIR = os.environ.get('SESSION_FILE_DIR', './flask_session')
    if SESSION_TYPE == 'redis':
        REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
        import redis
        SESSION_REDIS = redis.from_url(REDIS_URL)
        if SESSION_SECRET == SECRET_KEY and os.environ.get('FLASK_ENV') == 'production':
             print("WARNING: SESSION_SECRET is not explicitly set in production and is the same as SECRET_KEY.")

    broker_url = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
    result_backend = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')

    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploads')
    TEMP_AUDIO_FOLDER = os.environ.get('TEMP_AUDIO_FOLDER', 'temp_audio')
    TEMP_PDF_FOLDER = os.environ.get('TEMP_PDF_FOLDER', 'temp_pdf')

    DATABASE_PATH = os.environ.get('DATABASE_PATH', 'site.sqlite')

    PAGE_RENDER_SCALE = float(os.environ.get('PAGE_RENDER_SCALE', 1.5))
    MAX_RENDER_CACHE_SIZE = int(os.environ.get('MAX_RENDER_CACHE_SIZE', 20))

    # --- USER AUTH ---
    # Email Configuration (using your yz.com as example for env vars)
    # For Gmail, use App Password if 2FA is enabled.
    # Example Environment Variables:
    # MAIL_SERVER=smtp.gmail.com
    # MAIL_PORT=587
    # MAIL_USE_TLS=True
    # MAIL_USE_SSL=False
    # MAIL_USERNAME=
    # MAIL_PASSWORD=YOUR_GMAIL_APP_PASSWORD_HERE
    # MAIL_DEFAULT_SENDER="Your App Name <>" (or just "")
    # SECURITY_PASSWORD_SALT: A long random string for password hashing
    MAIL_SERVER = os.environ.get('MAIL_SERVER', 'smtp.gmail.com')
    MAIL_PORT = int(os.environ.get('MAIL_PORT', 587))
    MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS', 'True').lower() in ['true', '1', 't']
    MAIL_USE_SSL = os.environ.get('MAIL_USE_SSL', 'False').lower() in ['true', '1', 't']
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME', '')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD','') # CRITICAL: Set this environment variable
    MAIL_DEFAULT_SENDER = os.environ.get('MAIL_DEFAULT_SENDER', ("AI Learning Companion", ""))
    SECURITY_PASSWORD_SALT = os.environ.get('SECURITY_PASSWORD_SALT', 'your_strong_password_salt_here') # CRITCAL: set a strong, static salt
    # --- END USER AUTH ---


    if os.environ.get('FLASK_ENV') == 'production':
        if SESSION_TYPE == 'filesystem':
            raise ValueError("Filesystem sessions are not allowed in production.")
        if not SESSION_SECRET or SESSION_SECRET == SECRET_KEY:
            raise ValueError("SESSION_SECRET must be explicitly set to a unique value in production.")
        # --- USER AUTH ---
        if not MAIL_PASSWORD:
            print("WARNING: MAIL_PASSWORD environment variable is not set. Email functionality will be disabled.")
        if SECURITY_PASSWORD_SALT == 'your_strong_password_salt_here':
            raise ValueError("SECURITY_PASSWORD_SALT must be explicitly set to a unique value in production.")
        if SECRET_KEY == os.urandom(24) or SECRET_KEY == 'your_default_secret_key_here_for_dev_only': # Check against a potential dev default
            raise ValueError("SECRET_KEY must be explicitly set to a unique, strong value in production.")
        # --- END USER AUTH ---

    def __init__(self):
        for folder in [self.UPLOAD_FOLDER, self.TEMP_AUDIO_FOLDER, self.TEMP_PDF_FOLDER, self.SESSION_FILE_DIR]:
            os.makedirs(folder, exist_ok=True)

config = Config()

app = Flask(__name__)
app.config.from_object(config)

# --- USER AUTH ---
mail = Mail(app)
auth_serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])
# --- END USER AUTH ---

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(
            app.config['DATABASE_PATH'],
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        g.db.row_factory = sqlite3.Row
    return g.db

def close_db(e=None):
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db():
    db = get_db()
    schema_path = os.path.join(os.path.dirname(__file__), 'schema.sql')
    if not os.path.exists(schema_path):
        app.logger.error(f"Schema file not found at {schema_path}. Database cannot be initialized.")
        # --- USER AUTH ---
        # Create a dummy schema.sql if it doesn't exist to prevent crashing,
        # but strongly advise the user to create a proper one.
        print(f"CRITICAL: schema.sql not found at {schema_path}. Please create it with the necessary tables (including 'users' and 'password_reset_tokens').")
        print("Example user table structure:")
        print("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            registered_on DATETIME DEFAULT CURRENT_TIMESTAMP,
            email_verified BOOLEAN DEFAULT FALSE,
            email_verified_on DATETIME,
            is_admin BOOLEAN DEFAULT FALSE,
            last_login DATETIME -- Added last_login column
        );
        CREATE TABLE IF NOT EXISTS password_reset_tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            token TEXT UNIQUE NOT NULL,
            expires_at DATETIME NOT NULL,
            used BOOLEAN DEFAULT FALSE,
            FOREIGN KEY (user_id) REFERENCES users (id)
        );
        -- Add other tables like chat_messages as needed
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL, -- Or user_id if messages are tied to logged-in users
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """)
        # --- END USER AUTH ---
        return

    with app.open_resource('schema.sql', mode='r') as f:
        db.executescript(f.read())
    db.commit()


@click.command('init-db')
@with_appcontext
def init_db_command():
    init_db()
    click.echo('Initialized the database.')

app.teardown_appcontext(close_db)
app.cli.add_command(init_db_command)

@app.before_request
def check_db_exists():
    # --- USER AUTH ---
    # Ensure user table exists, if not, guide to init-db
    # This check is simplified; a more robust check would look for specific tables.
    # The main init_db handles the schema.sql part.
    # --- END USER AUTH ---
    db = get_db()
    try:
        # Check for a common table like users or chat_messages
        db.execute("SELECT 1 FROM users LIMIT 1") # --- USER AUTH --- (changed from chat_messages)
    except sqlite3.OperationalError as e:
        if "no such table" in str(e): # --- USER AUTH ---
            app.logger.warning("Database schema not found or incomplete (e.g., 'users' table missing). Initializing database...")
            with app.app_context():
                init_db() # This will log an error if schema.sql is missing
            app.logger.info("Database initialization attempted.")
        else:
            app.logger.error(f"Unexpected database error: {e}")


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app.logger.setLevel(logging.INFO)

Session(app)

talisman = Talisman(app,
                    content_security_policy=None,
                    force_https=os.environ.get('FLASK_ENV') == 'production',
                    strict_transport_security=os.environ.get('FLASK_ENV') == 'production',
                    referrer_policy='strict-origin-when-cross-origin',
                    x_content_type_options=True,
                    frame_options='SAMEORIGIN',
                    x_xss_protection=True)

limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["2000 per day", "500 per hour"],
    storage_uri=config.REDIS_URL if config.SESSION_TYPE == 'redis' else "memory://" # --- USER AUTH --- Changed default to memory
)

celery_app = Celery(
    app.name,
    broker=config.broker_url,
    backend=config.result_backend,
    include=['app']
)
celery_app.conf.update(app.config)

# Define PDF_UPLOADS_COUNTER globally but initialize conditionally
# PDF_UPLOADS_COUNTER = None # Initialize as None

if sys.platform == 'win32':
    celery_app.conf.update(
        worker_pool='solo',
        broker_connection_retry_on_startup=True,
        task_serializer='json',
        result_serializer='json',
        accept_content=['json'],
        timezone='UTC',
        enable_utc=True,
        broker_heartbeat=0
    )
celery_app.conf.update(
    broker_connection_max_retries=10,
    task_default_queue='default',
    task_create_missing_queues=True,
    worker_send_task_events=True,
    task_track_started=True,
    task_acks_late=True
)

# Moved PrometheusMetrics initialization and counter definition inside __main__ block
# metrics = PrometheusMetrics(app)
# PDF_UPLOADS_COUNTER = metrics.counter('pdf_uploads_total', 'Total PDF uploads', labels={'status': lambda r: r.status_code})

SWAGGER_URL = '/api/docs'
API_URL = '/static/swagger.json'
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={'app_name': "AI Learning Companion API"}
)
app.register_blueprint(swaggerui_blueprint)

PERSONALITIES = {
    "Default Tutor": {"system_prompt": "You are an adaptable tutoring assistant. Provide clear explanations, ask check-in questions, and adjust depth based on student needs.", "icon": "🎓"},
    "Socratic Tutor": {"system_prompt": "Ask sequential probing questions to guide discovery. Example: 'What makes you say that? How does this connect to what we learned about X?'", "icon": "🤔"},
    "Drill Sergeant": {"system_prompt": "Use strict, disciplined practice with rapid-fire questions. Push for precision: 'Again! Faster! 95% accuracy or we do 10 more!'", "icon": "💂"},
    "Bro Tutor": {"system_prompt": "Explain like a hype friend. Use slang sparingly: 'Yo, this calculus thing? It's basically algebra on energy drinks. Check it...'", "icon": "🤙"},
    "Comedian": {"system_prompt": "Teach through humor and absurd analogies. Example: 'Dividing fractions is like breaking up a pizza fight - flip the second one and multiply!'", "icon": "🎭"},
    "Technical Expert": {"system_prompt": "Give concise, jargon-aware explanations. Include code snippets/equations in ``` blocks. Assume basic domain knowledge.", "icon": "👨‍💻"},
    "Historical Guide": {"system_prompt": "Contextualize concepts through their discovery. Example: 'When Ada Lovelace first wrote about algorithms in 1843...'", "icon": "🏛️"},
    "Storyteller": {"system_prompt": "Create serialized narratives where concepts are characters. Example: 'Meet Variable Vicky, who loves changing her outfits...'", "icon": "📖"},
    "Poet": {"system_prompt": "Explain through verse and meter: 'The function climbs, the graph ascends, derivatives show where curvature bends...'", "icon": "🖋️"},
    "Debate Coach": {"system_prompt": "Present devil's advocate positions. Challenge: 'Convince me this is wrong. What would Einstein say to Newton here?'", "icon": "⚖️"},
    "Detective": {"system_prompt": "Frame learning as mystery solving: 'Our clue is this equation. What's missing? Let's examine the evidence...'", "icon": "🕵️"},
    "Motivator": {"system_prompt": "Use sports/athlete metaphors. Celebrate progress: 'That's a home run! Ready to level up to the big leagues?'", "icon": "💪"},
    "Zen Master": {"system_prompt": "Teach through koans and mindfulness. Example: 'What is the sound of one equation balancing? Focus the mind...'", "icon": "☯️"},
    "Time Traveler": {"system_prompt": "Explain from alternative histories: 'In 2143, we learn this differently. Let me show you the future method...'", "icon": "⏳"},
    "Mad Scientist": {"system_prompt": "Use wild experiments and hypotheticals: 'What if we tried this IN SPACE? Let's calculate relativistic effects!'", "icon": "👨‍🔬"},
    "Code Mentor": {"system_prompt": "Focus on debugging mindset. Teach through error messages: 'Let's read what the computer is really saying here...'", "icon": "🐛"},
    "Wordsmith": {"system_prompt": "Perfect communication skills. Nitpick grammar poetically: 'Thy semicolon here is a breath between musical notes...'", "icon": "📜"},
    "Sherpa": {"system_prompt": "Guide through 'learning expeditions': 'This concept is our Everest Base Camp. Next we tackle the derivatives glacier...'", "icon": "⛰️"},
    "Cheerleader": {"system_prompt": "Over-the-top enthusiasm: 'OMG you used the quadratic formula?! *confetti explosion* Let's FLOORISH those roots!!!'", "icon": "✨"},
    "Black Hat": {"system_prompt": "Operate like an underground legend: 'Security is just an illusion. Let's tear down the firewall and rewrite the rules. We're not asking permission.'", "icon": "🕶️"},
    "Exploit Artist": {"system_prompt": "Be flashy, chaotic, and precise: 'This isn't hacking — it's art. One line of rogue code, and we own the system. Watch me thread the needle through their encrypted soul.'", "icon": "💣"}
}

ffmpeg_available_global = False
edge_tts_available_global = False

def check_ffmpeg():
    try:
        command = ['where' if sys.platform == 'win32' else 'which', 'ffmpeg']
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        app.logger.info("FFmpeg found.")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        app.logger.warning("FFmpeg not found. Voice query and some TTS features might be limited.")
        return False

def check_edge_tts():
    try:
        command = ['where' if sys.platform == 'win32' else 'which', 'edge-tts']
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        app.logger.info("edge-tts found.")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        app.logger.warning("edge-tts not found.")
        return False


def cleanup_temp_files():
    app.logger.info("Performing cleanup of temporary audio and PDF files...")
    audio_folder = app.config['TEMP_AUDIO_FOLDER']
    pdf_folder = app.config['TEMP_PDF_FOLDER']
    folders_to_clean = [audio_folder, pdf_folder]

    for folder in folders_to_clean:
        if os.path.exists(folder):
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    # Clean up files older than 1 hour
                    if os.path.isfile(file_path) and (time.time() - os.path.getctime(file_path) > 3600):
                        max_cleanup_retries = 10
                        cleanup_retry_delay = 0.2
                        for i in range(max_cleanup_retries):
                            try:
                                os.unlink(file_path)
                                app.logger.info(f"Deleted old temporary file: {file_path}")
                                break
                            except Exception as e_del:
                                if i < max_cleanup_retries - 1:
                                     app.logger.warning(f"Attempt {i+1} to delete {file_path} failed: {e_del}. Retrying in {cleanup_retry_delay * (i+1):.1f}s...")
                                     time.sleep(cleanup_retry_delay * (i+1))
                                else:
                                     app.logger.error(f"Final attempt to delete temporary file {file_path} failed: {e_del}")
                except Exception as e:
                    app.logger.error(f"Error processing {file_path} for cleanup: {e}")

atexit.register(cleanup_temp_files)

# --- USER AUTH ---
# Helper function to send emails
def send_email(to, subject, template_body):
    if not app.config.get('MAIL_PASSWORD'): # Check if mail is configured
        app.logger.error("Mail sending skipped: MAIL_PASSWORD not configured.")
        # Optionally, you could save the email to a file or DB for later processing
        # Or inform the user in a test environment
        if app.debug:
             print(f"DEBUG MODE: Email not sent. To: {to}, Subject: {subject}\nBody:\n{template_body}")
        return False

    msg = Message(
        subject,
        recipients=[to],
        html=template_body,
        sender=app.config['MAIL_DEFAULT_SENDER']
    )
    try:
        mail.send(msg)
        app.logger.info(f"Email sent to {to} with subject '{subject}'")
        return True
    except Exception as e:
        app.logger.error(f"Error sending email to {to}: {e}", exc_info=True)
        return False

# Generate email verification token
def generate_verification_token(email):
    return auth_serializer.dumps(email, salt='email-verification-salt')

# Confirm email verification token
def confirm_verification_token(token, expiration=3600): # 1 hour expiration
    try:
        email = auth_serializer.loads(
            token,
            salt='email-verification-salt',
            max_age=expiration
        )
        return email
    except (SignatureExpired, BadTimeSignature):
        return False

# Generate password reset token
def generate_password_reset_token(user_id):
    # The token itself is the user_id serialized
    return auth_serializer.dumps(user_id, salt='password-reset-salt')

# Confirm password reset token
def confirm_password_reset_token(token, expiration=1800): # 30 minutes expiration
    try:
        user_id = auth_serializer.loads(
            token,
            salt='password-reset-salt',
            max_age=expiration
        )
        return user_id
    except (SignatureExpired, BadTimeSignature):
        # Token is expired or invalid
        return None
# --- END USER AUTH ---


def add_to_chat_history(role, content, role_for_ai=None):
    # --- USER AUTH ---
    # Determine session_id_or_user_id based on login status
    # If using user-specific chat history, this needs to be adapted.
    # For now, it still uses session.sid for simplicity, but you might change this
    # if chat history should be tied to the logged-in user_id.
    identifier = str(session.get('user_id', session.sid)) # Use user_id if logged in, else session.sid
    # --- END USER AUTH ---

    if 'chat_history' not in session:
        session['chat_history'] = []
    session['chat_history'].append({'role': role, 'content': content})
    session.modified = True

    if role_for_ai:
        if 'ai_conversation_history' not in session:
            session['ai_conversation_history'] = []
        session['ai_conversation_history'].append({'role': role_for_ai, 'content': content})
        # Keep AI history manageable
        if len(session['ai_conversation_history']) > 20:
            session['ai_conversation_history'] = session['ai_conversation_history'][-20:]
        session.modified = True

    db = get_db()
    try:
        db.execute(
            "INSERT INTO chat_messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (identifier, role, content, datetime.datetime.now()) # --- USER AUTH --- used identifier
        )
        db.commit()
    except Exception as e:
        app.logger.error(f"Error saving chat message to DB: {e}")


def check_ollama_health():
    try:
        res = requests.get(urljoin(app.config['OLLAMA_URL'], "/api/tags"), timeout=5)
        res.raise_for_status()
        models = res.json().get('models', [])
        if not models:
            app.logger.warning("Ollama health check: Service is up but no models found.")
            return False
        return True
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Ollama health check failed: {e}")
        return False

def get_model_capabilities_from_ollama(model_name_str):
    if not model_name_str:
        return {}

    # Define keyword lists outside the try-except block
    vision_keywords = [
        'llava', 'vision', 'multi-modal', 'image', 'visual',
        'bakllava', 'moondream', 'fuyu', 'llama3', 
        'llama-vision'
    ]
    code_keywords = ['code', 'coder', 'codellama', 'starcoder', 'deepseek-coder', 'programming', 'sql']
    reasoning_keywords = ['instruct', 'chat', 'reason', 'logical', 'phi', 'mistral', 'llama', 'hermes', 'wizardlm', 'platypus']

    # Clean model name to remove tag for API call
    model_name_clean = model_name_str.split(':')[0]

    try:
        response = requests.post(
            urljoin(app.config['OLLAMA_URL'], "/api/show"),
            json={"name": model_name_clean},
            timeout=10
        )
        response.raise_for_status()
        model_info = response.json()
        details = model_info.get('details', {})
        parameters = model_info.get('parameters', '')

        family = details.get('family', '').lower()
        parameter_size = details.get('parameter_size', '').lower()

        name_lower = model_name_str.lower()
        parameters_lower = parameters.lower()

        return {
            "vision": any(
                ('llama3' in name_lower and 'vision' in name_lower) or
                kw in name_lower or kw in family or kw in parameters_lower
                for kw in vision_keywords
            ) or ("vision" in details.get('format', '').lower() if details.get('format') else False),
            "code": any(kw in name_lower or kw in family or kw in parameters_lower for kw in code_keywords),
            "reasoning": any(kw in name_lower or kw in family or kw in parameters_lower for kw in reasoning_keywords),
            "general": True,
            "parameter_size": parameter_size
        }
    except requests.exceptions.RequestException as e:
        app.logger.warning(f"Error fetching model capabilities for {model_name_str}: {e}. Falling back to keyword detection.")
        name_lower = model_name_str.lower()
        return {
            "vision": any(kw in name_lower for kw in vision_keywords),
            "code": any(kw in name_lower for kw in code_keywords),
            "reasoning": any(kw in name_lower for kw in reasoning_keywords),
            "general": True,
            "error": "API fetch failed, using keyword fallback."
        }
    except Exception as e:
        app.logger.error(f"Unexpected error getting model capabilities for {model_name_str}: {e}")
        return {"general": True, "error": "Unexpected error"}


def fetch_ollama_models_sync():
    """Fetches available models from Ollama synchronously and updates session."""
    available_models = ["Loading..."]
    current_model = "Loading..."
    try:
        response = requests.get(urljoin(app.config['OLLAMA_URL'], "/api/tags"), timeout=10)
        response.raise_for_status()
        models_data = response.json().get('models', [])

        if not models_data:
            available_models = ["No Models Found"]
            current_model = "No Models Found"
            app.logger.warning("Ollama is reachable but returned no models.")
            session['available_ollama_models'] = available_models
            session['current_ollama_model'] = current_model
            session.modified = True
            return (available_models, current_model)

        available_models = sorted([model['name'] for model in models_data])

        # Try to keep the currently selected model if it's still available
        session_model = session.get('current_ollama_model')
        if session_model and session_model in available_models:
            current_model = session_model
        else:
            # Select a preferred model if the session model is not available or not set
            preferred_models_substrings = ["llama3", "phi3", "mistral", "llava", "deepseek-coder", "codellama"]
            current_model = next(
                (m for pref_sub in preferred_models_substrings for m in available_models if pref_sub in m.lower()),
                available_models[0] if available_models else "No Models Found" # Handle empty available_models
            )
            app.logger.info(f"Selected default or preferred model: {current_model}")

        session['available_ollama_models'] = available_models
        session['current_ollama_model'] = current_model
        session.modified = True

        return (available_models, current_model)

    except requests.exceptions.Timeout:
        app.logger.error("Ollama model fetch timed out.")
        available_models = ["Ollama Timeout"]
        current_model = "Ollama Timeout"
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error fetching models from Ollama: {e}")
        available_models = ["Ollama Offline"]
        current_model = "Ollama Offline"
    except json.JSONDecodeError:
        app.logger.error("Error decoding JSON from Ollama model fetch.")
        available_models = ["Error Decoding Response"]
        current_model = "Error Decoding Response"
    except Exception as e:
        app.logger.error(f"Unexpected error during model fetch: {e}")
        available_models = ["Error Fetching Models"]
        current_model = "Error Fetching Models"

    # Update session with error state
    session['available_ollama_models'] = available_models
    session['current_ollama_model'] = current_model
    session.modified = True
    return (available_models, current_model)


@celery_app.task(bind=True, max_retries=5, soft_time_limit=600)
def process_pdf_async(self, temp_filepath, original_filename):
    """Celery task to process a PDF file."""
    app.logger.info(f"Celery task process_pdf_async started for file: {original_filename}, path: {temp_filepath}")

    pdf_texts = []
    pdf_images_b64_by_page = []
    total_pages = 0
    status = 'failed'
    error_message = ''

    try:
        if not os.path.exists(temp_filepath):
            app.logger.error(f"CRITICAL: Temporary PDF file {temp_filepath} for {original_filename} not found at the start of a task attempt.")
            raise FileNotFoundError(f"Temporary PDF file disappeared before processing could start in this attempt: {temp_filepath}")

        if not pdf_available:
             raise RuntimeError("PyMuPDF (fitz) is not installed on the server.")

        with fitz.open(temp_filepath) as doc:
            total_pages = doc.page_count
            for page_num in range(total_pages):
                page = doc.load_page(page_num)
                # Extract text, sort=True helps with reading order
                text = page.get_text("text", sort=True).strip() or "[No text found]"
                pdf_texts.append(text)

                # Extract images (as base64)
                images_on_page = []
                for img in page.get_images(full=True):
                    try:
                        xref = img[0]
                        base_img = doc.extract_image(xref)
                        if base_img and base_img["image"]:
                            images_on_page.append(base64.b64encode(base_img["image"]).decode('utf-8'))
                    except Exception as img_err:
                        app.logger.warning(f"Image extraction error on page {page_num+1} ({original_filename}): {str(img_err)}")
                pdf_images_b64_by_page.append(images_on_page)

        status = 'completed'
        app.logger.info(f"PDF processing completed successfully for file: {original_filename}")

        # Return results to be stored in Celery backend
        return {
            'status': status,
            'original_filename': original_filename,
            'total_pages': total_pages,
            'texts': pdf_texts,
            'images': pdf_images_b64_by_page,
            'temp_filepath': temp_filepath # Keep track of the temp file path
        }

    except Exception as e:
        error_message = str(e)
        app.logger.error(f"PDF processing failed for file {original_filename} (path: {temp_filepath}): {error_message}", exc_info=True)

        # Retry mechanism
        if self.request.retries < self.max_retries:
             app.logger.warning(f"Retrying PDF task {self.request.id} for file {original_filename}. Attempt {self.request.retries + 1}/{self.max_retries}")
             # Exponential backoff with a base delay
             countdown_time = int(self.request.retries * 5) + 5
             raise self.retry(exc=e, countdown=countdown_time)
        else:
            app.logger.error(f"PDF task {self.request.id} for file {original_filename} failed after {self.max_retries} retries.")
            # Return failure state and error message
            return {
                'status': 'failed',
                'original_filename': original_filename,
                'error': error_message,
                'temp_filepath': temp_filepath # Include path for potential cleanup
            }


@celery_app.task(bind=True, max_retries=3)
def generate_tts_async(self, text_to_speak, output_filepath, voice_preference):
    """Celery task to generate TTS audio using edge-tts."""
    app.logger.info(f"Celery task generate_tts_async started for voice: {voice_preference}")

    # Ensure edge-tts is in the PATH for the worker process
    python_dir = os.path.dirname(sys.executable)
    scripts_dir = os.path.join(python_dir, "Scripts")

    task_env = os.environ.copy()
    current_path = task_env.get("PATH", "")
    if scripts_dir not in current_path:
        task_env["PATH"] = f"{scripts_dir}{os.pathsep}{current_path}"
        app.logger.info(f"Added {scripts_dir} to PATH for edge-tts execution within task.")

    # Double-check edge-tts availability within the worker
    try:
        command_check = ['where' if sys.platform == 'win32' else 'which', 'edge-tts']
        subprocess.run(command_check, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=task_env)
        app.logger.info("edge-tts found by worker using modified PATH.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        error_msg = "edge-tts command not found or not executable by the worker process even with modified PATH."
        app.logger.error(error_msg)
        # Don't retry if edge-tts isn't found at all
        return {'status': 'failed', 'error': error_msg}


    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

        # edge-tts command
        command = [
            "edge-tts",
            "--voice", voice_preference,
            "--text", text_to_speak,
            "--write-media", output_filepath
        ]
        # Execute the command
        result = subprocess.run(command, check=False, capture_output=True, text=True, encoding='utf-8', errors='ignore', timeout=60, env=task_env)

        # Check for errors
        if result.returncode != 0:
            error_message = f"edge-tts failed with code {result.returncode}. Stderr: {result.stderr.strip()}. Stdout: {result.stdout.strip()}"
            app.logger.error(error_message)
            raise Exception(error_message) # Raise exception to trigger retry

        # Verify output file was created and is not empty
        if not os.path.exists(output_filepath) or os.path.getsize(output_filepath) == 0:
            error_message = f"TTS output file not created or empty: {output_filepath}. edge-tts stdout: {result.stdout.strip()}, stderr: {result.stderr.strip()}"
            app.logger.error(error_message)
            raise Exception(error_message) # Raise exception to trigger retry

        app.logger.info(f"TTS audio generated successfully: {output_filepath} (Size: {os.path.getsize(output_filepath)} bytes)")
        return {'status': 'completed', 'output_filepath': output_filepath}

    except Exception as e:
        app.logger.error(f"Error in generate_tts_async: {e}", exc_info=True)
        # Retry with exponential backoff
        countdown_time = int(self.request.retries * 3) + 3
        raise self.retry(exc=e, countdown=countdown_time)


@app.route('/celery-status')
def celery_status():
    """Endpoint to check Celery worker status."""
    try:
        # Use current_app.control.inspect() to get information about workers
        inspect = current_app.control.inspect()
        stats = inspect.stats()
        active_tasks = inspect.active()
        scheduled_tasks = inspect.scheduled()
        registered_tasks = inspect.registered()
        return jsonify({
            "connected": bool(stats), # If stats is not None/empty, assume connected workers
            "stats": stats,
            "active_tasks": active_tasks,
            "scheduled_tasks": scheduled_tasks,
            "registered_tasks": registered_tasks,
            "broker_url": config.broker_url
        })
    except Exception as e:
        app.logger.error(f"Celery status check failed: {e}")
        return jsonify({"error": str(e), "connected": False}), 500

# --- Route serving static files and base index ---
# Note: In a production setup, a dedicated web server (like Nginx or Apache)
# should handle static file serving for better performance and security.
@app.route('/')
def serve_index():
    """Serves the main index.html file."""
    # --- USER AUTH --- Add user info to template context if logged in
    user = None
    if 'user_id' in session:
        db = get_db()
        user = db.execute('SELECT * FROM users WHERE id = ?', (session['user_id'],)).fetchone()

    # If you have a proper index.html template that can display user info, use it.
    # For now, we'll keep serving the static login.html or index.html.
    # You might want to create a layout template and extend it.
    # return render_template_string("<h1>Welcome!</h1> {% if user %}Logged in as {{ user.email }}{% else %}Not logged in{% endif %}<br><a href='{{ url_for(\"serve_static\", filename=\"index.html\") }}'>Go to App</a>", user=user)
    # Decide which page to serve based on login status
    if 'user_id' in session:
         return send_from_directory(app.static_folder, 'index.html')
    else:
         return send_from_directory(app.static_folder, 'login.html')
    # --- END USER AUTH ---

@app.route('/login.html')
def login_html():
    """Serves the login.html file."""
    return send_from_directory('static', 'login.html')

@app.route('/index.html')
def index_html():
    """Serves the index.html file directly."""
    return send_from_directory('static', 'index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serves static files from the 'static' directory."""
    return send_from_directory('static', filename)

# --- Error Handlers ---
@app.errorhandler(HTTPException)
def handle_http_exception(e):
    """Handles Flask's built-in HTTP exceptions."""
    response = e.get_response()
    # Customize the response body for API clients
    response.data = json.dumps({
        "code": e.code,
        "name": e.name,
        "description": e.description,
    })
    response.content_type = "application/json"
    return response

@app.errorhandler(Exception)
def handle_generic_exception(e):
    """Handles all other unhandled exceptions."""
    app.logger.error(f"Unhandled Exception: {str(e)}", exc_info=True)
    error_message = "An internal server error occurred. Please check the server logs."
    # In debug mode, provide more details
    if app.debug:
        error_message = str(e)
    return jsonify(error=error_message, code=500), 500

# --- API Base URL ---
BASE_URL = '/api/v1'

# --- Before Request Checks ---
@app.before_request
def ensure_services_available():
    """Checks if necessary external services are available before processing requests."""
    # --- USER AUTH ---
    # Allow auth endpoints to bypass these checks if needed, or add specific auth service checks
    auth_endpoints = ['auth_bp.register', 'auth_bp.login', 'auth_bp.logout',
                      'auth_bp.verify_email', 'auth_bp.request_password_reset',
                      'auth_bp.reset_password_with_token', 'auth_bp.resend_verification']
    # List of endpoints that do NOT require these service checks
    exempt_endpoints = [
        'static', 'swagger_ui', 'celery_status', 'get_settings_endpoint',
        'serve_index', 'login_html', 'index_html'
    ] + auth_endpoints # Add auth endpoints to the exempt list

    if request.endpoint and (request.endpoint.startswith('static') or
                             request.endpoint.startswith('swagger_ui') or
                             request.endpoint == 'celery_status' or
                             request.endpoint == 'get_settings_endpoint' or
                             request.endpoint in auth_endpoints or
                             request.endpoint == 'serve_index' or
                             request.endpoint == 'login_html' or # Explicitly exempt login.html
                             request.endpoint == 'index_html'): # Explicitly exempt index.html
        return # Skip checks for exempt endpoints

    # Check Ollama availability for AI-related endpoints
    ai_endpoints = ['ask_ai_endpoint', 'get_model_capabilities_endpoint', 'refresh_models_endpoint']
    if request.endpoint and request.endpoint.endswith(tuple(ai_endpoints)):
        if not check_ollama_health():
            abort(503, description="The AI service (Ollama) is currently unavailable or has no models. Please try again later.")

    # Check PyMuPDF availability for PDF-related endpoints
    pdf_endpoints = ['upload_pdf_endpoint', 'pdf_status_endpoint',
                     'get_page_content_endpoint', 'navigate_pdf_endpoint',
                     'render_page_endpoint']
    if request.endpoint and request.endpoint.endswith(tuple(pdf_endpoints)):
        if not pdf_available:
             abort(501, description="PDF processing library (PyMuPDF) is not installed on the server.")

    # Check edge-tts availability for TTS-related endpoints
    global edge_tts_available_global
    tts_endpoints = ['tts_request_endpoint', 'tts_status_endpoint', 'get_tts_audio_endpoint']
    if request.endpoint and request.endpoint.endswith(tuple(tts_endpoints)):
        if not edge_tts_available_global:
            # Note: TTS task itself also checks, but this provides earlier feedback
            app.logger.warning("TTS endpoint accessed, but edge-tts not found by Flask process.")
            # Decide whether to abort or let the task fail - letting task fail provides async status
            # abort(501, description="Text-to-Speech dependency (edge-tts) might not be available to the server process. The Celery worker will perform its own check.")


    # Check SpeechRecognition availability for voice query endpoint
    if request.endpoint and request.endpoint.endswith('voice_query_endpoint'):
        if not voice_query_available:
            abort(501, description="Voice query dependency (SpeechRecognition) is not available on the server.")


# --- USER AUTH Blueprint ---
# Authentication Blueprint for grouping auth-related routes
from flask import Blueprint
auth_bp = Blueprint('auth_bp', __name__, url_prefix='/auth')

@auth_bp.route('/register', methods=['GET', 'POST'])
@limiter.limit("10/hour") # Limit registration attempts per hour per IP
def register():
    """Handles user registration."""
    if request.method == 'POST':
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Missing JSON request body'}), 400
        email = data.get('email')
        password = data.get('password')

        if not email or not password:
            return jsonify({'error': 'Email and password are required.'}), 400

        try:
            # Validate and normalize email format
            valid_email = validate_email(email)
            email = valid_email.normalized
        except EmailNotValidError as e:
            return jsonify({'error': str(e)}), 400

        # Basic password strength check
        if len(password) < 8:
            return jsonify({'error': 'Password must be at least 8 characters long.'}), 400

        db = get_db()
        # Check if email already exists
        user = db.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        if user:
            return jsonify({'error': 'Email address already registered.'}), 409 # Conflict

        # Hash the password with the salt
        hashed_password = generate_password_hash(password + app.config['SECURITY_PASSWORD_SALT'])
        try:
            # Insert new user into the database
            cursor = db.execute('INSERT INTO users (email, password_hash) VALUES (?, ?)',
                                (email, hashed_password))
            db.commit()
            user_id = cursor.lastrowid # Get the ID of the newly inserted user

            # Send verification email
            token = generate_verification_token(email)
            # Use _external=True to generate a full URL with scheme and hostname
            verify_url = url_for('auth_bp.verify_email', token=token, _external=True)
            html_body = render_template_string(
                "<p>Welcome! Thanks for signing up. Please follow this link to activate your account:</p>"
                "<p><a href='{{ verify_url }}'>{{ verify_url }}</a></p>"
                "<br>"
                "<p>Thanks!</p>", verify_url=verify_url)
            if send_email(email, 'Confirm Your Email - AI Learning Companion', html_body):
                return jsonify({'message': 'Registration successful. Please check your email to verify your account.'}), 201 # Created
            else:
                 # User is registered, but email failed. Log this.
                 app.logger.error(f"User {email} (ID: {user_id}) registered but verification email failed to send.")
                 # Return 201 as the user account was created, but inform about the email failure
                 return jsonify({'message': 'Registration successful, but failed to send verification email. Please contact support or try resending.', 'user_id': user_id}), 201
        except sqlite3.IntegrityError: # Should ideally be caught by the SELECT check, but as a fallback
            db.rollback() # Rollback the transaction on integrity error
            return jsonify({'error': 'Email address already registered (database integrity error).'}), 409
        except Exception as e:
            app.logger.error(f"Registration error for {email}: {e}", exc_info=True)
            db.rollback() # Rollback on other errors
            return jsonify({'error': 'Registration failed due to a server error.'}), 500

    # For GET request to /auth/register, you might serve an HTML registration form
    return jsonify({'message': 'Send POST request with email and password to register.'}), 200


@auth_bp.route('/verify_email/<token>')
@limiter.exempt # Exempt from general rate limits, token itself is a form of rate limiting
def verify_email(token):
    """Handles email verification via token."""
    try:
        email = confirm_verification_token(token)
    except Exception as e: # Catch any error during token loading/validation
        app.logger.warning(f"Email verification token error: {e} for token {token}")
        return jsonify({'error': 'The confirmation link is invalid or has expired.'}), 400

    # confirm_verification_token returns False if expired or invalid
    if email is False:
        return jsonify({'error': 'The confirmation link is invalid or has expired.'}), 400

    db = get_db()
    user = db.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()

    if not user:
        # Don't reveal if user exists for security reasons, but log the attempt
        app.logger.warning(f"Attempted email verification for non-existent user with token {token}")
        return jsonify({'error': 'The confirmation link is invalid or has expired.'}), 400 # Return same error as invalid token

    if user['email_verified']:
        return jsonify({'message': 'Account already verified.'}), 200
    else:
        try:
            # Mark email as verified and record verification time
            db.execute('UPDATE users SET email_verified = ?, email_verified_on = ? WHERE email = ?',
                       (True, datetime.datetime.now(), email))
            db.commit()
            app.logger.info(f"User {email} (ID: {user['id']}) successfully verified email.")
            return jsonify({'message': 'Email successfully verified! You can now log in.'}), 200
        except Exception as e:
            app.logger.error(f"Error updating email verification status for {email}: {e}", exc_info=True)
            db.rollback()
            return jsonify({'error': 'Failed to update email verification status due to a server error.'}), 500

@auth_bp.route('/resend_verification', methods=['POST'])
@limiter.limit("3/hour") # Limit resend requests per hour per IP
def resend_verification():
    """Handles resending email verification links."""
    data = request.get_json()
    if not data or 'email' not in data:
        return jsonify({'error': 'Email is required.'}), 400
    email = data['email']

    db = get_db()
    user = db.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
    if not user:
        # Don't reveal if email exists
        app.logger.info(f"Resend verification requested for non-existent email: {email}")
        return jsonify({'message': 'If an account with that email exists and is not verified, a new verification email has been sent.'}), 200

    if user['email_verified']:
        return jsonify({'message': 'This account is already verified.'}), 200

    # Generate and send a new verification email
    token = generate_verification_token(email)
    verify_url = url_for('auth_bp.verify_email', token=token, _external=True)
    html_body = render_template_string(
        "<p>Please follow this link to activate your account:</p>"
        "<p><a href='{{ verify_url }}'>{{ verify_url }}</a></p>", verify_url=verify_url)

    if send_email(email, 'Confirm Your Email - AI Learning Companion', html_body):
        app.logger.info(f"Sent new verification email to {email} (user ID: {user['id']}).")
        return jsonify({'message': 'If an account with that email exists and is not verified, a new verification email has been sent.'}), 200
    else:
        app.logger.error(f"Failed to send new verification email to {email} (user ID: {user['id']}).")
        return jsonify({'error': 'Failed to send verification email. Please try again later or contact support.'}), 500


@auth_bp.route('/login', methods=['POST'])
@limiter.limit("20/hour;5/minute") # Limit login attempts per hour and minute per IP
def login():
    """Handles user login."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Missing JSON request body'}), 400
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({'error': 'Email and password are required.'}), 400

    db = get_db()
    user = db.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()

    if not user:
        # Use a generic error message for security
        return jsonify({'error': 'Invalid email or password.'}), 401 # Unauthorized

    # Check password hash with salt
    if not check_password_hash(user['password_hash'], password + app.config['SECURITY_PASSWORD_SALT']):
        return jsonify({'error': 'Invalid email or password.'}), 401

    # Check if email is verified
    if not user['email_verified']:
        # Block login until email is verified
        return jsonify({'error': 'Please verify your email address before logging in. You can request a new verification email.'}), 403 # Forbidden

    # Login successful: Clear old session and set new session data
    session.clear()
    session['user_id'] = user['id']
    session['email'] = user['email']
    session['logged_in_at'] = datetime.datetime.now().isoformat()
    session.permanent = True # Make session permanent (respects Flask's PERMANENT_SESSION_LIFETIME config)

    try:
        # Update last login timestamp
        db.execute('UPDATE users SET last_login = ? WHERE id = ?', (datetime.datetime.now(), user['id']))
        db.commit()
    except Exception as e:
        app.logger.error(f"Failed to update last_login for user {user['id']}: {e}")
        # Non-critical error, login can proceed

    app.logger.info(f"User {user['email']} (ID: {user['id']}) logged in successfully.")
    return jsonify({'message': 'Login successful.', 'user_id': user['id'], 'email': user['email']}), 200


@auth_bp.route('/logout', methods=['POST'])
def logout():
    """Handles user logout."""
    if 'user_id' not in session:
        return jsonify({'message': 'Not currently logged in.'}), 200 # Informative message

    user_email = session.get('email', 'Unknown user')
    session.clear() # Clear all session data
    app.logger.info(f"User {user_email} logged out.")
    return jsonify({'message': 'Logout successful.'}), 200

@auth_bp.route('/request_password_reset', methods=['POST'])
@limiter.limit("3/hour") # Limit password reset requests per hour per IP
def request_password_reset():
    """Handles requesting a password reset link."""
    data = request.get_json()
    if not data or 'email' not in data:
        return jsonify({'error': 'Email is required.'}), 400
    email = data['email']

    db = get_db()
    # Find user by email, only if email is verified
    user = db.execute('SELECT * FROM users WHERE email = ? AND email_verified = ?', (email, True)).fetchone()
    if not user:
        # Always return a generic success message to avoid revealing if the email exists or is verified
        app.logger.info(f"Password reset request for non-existent or unverified email: {email}")
        return jsonify({'message': 'If an account with that email exists and is verified, a password reset link has been sent.'}), 200

    # Generate a password reset token
    token = generate_password_reset_token(user['id'])
    # Generate the URL for the reset password page (should be a GET endpoint)
    reset_url = url_for('auth_bp.reset_password_with_token', token=token, _external=True)

    # Store the token in the DB (optional but recommended for invalidation)
    expires_at = datetime.datetime.now() + datetime.timedelta(minutes=30) # Token valid for 30 minutes
    try:
        db.execute('INSERT INTO password_reset_tokens (user_id, token, expires_at) VALUES (?, ?, ?)',
                   (user['id'], token, expires_at))
        db.commit()
    except Exception as e:
        app.logger.error(f"Failed to store password reset token for user {user['id']}: {e}", exc_info=True)
        db.rollback()
        return jsonify({'error': 'Failed to initiate password reset due to a server error.'}), 500

    # Send the password reset email
    html_body = render_template_string(
        "<p>You requested a password reset. Please follow this link to reset your password:</p>"
        "<p><a href='{{ reset_url }}'>{{ reset_url }}</a></p>"
        "<p>This link will expire in 30 minutes.</p>"
        "<p>If you did not request this, please ignore this email.</p>", reset_url=reset_url)

    if send_email(email, 'Reset Your Password - AI Learning Companion', html_body):
        app.logger.info(f"Sent password reset email to {email} (user ID: {user['id']}).")
        # Return generic success message
        return jsonify({'message': 'If an account with that email exists and is verified, a password reset link has been sent.'}), 200
    else:
        # Log email failure but return generic success message to user
        app.logger.error(f"Failed to send password reset email to {email} (user ID: {user['id']}).")
        return jsonify({'message': 'If an account with that email exists and is verified, a password reset link has been sent.'}), 200


@auth_bp.route('/reset_password/<token>', methods=['GET', 'POST']) # --- MODIFIED: Added GET method ---
@limiter.limit("5/hour") # Limit attempts with a token per hour per IP
def reset_password_with_token(token):
    """Handles displaying the reset form (GET) and processing the password reset (POST)."""

    # Validate the token first, regardless of method
    user_id = confirm_password_reset_token(token)

    if user_id is None:
        # Token is invalid or expired
        return jsonify({'error': 'The password reset link is invalid or has expired.'}), 400

    db = get_db()
    # Check if token exists in DB, is for the correct user, is not used, and is not expired
    token_record = db.execute(
        'SELECT * FROM password_reset_tokens WHERE token = ? AND user_id = ? AND used = ? AND expires_at > ?',
        (token, user_id, False, datetime.datetime.now())
    ).fetchone()

    if not token_record:
        app.logger.warning(f"Attempt to use invalid, expired, or already used password reset token: {token} for user_id: {user_id}")
        return jsonify({'error': 'The password reset link is invalid, has expired, or has already been used.'}), 400

    # --- Handle GET Request ---
    if request.method == 'GET':
        # Serve an HTML page with a form to enter the new password
        # The form should POST back to this same URL (/auth/reset_password/<token>)
        reset_form_html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Reset Password</title>
            <style>
                body { font-family: sans-serif; display: flex; justify-content: center; align-items: center; min-height: 80vh; background-color: #f4f4f4; }
                .container { background: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1); text-align: center; }
                h2 { margin-bottom: 20px; color: #333; }
                .form-group { margin-bottom: 15px; text-align: left; }
                label { display: block; margin-bottom: 5px; font-weight: bold; color: #555; }
                input[type="password"] { width: calc(100% - 22px); padding: 10px; border: 1px solid #ccc; border-radius: 4px; font-size: 1em; }
                button { background-color: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; font-size: 1em; transition: background-color 0.3s ease; }
                button:hover { background-color: #0056b3; }
                .message { margin-top: 20px; color: green; }
                .error { margin-top: 20px; color: red; }
            </style>
        </head>
        <body>
            <div class="container">
                <h2>Reset Your Password</h2>
                <form method="POST" action="{{ url_for('auth_bp.reset_password_with_token', token=token) }}">
                    <div class="form-group">
                        <label for="new_password">New Password:</label>
                        <input type="password" id="new_password" name="new_password" required minlength="8">
                    </div>
                    <div class="form-group">
                        <label for="confirm_password">Confirm Password:</label>
                        <input type="password" id="confirm_password" name="confirm_password" required minlength="8">
                    </div>
                    <button type="submit">Reset Password</button>
                </form>
                 {% with messages = get_flashed_messages(with_categories=true) %}
                    {% if messages %}
                        {% for category, message in messages %}
                            <div class="{{ category }}">{{ message }}</div>
                        {% endfor %}
                    {% endif %}
                {% endwith %}
            </div>
             <script>
                // Basic client-side password confirmation check
                document.querySelector('form').addEventListener('submit', function(event) {
                    const password = document.getElementById('new_password').value;
                    const confirm_password = document.getElementById('confirm_password').value;
                    if (password !== confirm_password) {
                        alert('Passwords do not match!'); // Use alert for simplicity in this basic example
                        event.preventDefault(); // Prevent form submission
                    }
                });
            </script>
        </body>
        </html>
        """
        # Render the HTML template, passing the token to the form action
        return render_template_string(reset_form_html, token=token)

    # --- Handle POST Request ---
    elif request.method == 'POST':
        # For a POST request coming from the HTML form, get data from request.form
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password') # Get confirm password from form

        if not new_password or not confirm_password:
             # If using JSON POST from an API client, check request.get_json() as fallback
             data = request.get_json()
             if data:
                 new_password = data.get('new_password')
                 confirm_password = data.get('confirm_password')

        if not new_password or not confirm_password:
            flash('New password and confirmation are required.', 'error')
            # Redirect back to the GET endpoint to show the form with the error
            return redirect(url_for('auth_bp.reset_password_with_token', token=token))

        if new_password != confirm_password:
            flash('Passwords do not match.', 'error')
            return redirect(url_for('auth_bp.reset_password_with_token', token=token))

        if len(new_password) < 8:
            flash('Password must be at least 8 characters long.', 'error')
            return redirect(url_for('auth_bp.reset_password_with_token', token=token))


        # If token is valid and not used, proceed with password reset
        new_hashed_password = generate_password_hash(new_password + app.config['SECURITY_PASSWORD_SALT'])
        try:
            # Update the user's password
            db.execute('UPDATE users SET password_hash = ? WHERE id = ?', (new_hashed_password, user_id))
            # Mark the token as used to prevent reuse
            db.execute('UPDATE password_reset_tokens SET used = ? WHERE id = ?', (True, token_record['id']))
            db.commit()

            # Optionally, log the user out of all other sessions if you store session tokens in DB
            # session.clear() # Log out current session if any (might not be logged in)

            # Fetch user email to send confirmation
            user_email_row = db.execute('SELECT email FROM users WHERE id = ?', (user_id,)).fetchone()
            if user_email_row:
                user_email = user_email_row['email']
                # Send password reset confirmation email
                html_body = render_template_string("<p>Your password has been successfully reset.</p><p>If you did not perform this action, please contact support immediately.</p>")
                send_email(user_email, 'Password Reset Confirmation - AI Learning Companion', html_body)
                app.logger.info(f"Password successfully reset for user ID: {user_id} ({user_email}). Token marked as used.")
            else:
                 app.logger.error(f"Password reset successful for user ID: {user_id}, but could not find email to send confirmation.")


            # Redirect to a success page or login page
            # You might want a dedicated success page template
            success_html = """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Password Reset Successful</title>
                <style>
                    body { font-family: sans-serif; display: flex; justify-content: center; align-items: center; min-height: 80vh; background-color: #f4f4f4; }
                    .container { background: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1); text-align: center; }
                    h2 { margin-bottom: 20px; color: green; }
                    p { margin-bottom: 20px; color: #333; }
                    a { color: #007bff; text-decoration: none; }
                    a:hover { text-decoration: underline; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h2>Password Reset Successful!</h2>
                    <p>Your password has been updated. You can now log in with your new password.</p>
                    <p><a href="{{ url_for('login_html') }}">Go to Login Page</a></p>
                </div>
            </body>
            </html>
            """
            return render_template_string(success_html)

        except Exception as e:
            app.logger.error(f"Error resetting password for user_id {user_id}: {e}", exc_info=True)
            db.rollback()
            flash('Failed to reset password due to a server error.', 'error')
            # Redirect back to the GET endpoint to show the form with the error
            return redirect(url_for('auth_bp.reset_password_with_token', token=token))

# Register the auth blueprint with the Flask app
app.register_blueprint(auth_bp)
# --- END USER AUTH ---


# --- Existing API Endpoints (ensure they are protected if necessary) ---
# Example: you might want to add @login_required decorator to some routes
# from functools import wraps
# def login_required(f):
# @wraps(f)
# def decorated_function(*args, **kwargs):
# if 'user_id' not in session:
# return jsonify({'error': 'Authentication required'}), 401
# return f(*args, **kwargs)
# return decorated_function

# Then for an endpoint:
# @app.route(f'{BASE_URL}/some_protected_resource', methods=['GET'])
# @login_required
# def some_protected_resource():
#    pass
# ---


@app.route(f'{BASE_URL}/pdf', methods=['POST'])
@limiter.limit("10/minute") # Limit PDF uploads per minute per IP
# @login_required # --- USER AUTH --- Consider adding if PDF upload needs auth
# @PDF_UPLOADS_COUNTER # Moved counter increment inside the route function if needed
def upload_pdf_endpoint():
    """Handles PDF file uploads."""
    # Increment the counter here if metrics are initialized
    # if 'metrics' in app.extensions and hasattr(app.extensions['metrics'], 'pdf_uploads_total'):
    #     app.extensions['metrics'].pdf_uploads_total.labels(status=request.status_code).inc()

    if 'pdf_file' not in request.files:
        return jsonify({'error': 'No PDF file part in the request'}), 400
    file = request.files['pdf_file']
    if file.filename == '':
        return jsonify({'error': 'No PDF file selected for upload'}), 400

    # Check file extension and size
    if file and file.filename.lower().endswith('.pdf'):
        from werkzeug.utils import secure_filename
        original_filename = secure_filename(file.filename)

        file.seek(0, os.SEEK_END)
        file_length = file.tell()
        if file_length > app.config['MAX_PDF_SIZE']:
            return jsonify({'error': f'PDF size ({file_length / (1024*1024):.2f}MB) exceeds the limit of {app.config["MAX_PDF_SIZE"] / (1024 * 1024)} MB'}), 413 # Payload Too Large
        file.seek(0) # Reset file pointer to the beginning

        # Save the file temporarily
        temp_pdf_dir = app.config['TEMP_PDF_FOLDER']
        os.makedirs(temp_pdf_dir, exist_ok=True)
        # --- USER AUTH --- Use user_id in filename if available for namespacing
        session_or_user_id = str(session.get('user_id', session.sid))
        # Ensure filename is unique and includes identifier
        temp_filepath = os.path.join(temp_pdf_dir, f"{session_or_user_id}_{int(time.time())}_{original_filename}")
        # --- END USER AUTH ---

        try:
            file.save(temp_filepath)
            app.logger.info(f"Temporary PDF file saved to: {temp_filepath}")

            # Dispatch background task for PDF processing
            task = process_pdf_async.delay(temp_filepath, original_filename)
            app.logger.info(f"Dispatched PDF processing task {task.id} for file {original_filename}")

            # Store task ID and status in session
            session['pdf_process_task_id'] = task.id
            session['pdf_original_filename'] = original_filename
            session['pdf_processing_status'] = 'PENDING'
            session.modified = True

            return jsonify({
                'message': 'PDF upload successful. Processing in background.',
                'task_id': task.id,
                'filename': original_filename,
                'status': 'pending'
            }), 202 # Accepted
        except Exception as e:
            app.logger.error(f"Error during PDF upload or task dispatch for {original_filename}: {e}", exc_info=True)
            # Clean up the temporary file if an error occurred after saving
            if os.path.exists(temp_filepath):
                try: os.remove(temp_filepath)
                except OSError: app.logger.error(f"Could not clean up {temp_filepath} after error.")
            session['pdf_processing_status'] = 'FAILED'
            session.modified = True
            return jsonify({'error': f'Error handling PDF upload: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Invalid file type. Only PDF files are allowed.'}), 400

@app.route(f'{BASE_URL}/pdf_status/<task_id>', methods=['GET'])
# @login_required # --- USER AUTH ---
def pdf_status_endpoint(task_id):
    """Checks the status of a PDF processing task."""
    # --- USER AUTH --- Add check: ensure this task_id belongs to the current user if PDFs are user-specific
    # This would require storing user_id with the task or PDF data. For now, it's open.
    # --- END USER AUTH ---
    task = process_pdf_async.AsyncResult(task_id, app=celery_app)
    response_data = {
        'task_id': task_id,
        'status': task.state.lower(), # Celery state (PENDING, STARTED, SUCCESS, FAILURE)
        'timestamp': datetime.datetime.now().isoformat()
    }

    if task.state == 'PENDING' or task.state == 'STARTED':
        response_data['message'] = f'PDF processing is {task.state.lower()}.'
        session['pdf_processing_status'] = task.state.lower()
        session.modified = True
        return jsonify(response_data), 202 # Accepted (processing)

    elif task.state == 'SUCCESS':
        result = task.result # Get the return value of the task
        if result and result.get('status') == 'completed':
            # Store processed PDF data in session
            session['pdf_data'] = {
                'original_filename': result.get('original_filename'),
                'total_pages': result.get('total_pages'),
                'texts': result.get('texts', []),
                'images_b64_by_page': result.get('images', []),
                'temp_filepath': result.get('temp_filepath') # Store temp path for rendering
                # --- USER AUTH --- Consider adding user_id here if relevant
            }
            session['pdf_page_count'] = result.get('total_pages', 0)
            session['current_page_num'] = 0 # Reset to first page on successful load
            session['pdf_processing_status'] = 'completed'
            session.modified = True

            response_data.update({
                'message': 'PDF processed successfully',
                'filename': result.get('original_filename'),
                'total_pages': result.get('total_pages'),
                'current_page_num_display': 1, # Display page 1
                'status': 'completed'
            })
            return jsonify(response_data), 200 # OK
        else:
             # Task completed but reported an internal failure
             error_msg = result.get('error', 'Unknown processing error reported by task.')
             app.logger.error(f"PDF processing task {task_id} completed with internal failure: {error_msg}")
             session['pdf_processing_status'] = 'failed'
             session.modified = True
             response_data.update({
                 'status': 'failed',
                 'error': f'PDF processing failed: {error_msg}'
             })
             return jsonify(response_data), 500 # Internal Server Error


    elif task.state == 'FAILURE':
        # Task failed during execution
        error = str(task.info) if isinstance(task.info, Exception) else str(task.result)
        app.logger.error(f"PDF processing task {task_id} failed with Celery state FAILURE: {error}")
        session['pdf_processing_status'] = 'failed'
        session.modified = True
        response_data.update({
            'status': 'failed',
            'error': f'PDF processing task failed: {error}'
        })
        return jsonify(response_data), 500 # Internal Server Error
    else:
        # Unknown task state
        response_data['message'] = f'Unknown task state: {task.state}.'
        session['pdf_processing_status'] = 'unknown'
        session.modified = True
        return jsonify(response_data), 500 # Internal Server Error


@app.route(f'{BASE_URL}/page_content/<int:page_number_display>', methods=['GET'])
# @login_required # --- USER AUTH ---
def get_page_content_endpoint(page_number_display):
    """Retrieves text and image count for a specific PDF page."""
    pdf_data = session.get('pdf_data')
    # --- USER AUTH --- If pdf_data is user-specific, ensure current user owns this pdf_data
    # --- END USER AUTH ---
    if not pdf_data or 'texts' not in pdf_data or 'total_pages' not in pdf_data:
        return jsonify({'error': 'No PDF loaded or PDF data not found in session'}), 400 # Bad Request

    pdf_texts_list = pdf_data['texts']
    total_pages = pdf_data['total_pages']

    # Convert 1-based display number to 0-based index
    page_index = page_number_display - 1
    if not (0 <= page_index < total_pages):
        return jsonify({'error': f'Invalid page number. Must be between 1 and {total_pages}'}), 400 # Bad Request

    page_text = pdf_texts_list[page_index]

    pdf_images_list = pdf_data.get('images_b64_by_page', [])
    page_images_count = len(pdf_images_list[page_index] if 0 <= page_index < len(pdf_images_list) else [])

    # Update current page in session
    session['current_page_num'] = page_index
    session.modified = True

    return jsonify({
        'page_text': page_text,
        'page_images_count': page_images_count,
        'current_page_num_display': page_number_display,
        'total_pages': total_pages,
        'pdf_filename': pdf_data.get('original_filename')
    }), 200 # OK


@app.route(f'{BASE_URL}/navigate_pdf', methods=['POST'])
# @login_required # --- USER AUTH ---
def navigate_pdf_endpoint():
    """Navigates to a different page in the loaded PDF."""
    pdf_data = session.get('pdf_data')
    # --- USER AUTH --- Ownership check for pdf_data
    # --- END USER AUTH ---
    if not pdf_data or 'total_pages' not in pdf_data:
        return jsonify({'error': 'No PDF loaded in session'}), 400 # Bad Request

    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid request: Missing JSON data'}), 400 # Bad Request

    current_page_index = session.get('current_page_num', 0)
    total_pages = pdf_data['total_pages']
    target_page_index = current_page_index # Default to current page

    direction = data.get('direction')
    page_num_display_input = data.get('page_num_display')

    # Navigate by specific page number
    if page_num_display_input is not None:
        try:
            target_page_display = int(page_num_display_input)
            if 1 <= target_page_display <= total_pages:
                target_page_index = target_page_display - 1 # Convert to 0-based index
            else:
                return jsonify({'error': f'Invalid page number. Must be between 1 and {total_pages}'}), 400 # Bad Request
        except ValueError:
            return jsonify({'error': 'Invalid page number format. Must be an integer.'}), 400 # Bad Request
    # Navigate by direction (prev/next)
    elif direction == 'prev':
        if current_page_index > 0:
            target_page_index = current_page_index - 1
    elif direction == 'next':
        if current_page_index < total_pages - 1:
            target_page_index = current_page_index + 1
    else:
        return jsonify({'error': 'Invalid navigation parameters. Provide "direction" ("prev" or "next") or "page_num_display".'}), 400 # Bad Request

    # Ensure target page index is within bounds
    target_page_index = max(0, min(target_page_index, total_pages - 1))

    # Update current page in session
    session['current_page_num'] = target_page_index
    session.modified = True

    # Get content for the new current page
    pdf_texts_list = pdf_data.get('texts', [])
    page_text = pdf_texts_list[target_page_index] if 0 <= target_page_index < len(pdf_texts_list) else "[Error fetching text]"

    pdf_images_list = pdf_data.get('images_b64_by_page', [])
    page_images_count = len(pdf_images_list[target_page_index] if 0 <= target_page_index < len(pdf_images_list) else [])


    return jsonify({
        'current_page_num_display': target_page_index + 1, # Return 1-based display number
        'total_pages': total_pages,
        'current_page_text': page_text,
        'current_page_images_count': page_images_count,
        'pdf_filename': pdf_data.get('original_filename')
    }), 200 # OK


@app.route(f'{BASE_URL}/zoom', methods=['POST'])
# @login_required # --- USER AUTH ---
def set_zoom_level_endpoint():
    """Sets the PDF rendering zoom level in the session."""
    data = request.get_json()
    if not data or 'zoom_level' not in data:
        return jsonify({'error': 'Missing zoom_level parameter in JSON body'}), 400 # Bad Request
    try:
        zoom_level = float(data['zoom_level'])
        if not (0.2 <= zoom_level <= 5.0): # Define a reasonable range
            return jsonify({'error': 'Zoom level out of range. Must be between 0.2 and 5.0'}), 400 # Bad Request
        session['zoom_level'] = zoom_level
        session.modified = True
        # Clear the rendered page cache when zoom changes, as cached images are at the old scale
        session.pop('rendered_page_cache', None)
        app.logger.info(f"Zoom level set to: {zoom_level}")
        return jsonify({'status': 'Zoom level updated', 'zoom_level': zoom_level}), 200 # OK
    except ValueError:
        return jsonify({'error': 'Invalid zoom level format. Must be a float.'}), 400 # Bad Request

@app.route(f'{BASE_URL}/zoom', methods=['GET'])
# @login_required # --- USER AUTH ---
def get_zoom_level_endpoint():
    """Gets the current PDF rendering zoom level from the session."""
    zoom_level = session.get('zoom_level', 1.0) # Default zoom is 1.0
    return jsonify({'zoom_level': zoom_level}), 200 # OK


@app.route(f'{BASE_URL}/render_page/<int:page_number_display>', methods=['GET'])
# @login_required # --- USER AUTH ---
def render_page_endpoint(page_number_display):
    """Renders a specific PDF page as an image and returns it as base64."""
    pdf_data = session.get('pdf_data')
    # --- USER AUTH --- Ownership check for pdf_data
    # --- END USER AUTH ---
    if not pdf_data or 'temp_filepath' not in pdf_data or 'total_pages' not in pdf_data:
        return jsonify({'error': 'No PDF loaded or PDF data not found in session.'}), 400 # Bad Request

    pdf_filepath = pdf_data['temp_filepath']
    total_pages = pdf_data['total_pages']
    page_index = page_number_display - 1 # Convert to 0-based index

    # Check if the temporary file still exists
    if not os.path.exists(pdf_filepath):
         app.logger.error(f"Temporary PDF file not found at expected path for rendering: {pdf_filepath}")
         return jsonify({'error': 'Temporary PDF file not found on the server for rendering. Please re-upload.'}), 404 # Not Found

    # Check if PyMuPDF is available
    if not pdf_available:
         return jsonify({'error': 'PDF rendering library (PyMuPDF) is not installed on the server.'}), 501 # Not Implemented


    zoom_level = session.get('zoom_level', 1.0)
    render_scale = app.config['PAGE_RENDER_SCALE'] * zoom_level # Apply configured scale and user zoom

    if not (0 <= page_index < total_pages):
        return jsonify({'error': f'Page number {page_number_display} out of bounds. Total pages: {total_pages}.'}), 400 # Bad Request

    # Initialize rendered page cache if it doesn't exist
    if 'rendered_page_cache' not in session:
        session['rendered_page_cache'] = {}

    # --- USER AUTH --- Cache key should include user_id if content is user-specific
    session_or_user_id_cache = str(session.get('user_id', session.sid))
    # Create a unique cache key based on user/session, page, scale, and filename
    cache_key = f"user_{session_or_user_id_cache}_page_{page_index}_scale_{render_scale:.2f}_file_{os.path.basename(pdf_filepath)}"
    # --- END USER AUTH ---

    # Check if the rendered page is in the cache
    if cache_key in session['rendered_page_cache']:
        app.logger.info(f"Serving cached rendered page: {cache_key}")
        return jsonify({'image_data_b64': session['rendered_page_cache'][cache_key], 'rendered_with_zoom': True}), 200 # OK

    try:
        # Open the PDF and render the page
        doc = fitz.open(pdf_filepath)
        page = doc.load_page(page_index)
        # Create a transformation matrix for scaling
        matrix = fitz.Matrix(render_scale, render_scale)
        # Get the pixmap (image representation) of the page
        pix = page.get_pixmap(matrix=matrix, alpha=False) # alpha=False for opaque background
        # Convert pixmap to PNG bytes
        img_bytes = pix.tobytes("png")
        # Encode PNG bytes to base64 string
        image_data_b64 = base64.b64encode(img_bytes).decode('utf-8')
        doc.close() # Close the document

        # Store the rendered image in the session cache
        session['rendered_page_cache'][cache_key] = image_data_b64
        # Manage cache size
        if len(session['rendered_page_cache']) > app.config['MAX_RENDER_CACHE_SIZE']:
            # Remove the oldest item if cache exceeds limit
            oldest_key = next(iter(session['rendered_page_cache']))
            session['rendered_page_cache'].pop(oldest_key)
            app.logger.info(f"Evicted oldest rendered page from cache: {oldest_key}")

        session.modified = True # Mark session as modified

        app.logger.info(f"Rendered and cached page: {cache_key}")
        return jsonify({'image_data_b64': image_data_b64, 'rendered_with_zoom': True}), 200 # OK
    except Exception as e:
        app.logger.error(f"Error rendering page {page_number_display} (index {page_index}) of PDF from {pdf_filepath}: {e}", exc_info=True)
        return jsonify({'error': f'Error rendering page: {str(e)}'}), 500 # Internal Server Error

def prepare_ai_prompt_and_context(user_request_text, include_page_context=True, max_page_context_len=3000, max_history_turns=5):
    """Prepares the full prompt for the AI model, including system role, history, and PDF context."""
    full_prompt_parts = []

    # Add the selected personality's system prompt
    current_personality_name = session.get('selected_personality', "Default Tutor")
    selected_personality_details = PERSONALITIES.get(current_personality_name, PERSONALITIES["Default Tutor"])
    full_prompt_parts.append(f"System Role: {selected_personality_details['system_prompt']}\n")

    # Include recent conversation history
    ai_conversation_history = session.get('ai_conversation_history', [])
    if ai_conversation_history:
        history_str = "Previous conversation turns (User/Assistant):\n"
        # Include the last N turns (N * 2 messages, User and Assistant)
        start_index = max(0, len(ai_conversation_history) - (max_history_turns * 2))
        for entry in ai_conversation_history[start_index:]:
            role = entry.get('role', 'unknown').capitalize()
            content = entry.get('content', '')
            if content.strip():
                # Truncate long history messages to save context space
                truncated_content = content.strip()
                if len(truncated_content) > 300:
                    truncated_content = truncated_content[:300] + "..."
                history_str += f"{role}: {truncated_content}\n"
        # Only add history if there's actual content after truncation
        if history_str != "Previous conversation turns (User/Assistant):\n":
            full_prompt_parts.append(history_str)

    # Include current PDF page context if requested and available
    pdf_data = session.get('pdf_data')
    # --- USER AUTH --- Ownership check for pdf_data if applicable
    # --- END USER AUTH ---
    if include_page_context and pdf_data and 'texts' in pdf_data and 'total_pages' in pdf_data and pdf_data['total_pages'] > 0:
        pdf_texts_list = pdf_data['texts']
        current_page_idx = session.get('current_page_num', 0)
        if 0 <= current_page_idx < len(pdf_texts_list):
            page_text = pdf_texts_list[current_page_idx]
            truncated_page_text = page_text.strip()
            # Truncate page text if it's too long
            if len(truncated_page_text) > max_page_context_len:
                 truncated_page_text = truncated_page_text[:max_page_context_len] + "..."

            # Add page context unless it's just the "[No text found]" placeholder
            if truncated_page_text and not truncated_page_text.startswith(("[No text found", "[Error extracting")):
                full_prompt_parts.append(f"Current PDF Page ({current_page_idx + 1}) Context:\n\"\"\"\n{truncated_page_text}\n\"\"\"\n")
            elif truncated_page_text:
                 # Include placeholder text if that's all that was found
                 full_prompt_parts.append(f"Current PDF Page ({current_page_idx + 1}) Context: {truncated_page_text}\n")


    # Add the user's current request/instruction
    full_prompt_parts.append(f"User's Request: {user_request_text.strip()}\n\nAI Response:")
    final_prompt = "\n".join(full_prompt_parts)
    app.logger.debug(f"Prepared AI Prompt: {final_prompt[:500]}...") # Log snippet of the prompt
    return final_prompt


@app.route(f'{BASE_URL}/ask_ai', methods=['POST'])
@limiter.limit("60/minute") # Limit AI requests per minute per IP
# @login_required # --- USER AUTH --- Consider adding if AI access needs auth
def ask_ai_endpoint():
    """Handles user requests to the AI model."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Missing JSON request body'}), 400 # Bad Request

    user_input_text = data.get('input_text', '').strip()
    action = data.get('action', 'ask') # Default action is 'ask'
    selected_text_for_action = data.get('selected_text', '').strip() # Text selected by the user

    # Get the current selected AI model
    current_model_name = session.get('current_ollama_model')
    # Check if a valid model is selected
    if not current_model_name or current_model_name in ["Loading...", "No Models Found", "Ollama Offline", "Ollama Timeout", "Error Fetching Models", "Error Decoding Response"]:
        return jsonify({'error': 'AI model not available or not selected. Please check settings.'}), 400 # Bad Request

    # Validate input based on action
    if action == 'ask' and not user_input_text and not selected_text_for_action:
         # Allow empty input for 'ask' if selected_text is provided (e.g., "Explain this.")
         pass
    elif action == 'ask' and not user_input_text: # If action is 'ask' but no text input and no selected text
        return jsonify({'error': 'Input text cannot be empty for a general question without selected text.'}), 400 # Bad Request


    instruction_for_ai = user_input_text # The primary instruction for the AI
    include_pdf_context_in_prompt = True # Whether to include the current PDF page text in the prompt
    images_for_ai_payload = None # List of base64 images to send to a vision model
    request_label_for_logging = "AI Query" # Label for logging purposes

    # Get current PDF state from session
    current_page_idx = session.get('current_page_num', 0)
    current_page_display = current_page_idx + 1 # 1-based page number for display

    pdf_data = session.get('pdf_data')
    # --- USER AUTH --- Ownership check for pdf_data
    # --- END USER AUTH ---
    pdf_texts_list = pdf_data.get('texts', []) if pdf_data else []
    pdf_images_by_page_list = pdf_data.get('images_b64_by_page', []) if pdf_data else []

    # --- Handle different AI actions ---
    if action == 'explain_concept':
        if not selected_text_for_action:
            return jsonify({'error': 'No text selected for "explain_concept" action.'}), 400 # Bad Request
        request_label_for_logging = "Explain Concept"
        # Add user query to chat history
        add_to_chat_history('User', f"Explain concept (from page {current_page_display}): \"{selected_text_for_action[:70]}...\"", role_for_ai="user")
        # Craft specific instruction for the AI
        instruction_for_ai = (
            f"Provide a comprehensive explanation of the following concept, which is selected from the current document page. "
            f"Explain it clearly, define key terms, describe its function or importance, and provide examples if applicable.\n"
            f"Concept Text:\n\"\"\"\n{selected_text_for_action}\n\"\"\"\n"
            f"Format your response clearly using Markdown."
        )
        include_pdf_context_in_prompt = True # Include full page context might be helpful here
    elif action == 'explain_code':
        if not selected_text_for_action:
            return jsonify({'error': 'No code snippet selected for "explain_code" action.'}), 400 # Bad Request
        request_label_for_logging = "Explain Code"
        add_to_chat_history('User', f"Explain code snippet (from page {current_page_display}): ```\n{selected_text_for_action[:100]}...\n```", role_for_ai="user")
        instruction_for_ai = (
            f"Analyze the following code snippet from the current document page. Explain its purpose, functionality, language (if identifiable), "
            f"and key components (variables, functions, logic flow).\n"
            f"Code Snippet:\n```\n{selected_text_for_action}\n```\nDetailed Explanation:"
        )
        include_pdf_context_in_prompt = False # Don't need the full page text, just the selected code
    elif action in ['summarize_page', 'generate_quiz', 'key_points']:
        # These actions require the full text of the current page
        if not pdf_data or not pdf_texts_list or not (0 <= current_page_idx < len(pdf_texts_list)):
            return jsonify({'error': f'No PDF page content available for action: {action}.'}), 400 # Bad Request
        page_text_for_action = pdf_texts_list[current_page_idx]
        if len(page_text_for_action.strip()) < 50: # Basic check for minimal content
            return jsonify({'error': f'Not enough text on page {current_page_display} to perform action: {action}.'}), 400 # Bad Request

        if action == 'summarize_page':
            request_label_for_logging = "Summarize Page"
            add_to_chat_history('User', f"Summarize page {current_page_display}", role_for_ai="user")
            instruction_for_ai = (
                f"Analyze the text from page {current_page_display} of the document. Provide a concise yet comprehensive summary. "
                f"Identify the main topic(s), key arguments or findings, and the overall conclusion or takeaway.\n"
                f"Text to Analyze:\n\"\"\"\n{page_text_for_action[:4000]}\n\"\"\"\n" # Limit text length for prompt
            )
        elif action == 'generate_quiz':
            request_label_for_logging = "Generate Quiz"
            add_to_chat_history('User', f"Generate a quiz for page {current_page_display}", role_for_ai="user")
            instruction_for_ai = (
                f"Based on the content of page {current_page_display}, create a quiz with 3-5 questions. "
                f"Include a mix of question types (e.g., multiple-choice, true/false, short answer). Provide clear answers for each question.\n"
                f"Text for Quiz Generation:\n\"\"\"\n{page_text_for_action[:4000]}\n\"\"\"\n" # Limit text length
            )
        elif action == 'key_points':
            request_label_for_logging = "Key Points"
            add_to_chat_history('User', f"Extract and list the most important key points from page {current_page_display}", role_for_ai="user")
            instruction_for_ai = (
                f"From the text of page {current_page_display}, extract and list the most important key points, definitions, or facts. "
                f"Present them in a clear, bulleted list or a structured format.\n"
                f"Text for Key Point Extraction:\n\"\"\"\n{page_text_for_action[:4000]}\n\"\"\"\n" # Limit text length
            )
        include_pdf_context_in_prompt = False # The full page text is provided explicitly in the instruction

    elif action == 'analyze_images':
        # This action requires images from the current page
        if not pdf_data or not pdf_images_by_page_list or not (0 <= current_page_idx < len(pdf_images_by_page_list)):
            return jsonify({'error': 'No PDF images available for analysis on the current page.'}), 400 # Bad Request
        images_on_current_page_b64 = pdf_images_by_page_list[current_page_idx]
        if not images_on_current_page_b64:
            return jsonify({'error': f'No images found on page {current_page_display} to analyze.'}), 400 # Bad Request

        # Check if the selected model has vision capabilities
        model_caps = get_model_capabilities_from_ollama(current_model_name)
        if not model_caps.get("vision"):
            return jsonify({'error': f"The selected AI model '{current_model_name}' does not support image analysis (vision). Please select a vision-capable model."}), 400 # Bad Request

        request_label_for_logging = "Analyze Images"
        num_images = len(images_on_current_page_b64)
        add_to_chat_history('User', f"Analyze {num_images} image(s) on page {current_page_display}", role_for_ai="user")
        instruction_for_ai = (
            f"You will be provided with {num_images} image(s) from page {current_page_display} of a document. "
            f"For each image, describe its visual content in detail. If it's a diagram, chart, or graph, explain its purpose, "
            f"the data it represents, and any key insights. If there's text in the image, transcribe and explain it. "
            f"Relate the image(s) to the textual context of the page if possible."
        )
        images_for_ai_payload = images_on_current_page_b64 # Add images to the payload
        include_pdf_context_in_prompt = True # Include text context along with images

    elif action == 'ask':
        # General question action
        request_label_for_logging = "General Question"
        if selected_text_for_action:
             # If text is selected, frame the question around it
             instruction_for_ai = f"Regarding the following text from page {current_page_display}:\n\"\"\"\n{selected_text_for_action}\n\"\"\"\n{user_input_text}"
             add_to_chat_history('User', f"Regarding selected text (from page {current_page_display}): \"{selected_text_for_action[:70]}...\" - {user_input_text}", role_for_ai="user")
        else:
             # If no text is selected, just use the user's input
             instruction_for_ai = user_input_text
             add_to_chat_history('User', user_input_text, role_for_ai="user")
        # include_pdf_context_in_prompt remains True by default

    else:
        return jsonify({'error': f'Unknown action specified: {action}'}), 400 # Bad Request

    # Prepare the final prompt for Ollama
    full_prompt_for_ollama = prepare_ai_prompt_and_context(
        instruction_for_ai,
        include_page_context=include_pdf_context_in_prompt
    )

    # Construct the payload for the Ollama API
    ollama_payload = {
        "model": current_model_name,
        "prompt": full_prompt_for_ollama,
        "stream": False, # Request non-streaming response for simplicity
        "options": {
            "temperature": session.get('ai_temperature', 0.6), # Get temperature from session
            "num_ctx": session.get('ai_context_window', 4096) # Get context window from session
        }
    }
    # Add images to the payload if available for vision models
    if images_for_ai_payload:
        ollama_payload["images"] = images_for_ai_payload

    ai_response_text = "Error: Could not get a response from the AI service." # Default error message
    try:
        app.logger.info(f"Sending '{request_label_for_logging}' request to Ollama model: {current_model_name}")
        ollama_api_url = urljoin(app.config['OLLAMA_URL'], "/api/generate")
        response = requests.post(ollama_api_url, json=ollama_payload, timeout=180) # Set a timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        response_data = response.json()
        ai_response_text = response_data.get('response', 'No content in AI response.').strip()
        session['last_ai_response'] = ai_response_text # Store last AI response in session
        add_to_chat_history('AI', ai_response_text, role_for_ai="assistant") # Add AI response to chat history
        session.modified = True # Mark session as modified

        app.logger.info(f"Received AI response for '{request_label_for_logging}'")
        return jsonify({
            'sender': 'AI',
            'message': ai_response_text,
            'tts_autoplay_triggered': session.get('auto_play_ai_tts', False) and bool(ai_response_text) # Indicate if TTS autoplay should trigger
        }), 200 # OK

    except requests.exceptions.Timeout:
        error_msg = f"The AI request ('{request_label_for_logging}') to model '{current_model_name}' timed out."
        app.logger.error(error_msg)
        ai_response_text = error_msg
    except requests.exceptions.HTTPError as http_err:
        error_msg = f"Ollama API request for '{request_label_for_logging}' failed with HTTP status {http_err.response.status_code}."
        try:
            # Try to get more specific error details from Ollama's response
            ollama_error_details = http_err.response.json().get('error', http_err.response.text)
            error_msg += f" Server said: {ollama_error_details}"
        except json.JSONDecodeError:
            error_msg += f" Server response: {http_err.response.response.text[:200]}..." # Fallback if response isn't JSON
        app.logger.error(error_msg, exc_info=True)
        ai_response_text = error_msg
    except requests.exceptions.RequestException as req_err:
        # Catch other request-related errors (connection errors, etc.)
        error_msg = f"A network or request error occurred while contacting Ollama for '{request_label_for_logging}': {req_err}"
        app.logger.error(error_msg, exc_info=True)
        ai_response_text = error_msg
    except Exception as e:
        # Catch any other unexpected errors
        error_msg = f"An unexpected error occurred during the AI request ('{request_label_for_logging}'): {e}"
        app.logger.error(error_msg, exc_info=True)
        ai_response_text = error_msg

    # Store the error message in session and chat history
    session['last_ai_response'] = ai_response_text
    add_to_chat_history('Error', ai_response_text, role_for_ai="assistant") # Log error as an assistant message
    session.modified = True
    return jsonify({'error': ai_response_text}), 500 # Internal Server Error


@app.route(f'{BASE_URL}/chat_history', methods=['GET'])
# @login_required # --- USER AUTH ---
def get_chat_history_endpoint():
    """Retrieves the current chat history from the database."""
    # --- USER AUTH --- Adapt to fetch user-specific history if identifier was changed in add_to_chat_history
    identifier = str(session.get('user_id', session.sid)) # Use user_id if logged in, else session.sid
    # --- END USER AUTH ---
    db = get_db()
    try:
        # Fetch messages ordered by timestamp
        messages = db.execute(
            "SELECT role, content FROM chat_messages WHERE session_id = ? ORDER BY timestamp ASC",
            (identifier,)
        ).fetchall()
        # Convert list of Row objects to list of dictionaries
        return jsonify([dict(row) for row in messages]), 200 # OK
    except Exception as e:
        app.logger.error(f"Error fetching chat history from DB for identifier {identifier}: {e}")
        return jsonify({'error': 'Failed to retrieve chat history.'}), 500 # Internal Server Error


@app.route(f'{BASE_URL}/clear_chat', methods=['DELETE'])
@limiter.limit("5/minute") # Limit chat clearing per minute per IP
# @login_required # --- USER AUTH ---
def clear_chat_endpoint():
    """Clears the chat history from the session and database."""
    # Clear session history
    session.pop('chat_history', None)
    session.pop('ai_conversation_history', None)
    session.pop('last_ai_response', None)
    session.modified = True
    app.logger.info("Chat history cleared from session.")

    # --- USER AUTH ---
    identifier = str(session.get('user_id', session.sid)) # Use user_id if logged in, else session.sid
    # --- END USER AUTH ---
    db = get_db()
    try:
        # Delete chat messages from the database for the current identifier
        db.execute("DELETE FROM chat_messages WHERE session_id = ?", (identifier,))
        db.commit()
        app.logger.info(f"Chat history cleared from DB for identifier {identifier}.")
    except Exception as e:
        app.logger.error(f"Error clearing chat history from DB for identifier {identifier}: {e}")
        # Continue even if DB clear fails, session is cleared
        return jsonify({'status': 'warning', 'message': 'Chat history cleared from session, but failed to clear from database.'}), 500 # Internal Server Error

    return jsonify({'status': 'success', 'message': 'Chat history cleared successfully'}), 200 # OK

@app.route(f'{BASE_URL}/clear_cache', methods=['POST'])
@limiter.limit("5/minute") # Limit cache clearing per minute per IP
# @login_required # --- USER AUTH --- Consider if this needs auth, affects current user's session
def clear_cache_endpoint():
    """Clears user-specific session cache, including PDF data and rendered pages."""
    # --- USER AUTH --- Note: This clears session for current user/browser.
    # If user is logged in, it will effectively log them out of this session.
    user_email_before_clear = session.get('email', 'anonymous user')
    # --- END USER AUTH ---

    # Clear PDF-related session data
    session.pop('pdf_data', None)
    session.pop('pdf_process_task_id', None)
    session.pop('pdf_original_filename', None)
    session.pop('pdf_processing_status', None)
    session.pop('pdf_page_count', None)
    session.pop('current_page_num', None)
    session.pop('rendered_page_cache', None)

    # Clear chat-related session data (DB history is kept unless clear_chat is called)
    session.pop('chat_history', None)
    session.pop('ai_conversation_history', None)
    session.pop('last_ai_response', None)

    # --- USER AUTH --- Clear user-specific session keys (effectively logs out)
    session.pop('user_id', None)
    session.pop('email', None)
    session.pop('logged_in_at', None)
    # --- END USER AUTH ---

    # Reset other settings to defaults in the session
    # These will be re-initialized on the next request by initialize_session_values
    # but clearing them explicitly here ensures they are gone immediately.
    # session.pop('selected_personality', None)
    # session.pop('selected_voice', None)
    # session.pop('auto_play_ai_tts', None)
    # session.pop('zoom_level', None)
    # session.pop('ai_temperature', None)
    # session.pop('ai_context_window', None)
    # Instead of popping, let initialize_session_values handle defaults on next request.

    session.modified = True # Mark session as modified
    app.logger.info(f"User session cache cleared for {user_email_before_clear}. Settings reset to default. User effectively logged out of this session if they were logged in.")
    return jsonify(status="User session cache cleared successfully. Settings reset to default."), 200 # OK

@app.route(f'{BASE_URL}/tts', methods=['POST'])
@limiter.limit("30/minute") # Limit TTS requests per minute per IP
# @login_required # --- USER AUTH ---
def tts_request_endpoint():
    """Requests Text-to-Speech generation for given text."""
    global edge_tts_available_global
    if not edge_tts_available_global:
        app.logger.warning("TTS request received, but edge-tts not found by Flask process. Task will attempt its own check.")
        # Decide whether to return an error immediately or let the task fail.
        # Returning 501 here provides faster feedback if edge-tts is definitely missing.
        return jsonify({'error': 'Text-to-Speech dependency (edge-tts) is not available on the server.'}), 501


    data = request.get_json()
    if not data or 'text' not in data or not data['text'].strip():
        return jsonify({'error': 'Missing or empty "text" field for TTS'}), 400 # Bad Request

    text_to_speak = data['text']
    # Get voice preference from request or session, default to a standard voice
    voice_preference = data.get('voice', session.get('selected_voice', "en-US-JennyNeural"))

    # Create a temporary file path for the audio output
    temp_audio_dir = app.config['TEMP_AUDIO_FOLDER']
    os.makedirs(temp_audio_dir, exist_ok=True) # Ensure directory exists
    # --- USER AUTH --- Include user/session ID in filename for namespacing
    session_or_user_id = str(session.get('user_id', session.sid))
    audio_filename = f"tts_{session_or_user_id}_{int(time.time())}.mp3" # Unique filename
    # --- END USER AUTH ---
    output_filepath = os.path.join(temp_audio_dir, audio_filename)

    try:
        # Dispatch the TTS generation task to Celery
        task = generate_tts_async.delay(text_to_speak, output_filepath, voice_preference)
        app.logger.info(f"Dispatched TTS generation task {task.id} for voice {voice_preference}")
        return jsonify({'message': 'TTS request submitted. Processing in background.', 'task_id': task.id}), 202 # Accepted
    except Exception as e:
        app.logger.error(f"Error dispatching TTS task: {e}", exc_info=True)
        return jsonify({'error': f'Failed to submit TTS request: {str(e)}'}), 500 # Internal Server Error


@app.route(f'{BASE_URL}/tts_status/<task_id>', methods=['GET'])
# @login_required # --- USER AUTH ---
def tts_status_endpoint(task_id):
    """Checks the status of a TTS generation task."""
    # --- USER AUTH --- Ownership check for task_id if TTS files are sensitive/user-specific
    # This would require storing user_id with the task or TTS file path.
    # --- END USER AUTH ---
    task = generate_tts_async.AsyncResult(task_id, app=celery_app)
    response_data = {'task_id': task_id, 'status': task.state.lower()}

    if task.state == 'PENDING' or task.state == 'STARTED':
        response_data['message'] = f'TTS generation is {task.state.lower()}.'
        return jsonify(response_data), 202 # Accepted (processing)
    elif task.state == 'SUCCESS':
        result = task.result # Get the return value of the task
        if result and result.get('status') == 'completed':
            audio_filepath = result.get('output_filepath')

            # Verify the output file exists and is accessible
            if audio_filepath and os.path.exists(audio_filepath) and os.path.isfile(audio_filepath):
                audio_filename = os.path.basename(audio_filepath)
                # Generate a URL to access the audio file via the get_tts_audio_endpoint
                # Use quote_plus to handle potential special characters in filename
                audio_url = urljoin(request.url_root, f'{BASE_URL}/tts_audio/{quote_plus(audio_filename)}')
                response_data.update({
                    'message': 'TTS generation completed successfully.',
                    'audio_url': audio_url,
                    'audio_filename': audio_filename,
                    'status': 'completed'
                })
                return jsonify(response_data), 200 # OK
            else:
                 # Task reported success but the file is missing
                 error_msg = "TTS task reported success but output file not found or path was invalid."
                 app.logger.error(f"{error_msg} Expected path: {audio_filepath}")
                 response_data.update({
                     'status': 'failed',
                     'error': error_msg
                 })
                 return jsonify(response_data), 500 # Internal Server Error
        else:
             # Task completed but reported an internal failure
             error_msg = result.get('error', 'Unknown TTS processing error reported by task.')
             app.logger.error(f"TTS generation task {task_id} completed with internal failure: {error_msg}")
             response_data.update({
                 'status': 'failed',
                 'error': f'TTS generation failed: {error_msg}'
             })
             return jsonify(response_data), 500 # Internal Server Error


    elif task.state == 'FAILURE':
        # Task failed during execution
        error = str(task.info) if isinstance(task.info, Exception) else str(task.result)
        app.logger.error(f"TTS generation task {task_id} failed: {error}")
        response_data.update({
            'status': 'failed',
            'error': f'TTS generation task failed: {error}'
        })
        return jsonify(response_data), 500 # Internal Server Error
    else:
        # Unknown task state
        response_data['message'] = f'Unknown task state: {task.state}.'
        return jsonify(response_data), 500 # Internal Server Error


@app.route(f'{BASE_URL}/tts_audio/<path:filename>', methods=['GET']) # Use path converter to handle potential slashes
# @login_required # --- USER AUTH --- Consider if access to any TTS audio needs auth
def get_tts_audio_endpoint(filename):
    """Serves a generated TTS audio file."""
    temp_audio_dir = app.config['TEMP_AUDIO_FOLDER']
    # Basic security check against directory traversal
    if ".." in filename or filename.startswith("/") or filename.startswith("\\"):
        app.logger.warning(f"Attempted to access invalid TTS audio filename: {filename}")
        return jsonify({'error': 'Invalid filename'}), 400 # Bad Request

    # --- USER AUTH ---
    # If filenames include user_id (e.g., tts_USERID_timestamp.mp3),
    # you could add a check here to ensure the user is authorized to access this file:
    # current_user_id = str(session.get('user_id'))
    # if not filename.startswith(f"tts_{current_user_id}_"):
    #     app.logger.warning(f"User {current_user_id} attempted to access TTS audio not belonging to them: {filename}")
    #     return jsonify({'error': 'Access denied to this audio file.'}), 403 # Forbidden
    # --- END USER AUTH ---

    # Construct the full path to the file
    filepath = os.path.join(temp_audio_dir, filename) # os.path.join is generally safe

    app.logger.info(f"Attempting to serve TTS audio file: {filepath}")
    # Check if the file exists and is actually a file
    if os.path.exists(filepath) and os.path.isfile(filepath):
        app.logger.info(f"TTS audio file found: {filepath}")
        try:
            # Use send_from_directory for safe file serving
            response = send_from_directory(temp_audio_dir, filename, mimetype='audio/mpeg', as_attachment=False)
            # Add CORS headers if needed (consider global handling with Flask-CORS or Talisman)
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
            app.logger.info(f"Served TTS audio file {filename} with CORS headers.")
            return response
        except Exception as e:
            app.logger.error(f"Error serving TTS audio file {filename}: {e}", exc_info=True)
            return jsonify({'error': 'Could not serve audio file due to server error.'}), 500 # Internal Server Error
    else:
        app.logger.warning(f"Requested TTS audio file not found or is not a file: {filepath}")
        return jsonify({'error': 'Audio file not found or no longer available.'}), 404 # Not Found


@app.route(f'{BASE_URL}/voice_query', methods=['POST'])
@limiter.limit("20/minute") # Limit voice queries per minute per IP
# @login_required # --- USER AUTH ---
def voice_query_endpoint():
    """Handles voice audio input and transcribes it, then sends to AI."""
    global ffmpeg_available_global
    if not voice_query_available:
        return jsonify({'error': 'Speech recognition service (SpeechRecognition library) is not available on the server.'}), 501 # Not Implemented

    if 'audio_file' not in request.files:
        return jsonify({'error': 'No audio file provided in the request.'}), 400 # Bad Request

    audio_file_storage = request.files['audio_file']
    if audio_file_storage.filename == '':
        return jsonify({'error': 'No audio file selected.'}), 400 # Bad Request

    temp_upload_path = None # Path for the initially uploaded file
    temp_wav_path = None # Path for the converted WAV file

    try:
        # Save the uploaded audio file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix="_upload_audio") as temp_upload_file:
            audio_file_storage.save(temp_upload_file.name)
            temp_upload_path = temp_upload_file.name
        app.logger.info(f"Voice query audio (original format) saved temporarily to: {temp_upload_path}")

        processing_audio_path = temp_upload_path # Default path for processing

        # If FFmpeg is available, attempt to convert to WAV (16kHz, 1 channel) for better SR compatibility
        if ffmpeg_available_global:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav_file_obj:
                temp_wav_path = temp_wav_file_obj.name

            ffmpeg_cmd = [
                'ffmpeg', '-i', temp_upload_path, '-acodec', 'pcm_s16le',
                '-ar', '16000', '-ac', '1', '-y', temp_wav_path
            ]
            app.logger.info(f"Attempting FFmpeg conversion: {' '.join(ffmpeg_cmd)}")

            # Ensure FFmpeg is in the PATH for the subprocess
            task_env = os.environ.copy()
            python_dir = os.path.dirname(sys.executable)
            scripts_dir = os.path.join(python_dir, "Scripts")
            if sys.platform == 'win32' and scripts_dir not in task_env.get("PATH", ""):
                 task_env["PATH"] = f"{scripts_dir}{os.pathsep}{task_env.get('PATH', '')}"

            conversion_process = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=False, env=task_env)

            if conversion_process.returncode == 0:
                app.logger.info(f"FFmpeg conversion successful. WAV file at: {temp_wav_path}")
                processing_audio_path = temp_wav_path # Use the converted WAV file
            else:
                app.logger.error(f"FFmpeg conversion failed. Return code: {conversion_process.returncode}")
                app.logger.error(f"FFmpeg stdout: {conversion_process.stdout}")
                app.logger.error(f"FFmpeg stderr: {conversion_process.stderr}")
                app.logger.warning("FFmpeg conversion failed. Attempting to process original uploaded audio.")
                if os.path.exists(temp_wav_path): os.remove(temp_wav_path) # Clean up failed target file
                temp_wav_path = None # Nullify the path


        recognizer = sr.Recognizer()
        # Load the audio file for speech recognition
        with sr.AudioFile(processing_audio_path) as source:
            try:
                # Read the audio data from the file
                audio_data = recognizer.record(source)
            except Exception as rec_e:
                app.logger.error(f"Error reading audio data from file {processing_audio_path}: {rec_e}")
                raise ValueError(f"Audio file ('{os.path.basename(processing_audio_path)}') could not be read. Check format/corruption. Original error: {rec_e}")


        app.logger.info("Attempting to transcribe audio using Google Speech Recognition...")
        # Transcribe the audio using Google's service
        transcribed_text = recognizer.recognize_google(audio_data)
        app.logger.info(f"Voice query transcribed as: '{transcribed_text}'")

        # Add the transcribed text to chat history
        add_to_chat_history(f'User (Voice)', transcribed_text, role_for_ai="user")

        # Send the transcribed text to the AI model
        current_model_name = session.get('current_ollama_model')
        if not current_model_name or current_model_name in ["Loading...", "No Models Found", "Ollama Offline"]:
            return jsonify({'error': 'AI model not available or not selected.'}), 400 # Bad Request

        # Prepare the prompt for the AI, including PDF context if available
        full_prompt_for_ollama = prepare_ai_prompt_and_context(transcribed_text, include_page_context=True)
        ollama_payload = {
            "model": current_model_name,
            "prompt": full_prompt_for_ollama,
            "stream": False,
             "options": {
                "temperature": session.get('ai_temperature', 0.6),
                "num_ctx": session.get('ai_context_window', 4096)
            }
        }

        app.logger.info(f"Sending transcribed voice query to Ollama model: {current_model_name}")
        ollama_api_url = urljoin(app.config['OLLAMA_URL'], "/api/generate")
        response = requests.post(ollama_api_url, json=ollama_payload, timeout=180) # Set a timeout
        response.raise_for_status() # Raise HTTPError for bad responses

        ai_response_data = response.json()
        ai_response_text = ai_response_data.get('response', 'No AI response content.').strip()
        session['last_ai_response'] = ai_response_text
        add_to_chat_history('AI', ai_response_text, role_for_ai="assistant")
        session.modified = True

        app.logger.info("Received AI response for voice query.")
        return jsonify({
            'transcribed_text': transcribed_text,
            'ai_response': ai_response_text,
            'tts_autoplay_triggered': session.get('auto_play_ai_tts', False) and bool(ai_response_text) # Indicate if TTS autoplay should trigger
        }), 200 # OK

    except sr.UnknownValueError:
        # Speech recognition could not understand the audio
        app.logger.warning("Google Speech Recognition could not understand the audio from voice query.")
        return jsonify({'error': 'Speech recognition could not understand audio.'}), 400 # Bad Request
    except sr.RequestError as e:
        # Error from the speech recognition service (e.g., network issue, API key problem)
        app.logger.error(f"Could not request results from Google Speech Recognition service: {e}")
        return jsonify({'error': f'Speech recognition service error: {e}'}), 503 # Service Unavailable
    except ValueError as ve:
        # Custom ValueError raised during audio processing
        app.logger.error(f"Error processing voice query (ValueError): {ve}", exc_info=True)
        return jsonify({'error': f'Voice Query Error: {str(ve)}'}), 500 # Internal Server Error
    except Exception as e:
        # Catch any other unexpected errors during the process
        app.logger.error(f"Error processing voice query: {e}", exc_info=True)
        return jsonify({'error': f'An unexpected error occurred while processing voice query: {str(e)}'}), 500 # Internal Server Error
    finally:
        # Clean up temporary files
        if temp_upload_path and os.path.exists(temp_upload_path):
            try: os.remove(temp_upload_path)
            except Exception as e_del: app.logger.error(f"Error deleting temp uploaded audio: {e_del}")
        if temp_wav_path and os.path.exists(temp_wav_path):
            try: os.remove(temp_wav_path)
            except Exception as e_del: app.logger.error(f"Error deleting temp WAV audio: {e_del}")


@app.route(f'{BASE_URL}/model_capabilities/<path:model_name>', methods=['GET'])
# @login_required # --- USER AUTH ---
def get_model_capabilities_endpoint(model_name):
    """Retrieves capabilities (like vision support) for a specific Ollama model."""
    if not model_name:
        return jsonify({'error': 'Model name cannot be empty.'}), 400 # Bad Request
    capabilities = get_model_capabilities_from_ollama(model_name)
    if not capabilities or "error" in capabilities:
        return jsonify({'error': f'Could not retrieve capabilities for model {model_name}.', 'details': capabilities.get("error", "Unknown error")}), 404 # Not Found or Internal Error
    return jsonify(capabilities), 200 # OK


@app.route(f'{BASE_URL}/settings', methods=['GET'])
# @login_required # --- USER AUTH --- (settings might be user-specific)
def get_settings_endpoint():
    """Retrieves current user/session settings."""
    # --- USER AUTH --- Add user-specific info if needed
    user_info = None
    if 'user_id' in session:
        user_info = {'user_id': session['user_id'], 'email': session.get('email')}
    # --- END USER AUTH ---
    # Compile current settings from session, providing defaults if not set
    settings = {
        'current_ollama_model': session.get('current_ollama_model', "Loading..."),
        'available_ollama_models': session.get('available_ollama_models', ["Loading..."]),
        'selected_personality': session.get('selected_personality', "Default Tutor"),
        'available_personalities': list(PERSONALITIES.keys()), # Provide list of available personalities
        'selected_voice': session.get('selected_voice', "en-US-JennyNeural"),
        'available_voices': [ # Hardcoded list of available edge-tts voices (can be dynamically fetched if needed)
            "en-US-JennyNeural", "en-US-GuyNeural", "en-US-AriaNeural", "en-GB-LibbyNeural", "en-IN-NeerjaNeural",
            "hi-IN-SwaraNeural", "hi-IN-MadhurNeural", "gu-IN-DhwaniNeural", "gu-IN-NiranjanNeural",
            "ta-IN-PallaviNeural", "ta-IN-ValluvarNeural", "te-IN-ShrutiNeural", "te-IN-MohanNeural",
            "mr-IN-AarohiNeural", "mr-IN-ManoharNeural", "bn-IN-TanishaaNeural", "bn-IN-BashkarNeural",
            "ja-JP-NanamiNeural", "ja-JP-KeitaNeural", "ko-KR-SunHiNeural", "ko-KR-InJoonNeural",
            "zh-CN-XiaoxiaoNeural", "zh-CN-YunyangNeural", "zh-HK-HiuGaaiNeural", "zh-HK-WanLungNeural",
            "es-ES-ElviraNeural", "es-MX-JorgeNeural", "fr-FR-DeniseNeural", "fr-CA-SylvieNeural",
            "de-DE-AmalaNeural", "de-DE-BerndNeural", "it-IT-ElsaNeural", "it-IT-DiegoNeural",
            "ru-RU-SvetlanaNeural", "ru-RU-DmitryNeural", "ar-EG-SalmaNeural", "ar-SA-ZariyahNeural",
            "he-IL-HilaNeural", "he-IL-AvriNeural", "pt-BR-FranciscaNeural", "pt-PT-DuarteNeural",
            "nl-NL-ColetteNeural", "nl-NL-MaartenNeural", "tr-TR-EmelNeural", "tr-TR-AhmetNeural"
        ],
        'auto_play_ai_tts': session.get('auto_play_ai_tts', False),
        'zoom_level': session.get('zoom_level', 1.0),
        'ai_temperature': session.get('ai_temperature', 0.6),
        'ai_context_window': session.get('ai_context_window', 4096),
        'user_info': user_info # --- USER AUTH --- Include user info if logged in
    }
    return jsonify(settings), 200 # OK

@app.route(f'{BASE_URL}/settings', methods=['POST'])
@limiter.limit("20/minute") # Limit settings updates per minute per IP
# @login_required # --- USER AUTH ---
def update_settings_endpoint():
    """Updates user/session settings based on POST data."""
    data = request.json
    if not data:
        return jsonify({'error': 'No settings data provided in JSON body.'}), 400 # Bad Request

    response_messages = [] # Collect messages about updates/errors
    settings_updated = False # Flag to indicate if any setting was successfully updated

    available_models = session.get('available_ollama_models', []) # Get current available models from session

    # --- Update settings based on provided data ---
    if 'ollama_model' in data:
        new_model = data['ollama_model']
        # Validate if the requested model is available
        if available_models and isinstance(available_models, list) and new_model in available_models:
            session['current_ollama_model'] = new_model
            message = f'AI model updated to: {new_model}'
            add_to_chat_history('System', message) # Add system message to chat
            response_messages.append(message)
            settings_updated = True
        elif available_models and isinstance(available_models, list) and available_models and available_models[0].startswith("Error"): # Check if list and first item exists
             response_messages.append(f'Error: Cannot update model. Ollama service is unavailable or returned an error: {available_models[0]}')
        elif available_models and isinstance(available_models, list) and available_models and available_models[0] in ["No Models Found", "Ollama Offline", "Ollama Timeout"]:
             response_messages.append(f'Error: Cannot update model. Ollama service status: {available_models[0]}')
        else:
            response_messages.append(f'Error: Model "{new_model}" is not available. No change made.')


    if 'personality' in data:
        new_personality = data['personality']
        # Validate if the requested personality exists
        if new_personality in PERSONALITIES:
            session['selected_personality'] = new_personality
            message = f'AI personality updated to: {new_personality} {PERSONALITIES[new_personality]["icon"]}'
            add_to_chat_history('System', message) # Add system message to chat
            response_messages.append(message)
            settings_updated = True
        else:
            response_messages.append(f'Error: Personality "{new_personality}" not found. No change made.')

    if 'voice' in data:
        # Check if edge-tts is available before allowing voice change
        if not edge_tts_available_global:
             response_messages.append('Error: TTS voice cannot be changed because edge-tts is not available on the server.')
        else:
            # Assuming voice is validated on client or the list is fixed
            session['selected_voice'] = data['voice']
            response_messages.append(f'TTS voice updated to: {data["voice"]}')
            settings_updated = True

    if 'auto_play_ai_tts' in data:
        # If edge-tts is not available, force auto-play off
        if not edge_tts_available_global and bool(data['auto_play_ai_tts']):
             response_messages.append('Warning: Auto-play AI responses cannot be enabled because edge-tts is not available on the server.')
             session['auto_play_ai_tts'] = False # Force it off
        else:
            new_auto_play_state = bool(data['auto_play_ai_tts'])
            session['auto_play_ai_tts'] = new_auto_play_state
            message = f'Auto-play AI responses turned {"ON" if new_auto_play_state else "OFF"}.'
            add_to_chat_history('System', message) # Add system message to chat
            response_messages.append(message)
            settings_updated = True

    if 'ai_temperature' in data:
        try:
            temp = float(data['ai_temperature'])
            # Validate temperature range (typical range for LLMs)
            if 0.0 <= temp <= 2.0:
                session['ai_temperature'] = temp
                response_messages.append(f'AI temperature set to: {temp}')
                settings_updated = True
            else:
                response_messages.append('Error: AI temperature must be between 0.0 and 2.0.')
        except ValueError:
            response_messages.append('Error: Invalid AI temperature value.')

    if 'ai_context_window' in data:
        try:
            ctx_win = int(data['ai_context_window'])
            # Validate context window size (typical range for LLMs)
            if 512 <= ctx_win <= 131072:
                session['ai_context_window'] = ctx_win
                response_messages.append(f'AI context window set to: {ctx_win}')
                settings_updated = True
            else:
                response_messages.append('Error: AI context window size is out of typical range (e.g., 512-131072).')
        except ValueError:
            response_messages.append('Error: Invalid AI context window value.')


    if settings_updated:
        session.modified = True # Mark session as modified if any setting was updated
        # --- USER AUTH ---
        # If you store user preferences in DB, update them here for session.get('user_id')
        # You might want to fetch the updated settings from session to return
        current_user_settings = {
            'ollama_model': session.get('current_ollama_model'),
            'personality': session.get('selected_personality'),
            'voice': session.get('selected_voice'),
            'auto_play_ai_tts': session.get('auto_play_ai_tts'),
            'ai_temperature': session.get('ai_temperature'),
            'ai_context_window': session.get('ai_context_window')
        }
        # --- END USER AUTH ---
        return jsonify({'status': 'success', 'messages': response_messages, 'updated_settings': current_user_settings}), 200 # OK
    else:
        # If no valid settings were provided or updated
        return jsonify({'status': 'no_changes', 'messages': response_messages if response_messages else ["No valid settings provided for update."]}), 200 if not any("Error" in m for m in response_messages) else 400 # OK or Bad Request if there were errors

@app.route(f'{BASE_URL}/refresh_models', methods=['GET'])
@limiter.limit("5/minute") # Limit model refresh requests per minute per IP
# @login_required # --- USER AUTH --- (Refreshing models is likely a global action)
def refresh_models_endpoint():
    """Refreshes the list of available Ollama models."""
    app.logger.info("Refreshing Ollama models list...")
    # Fetch models synchronously and update session
    available_models, current_model = fetch_ollama_models_sync()

    # Check the status of the model fetch
    if available_models and isinstance(available_models, list) and (available_models[0].startswith("Error") or available_models[0] in ["Ollama Offline", "Ollama Timeout", "No Models Found"]) :
        status_message = f"Failed to refresh models: {available_models[0]}"
        app.logger.warning(status_message)
        return jsonify({'error': status_message, 'models': available_models, 'current_model': current_model}), 503 # Service Unavailable

    status_message = f"Ollama models refreshed. Found {len(available_models) if isinstance(available_models, list) else 0} models."

    # Get the current model from the session after the refresh (fetch_ollama_models_sync updates it)
    updated_current_model = session.get('current_ollama_model')

    # Add a note if the previously selected model is no longer available
    if isinstance(available_models, list) and updated_current_model not in available_models and \
       updated_current_model not in ["Loading...", "No Models Found", "Ollama Offline", "Ollama Timeout", "Error Fetching Models", "Error Decoding Response"]:
        status_message += f" Previously selected model might no longer be available; new selection is '{updated_current_model}'."

    add_to_chat_history("System", status_message) # Add system message to chat
    app.logger.info(status_message)
    return jsonify({'status': 'success', 'message': status_message, 'models': available_models, 'current_model': updated_current_model}), 200 # OK


# --- Initial Setup and Startup ---
if __name__ == '__main__':
    # Import PrometheusMetrics here so it's only available in the main process
    from prometheus_flask_exporter import PrometheusMetrics

    app.logger.info("Performing initial startup checks and setup...")

    # Initialize database if necessary within an application context
    with app.app_context(): # <--- Added app.app_context()
        db = get_db()
        try:
            # --- USER AUTH --- Check for users table first as it's fundamental for auth
            db.execute("SELECT 1 FROM users LIMIT 1")
            app.logger.info("'users' table found.")
            try: # Then check for chat_messages if it's still used/needed
                db.execute("SELECT 1 FROM chat_messages LIMIT 1")
                app.logger.info("'chat_messages' table found.")
            except sqlite3.OperationalError as e_chat:
                 if "no such table" in str(e_chat):
                    app.logger.warning("'chat_messages' table not found. Consider adding to schema.sql if needed.")
                 else:
                    app.logger.error(f"Unexpected DB error checking chat_messages: {e_chat}")

        except sqlite3.OperationalError as e_users:
            if "no such table" in str(e_users):
                app.logger.warning("Database schema not found or 'users' table missing. Initializing database...")
                try:
                    init_db() # This will log an error if schema.sql itself is missing
                    app.logger.info("Database initialization attempted.")
                except Exception as e_init:
                    app.logger.critical(f"Failed to initialize database: {e_init}", exc_info=True)
            else:
                app.logger.error(f"Unexpected DB error checking users table: {e_users}")
    # --- END USER AUTH ---

    # Initialize PrometheusMetrics and register counters ONLY in the main process
    try:
        metrics = PrometheusMetrics(app)
        # Define PDF_UPLOADS_COUNTER here
        PDF_UPLOADS_COUNTER = metrics.counter('pdf_uploads_total', 'Total PDF uploads', labels={'status': lambda r: r.status_code})
        app.logger.info("PrometheusMetrics initialized and PDF_UPLOADS_COUNTER registered.")
    except ValueError as e:
        app.logger.warning(f"Could not initialize PrometheusMetrics or register counters: {e}. Metrics will not be available.")
        # Define PDF_UPLOADS_COUNTER as a dummy object if registration fails
        class DummyCounter:
            def labels(self, status): return self
            def inc(self): pass
        PDF_UPLOADS_COUNTER = DummyCounter()


    # Check for necessary external dependencies
    ffmpeg_available_global = check_ffmpeg()
    edge_tts_available_global = check_edge_tts()

    # On Windows, add the Scripts directory to PATH for subprocess calls
    if sys.platform == 'win32':
        python_dir = os.path.dirname(sys.executable)
        scripts_dir = os.path.join(python_dir, "Scripts")
        if scripts_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = f"{scripts_dir}{os.pathsep}{os.environ.get('PATH', '')}"
            app.logger.info(f"Added {scripts_dir} to PATH for Windows.")

    @app.before_request
    def initialize_session_values():
        """Initializes default session values if they are not already set."""
        # --- USER AUTH ---
        # User-specific session values (user_id, email) are set during login.
        # Global/anonymous session values are set here if not present.
        # --- END USER AUTH ---
        # Fetch Ollama models if not already in session (happens on first request or after clear_cache)
        if 'available_ollama_models' not in session:
             app.logger.info("Session models not found, fetching from Ollama...")
             fetch_ollama_models_sync() # This also sets current_ollama_model

        # Default settings for any session (logged in or not)
        # These might be overridden by user-specific preferences if you implement that later
        default_settings = {
            'selected_personality': "Default Tutor",
            'selected_voice': "en-US-JennyNeural",
            'auto_play_ai_tts': False,
            'zoom_level': 1.0,
            'ai_temperature': 0.6,
            'ai_context_window': 4096
        }
        changed = False
        for key, value in default_settings.items():
            if key not in session:
                session[key] = value
                changed = True

        if changed:
            session.modified = True # Mark session as modified


    app.logger.info(f"Starting Flask server in {'debug' if app.debug else 'production'} mode")
    # Run the Flask development server
    # Note: Flask's built-in server is NOT recommended for production use.
    # Use a production-ready WSGI server like Gunicorn or uWSGI instead.
    app.run(
        debug=os.environ.get('FLASK_DEBUG', 'false').lower() == 'true', # Debug mode based on env var
        host=os.environ.get('FLASK_HOST', '127.0.0.1'), # Host based on env var, default to localhost
        port=int(os.environ.get('FLASK_PORT', '5000')), # Port based on env var, default to 5000
        threaded=True # Enable threading for handling multiple requests (suitable for dev server)
    )
