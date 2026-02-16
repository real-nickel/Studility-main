import os
import uuid
import secrets
import requests  # Make sure to pip install requests
import logging
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import io
import re
from urllib.parse import urlparse, parse_qs

# --- CONFIGURATION ---
app = Flask(__name__)
# Security: prefer environment-provided secret; fall back to a dev key
app.secret_key = os.getenv('SECRET_KEY', 'rhgjpi3ohrgjj23ohwejglghgrioj')
# basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DATABASE: allow DATABASE_URL env var (Vercel/Supabase/PlanetScale friendly). Fall back to sqlite for local dev.
database_url = os.getenv('DATABASE_URL') or os.getenv('DATABASE_URI')
if database_url:
    # Normalize common Heroku-style postgres URL to SQLAlchemy driver format
    db_lower = database_url.lower()
    if db_lower.startswith('postgres://'):
        # Replace scheme to the modern SQLAlchemy driver form
        database_url = database_url.replace('postgres://', 'postgresql+psycopg2://', 1)
        logger.info('Normalized DATABASE_URL scheme to postgresql+psycopg2')
    elif db_lower.startswith('postgresql://') and '+psycopg2' not in db_lower:
        database_url = database_url.replace('postgresql://', 'postgresql+psycopg2://', 1)

    app.config['SQLALCHEMY_DATABASE_URI'] = database_url
    # Optional SSL support for managed Postgres. Set DB_SSLMODE=require to enable.
    if 'postgres' in db_lower:
        sslmode = os.getenv('DB_SSLMODE') or os.getenv('PGSSLMODE')
        if sslmode:
            app.config.setdefault('SQLALCHEMY_ENGINE_OPTIONS', {})
            app.config['SQLALCHEMY_ENGINE_OPTIONS'].setdefault('connect_args', {})
            app.config['SQLALCHEMY_ENGINE_OPTIONS']['connect_args']['sslmode'] = sslmode
            logger.info(f"Postgres SSL mode set to: {sslmode}")
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///studility.db'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# Helpful engine options (pre-ping to avoid stale connections in serverless environments)
app.config.setdefault('SQLALCHEMY_ENGINE_OPTIONS', {})
app.config['SQLALCHEMY_ENGINE_OPTIONS'].setdefault('pool_pre_ping', True)

# basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

db = SQLAlchemy(app)
migrate = Migrate(app, db)

# --- DEEPINFRA CONFIGURATION ---
# usage: Get your key from https://deepinfra.com/
GROQ_API_KEY = "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"

def ask_ai(system_prompt, user_prompt):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.3-70b-versatile", # Very powerful model
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status() # Raise error if API fails (400/500 codes)
        
        data = response.json()
        return data['choices'][0]['message']['content']
        
    except Exception as e:
        print(f"AI Error: {e}")
        return "Error: Could not contact API. Please check your API key."

# --- DATABASE MODELS ---

class Notebook(db.Model):
    id = db.Column(db.String(36), primary_key=True)
    owner_name = db.Column(db.String(100), nullable=False)
    edit_password_hash = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    sources = db.relationship('Source', backref='notebook', lazy=True)
    chats = db.relationship('Chat', backref='notebook', lazy=True)
    documents = db.relationship('Document', backref='notebook', lazy=True)

class Source(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    title = db.Column(db.String(200), default="Untitled Source")
    notebook_id = db.Column(db.String(36), db.ForeignKey('notebook.id'), nullable=False)


class Document(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), default="Untitled Document")
    content = db.Column(db.Text, nullable=False)
    last_edited = db.Column(db.DateTime, default=datetime.utcnow)
    notebook_id = db.Column(db.String(36), db.ForeignKey('notebook.id'), nullable=False)

class Chat(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    role = db.Column(db.String(10), nullable=False)
    message = db.Column(db.Text, nullable=False)
    notebook_id = db.Column(db.String(36), db.ForeignKey('notebook.id'), nullable=False)


def _extract_text_from_url(url, max_chars=20000):
    """Try to extract useful text from a URL.

    Supports HTML pages (BeautifulSoup), PDFs (PyPDF2) and YouTube transcript (youtube-transcript-api).
    Returns extracted text or raises RuntimeError with a helpful message.
    """
    try:
        # Lazy imports so the app can still start if optional packages aren't installed
        import requests
    except Exception:
        raise RuntimeError("Missing dependency: requests. Please `pip install requests`.")

    parsed = urlparse(url)
    netloc = parsed.netloc.lower()

    # YouTube handling
    if 'youtube.com' in netloc or 'youtu.be' in netloc:
        try:
            # extract video id
            video_id = None
            if 'youtube.com' in netloc:
                qs = parse_qs(parsed.query)
                video_id = qs.get('v', [None])[0]
            if not video_id and 'youtu.be' in netloc:
                video_id = parsed.path.lstrip('/')

            if not video_id:
                raise RuntimeError('Could not determine YouTube video id from URL')

            # Try multiple strategies to call the installed youtube-transcript-api (which can expose different APIs)
            transcript_list = None
            yta_mod = None
            try:
                import youtube_transcript_api as yta_mod
            except Exception:
                yta_mod = None

            # 1) Prefer the common class-based API if present
            if yta_mod is not None:
                YouTubeTranscriptApi = getattr(yta_mod, 'YouTubeTranscriptApi', None)
                try:
                    if YouTubeTranscriptApi is not None and hasattr(YouTubeTranscriptApi, 'get_transcript'):
                        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                except Exception:
                    transcript_list = None

            # 2) Try a module-level convenience function
            if transcript_list is None:
                try:
                    from youtube_transcript_api import get_transcript as yta_get_transcript
                    transcript_list = yta_get_transcript(video_id)
                except Exception:
                    transcript_list = None

            # 3) Heuristic: scan module for callables with 'transcript' or 'fetch' in the name and try them
            if transcript_list is None and yta_mod is not None:
                import inspect
                tried = []
                for name in dir(yta_mod):
                    if name.startswith('_'):
                        continue
                    lname = name.lower()
                    if 'transcript' in lname or 'fetch' in lname:
                        attr = getattr(yta_mod, name)
                        if callable(attr):
                            # inspect signature to avoid accidentally calling functions that need many params
                            try:
                                sig = inspect.signature(attr)
                                params = [p for p in sig.parameters.values() if p.default is inspect._empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
                                # only call functions that require 0 or 1 required positional arg (video id)
                                if len(params) <= 1:
                                    tried.append(name)
                                    try:
                                        result = attr(video_id) if len(params) == 1 else attr()
                                        if result:
                                            transcript_list = result
                                            break
                                    except Exception:
                                        continue
                            except Exception:
                                # If we cannot inspect signature, still try to call safely
                                try:
                                    tried.append(name)
                                    result = attr(video_id)
                                    if result:
                                        transcript_list = result
                                        break
                                except Exception:
                                    continue

            # 4) Try classes like Transcript / FetchedTranscript that may have constructors or helpers
            if transcript_list is None and yta_mod is not None:
                for cls_name in ('Transcript', 'FetchedTranscript', 'TranscriptList'):
                    cls = getattr(yta_mod, cls_name, None)
                    if cls is not None:
                        try:
                            # try class(video_id) or class.from_id(video_id)
                            if callable(cls):
                                try:
                                    candidate = cls(video_id)
                                    if candidate:
                                        transcript_list = candidate
                                        break
                                except Exception:
                                    # try common factory methods
                                    for method in ('from_video_id', 'from_id', 'fetch', 'get'):
                                        m = getattr(cls, method, None)
                                        if callable(m):
                                            try:
                                                candidate = m(video_id)
                                                if candidate:
                                                    transcript_list = candidate
                                                    break
                                            except Exception:
                                                continue
                                    if transcript_list:
                                        break
                        except Exception:
                            continue

            # If still nothing, provide debug info and bail
            if not transcript_list:
                extra = ''
                try:
                    if yta_mod is not None:
                        attrs = [a for a in dir(yta_mod) if not a.startswith('_')]
                        extra = f" Available module attributes: {attrs[:50]}"
                except Exception:
                    extra = ''
                raise RuntimeError('YouTube transcript API unavailable or no transcript found. Consider installing/updating youtube-transcript-api.' + extra)

            # Normalize transcript_list into text
            # If it's an iterable of dicts with 'text'
            text_parts = []
            try:
                # If it's a dict-like Transcript object, try to extract 'text' fields
                if isinstance(transcript_list, dict):
                    # maybe contains 'transcript' key
                    for v in transcript_list.values():
                        if isinstance(v, (list, tuple)):
                            for t in v:
                                if isinstance(t, dict) and 'text' in t:
                                    text_parts.append(t['text'])
                else:
                    for item in transcript_list:
                        if isinstance(item, dict) and 'text' in item:
                            text_parts.append(item['text'])
                        else:
                            # fallback: string representation
                            text_parts.append(str(item))
            except Exception:
                # Last resort: stringify the result
                text_parts = [str(transcript_list)]

            text = "\n".join(text_parts)
            return text[:max_chars]
        except Exception as e:
            raise RuntimeError(f"YouTube transcript error: {e}")

    # Fetch content
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Failed to fetch URL: {e}")

    content_type = resp.headers.get('content-type', '')

    # PDF handling
    if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
        try:
            try:
                from PyPDF2 import PdfReader
            except Exception:
                raise RuntimeError("Missing dependency: PyPDF2. Please `pip install PyPDF2`.")

            reader = PdfReader(io.BytesIO(resp.content))
            texts = []
            for page in reader.pages:
                try:
                    texts.append(page.extract_text() or "")
                except Exception:
                    continue
            return "\n\n".join(texts)[:max_chars]
        except Exception as e:
            raise RuntimeError(f"PDF parsing error: {e}")

    # HTML handling
    try:
        try:
            from bs4 import BeautifulSoup
        except Exception:
            raise RuntimeError("Missing dependency: beautifulsoup4. Please `pip install beautifulsoup4`.")

        soup = BeautifulSoup(resp.text, 'html.parser')
        # Remove scripts/styles
        for tag in soup(['script', 'style', 'noscript', 'header', 'footer', 'svg', 'form']):
            tag.decompose()
        text = soup.get_text(separator='\n')
        # Collapse multiple newlines and whitespace
        text = re.sub(r'\n\s+\n', '\n\n', text)
        text = re.sub(r'[ \t]{2,}', ' ', text)
        text = text.strip()
        return text[:max_chars]
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"HTML parsing error: {e}")


@app.route('/n/<notebook_id>/doc', methods=['GET'])
def get_document(notebook_id):
    """Return the primary document for a notebook, creating one if necessary."""
    notebook = Notebook.query.get_or_404(notebook_id)
    doc = Document.query.filter_by(notebook_id=notebook_id).first()
    if not doc:
        doc = Document(title='Untitled Document', content='', notebook_id=notebook_id)
        db.session.add(doc)
        db.session.commit()

    return jsonify({
        'doc_id': doc.id,
        'title': doc.title,
        'content': doc.content,
        'last_edited': doc.last_edited.isoformat() if doc.last_edited else None
    })


@app.route('/n/<notebook_id>/doc/save', methods=['POST'])
def save_document(notebook_id):
    """Save document content. Requires edit access."""
    if not session.get(f'edit_{notebook_id}'):
        return jsonify({"error": "Unauthorized"}), 403

    data = request.get_json() or {}
    doc_id = data.get('doc_id')
    content = data.get('content', '')
    title = data.get('title')

    if doc_id:
        doc = Document.query.get(doc_id)
        if not doc or doc.notebook_id != notebook_id:
            return jsonify({"error": "Document not found"}), 404
    else:
        doc = Document(title=title or 'Untitled Document', content='', notebook_id=notebook_id)
        db.session.add(doc)

    doc.content = content
    if title:
        doc.title = title[:200]
    doc.last_edited = datetime.utcnow()
    db.session.commit()

    return jsonify({"success": True, "doc_id": doc.id, "last_edited": doc.last_edited.isoformat()})


@app.route('/n/<notebook_id>/add_source_by_url', methods=['POST'])
def add_source_by_url(notebook_id):
    """Fetch a URL (HTML / PDF / YouTube) and add its text as a Source for the notebook. Requires edit access."""
    if not session.get(f'edit_{notebook_id}'):
        return jsonify({"error": "Unauthorized"}), 403

    data = request.get_json() or request.form
    url = data.get('url')
    title = data.get('title') or url

    if not url:
        return jsonify({"error": "URL is required"}), 400

    try:
        text = _extract_text_from_url(url)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 400

    if not text:
        return jsonify({"error": "No text could be extracted from the provided URL."}), 400

    # Create Source
    source = Source(title=title[:200], content=text, notebook_id=notebook_id)
    db.session.add(source)
    db.session.commit()
    return jsonify({"success": True, "source_id": source.id})

# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/create', methods=['POST'])
def create_notebook():
    username = request.form.get('username')
    if not username:
        flash("Username is required")
        return redirect(url_for('index'))

    notebook_id = str(uuid.uuid4())
    raw_password = secrets.token_urlsafe(8)
    
    new_notebook = Notebook(
        id=notebook_id,
        owner_name=username,
        edit_password_hash=generate_password_hash(raw_password)
    )
    
    db.session.add(new_notebook)
    db.session.commit()
    
    return render_template('index.html', new_notebook=new_notebook, raw_password=raw_password)

@app.route('/n/<notebook_id>')
def view_notebook(notebook_id):
    notebook = Notebook.query.get_or_404(notebook_id)
    can_edit = session.get(f'edit_{notebook_id}') == True
    return render_template('notebook.html', notebook=notebook, can_edit=can_edit)

@app.route('/n/<notebook_id>/login', methods=['POST'])
def login_notebook(notebook_id):
    notebook = Notebook.query.get_or_404(notebook_id)
    password = request.form.get('password')
    
    if check_password_hash(notebook.edit_password_hash, password):
        session[f'edit_{notebook_id}'] = True
        flash("Edit access granted!")
    else:
        flash("Incorrect password")
    
    return redirect(url_for('view_notebook', notebook_id=notebook_id))

@app.route('/n/<notebook_id>/add_source', methods=['POST'])
def add_source(notebook_id):
    if not session.get(f'edit_{notebook_id}'):
        return jsonify({"error": "Unauthorized"}), 403
        
    title = request.form.get('title')
    content = request.form.get('content')
    
    source = Source(title=title, content=content, notebook_id=notebook_id)
    db.session.add(source)
    db.session.commit()
    return jsonify({"success": True})

@app.route('/n/<notebook_id>/query', methods=['POST'])
def query_ai(notebook_id):
    notebook = Notebook.query.get_or_404(notebook_id)
    user_question = request.json.get('question')
    
    # RAG Context Construction
    context = "\n\n".join([f"Source '{s.title}': {s.content}" for s in notebook.sources])
    
    if not context:
        return jsonify({"answer": "Please add some sources first."})

    # System Prompt
    system_prompt = f"""
    You are Studility, an AI study assistant. Answer based mostly on the context below.
    CONTEXT:
    {context}
    """
    
    answer = ask_ai(system_prompt, user_question)
    
    db.session.add(Chat(role='user', message=user_question, notebook_id=notebook_id))
    db.session.add(Chat(role='ai', message=answer, notebook_id=notebook_id))
    db.session.commit()
    
    return jsonify({"answer": answer})

@app.route('/n/<notebook_id>/generate/<tool_type>', methods=['POST'])
def generate_tool(notebook_id, tool_type):
    notebook = Notebook.query.get_or_404(notebook_id)
    context = "\n\n".join([s.content for s in notebook.sources])
    
    if not context:
        return jsonify({"result": "No sources available."})

    prompts = {
        "flashcards": "Generate 5 flashcards (Front/Back) based on the text. Return ONLY JSON format: [{\"front\": \"...\", \"back\": \"...\"}].",
        "quiz": "Generate a multiple choice quiz with 3 questions. Return text format.",
        "slides": "Generate an outline for a 5-slide presentation."
    }
    
    prompt = prompts.get(tool_type, "Summarize this.")
    response = ask_ai(f"Context: {context}", prompt)
    
    return jsonify({"result": response})

# Initialize DB: create tables automatically for local sqlite, otherwise recommend migrations on managed DBs
with app.app_context():
    db_uri = app.config.get('SQLALCHEMY_DATABASE_URI', '')
    if db_uri.startswith('sqlite:'):
        db.create_all()
        logger.info('Initialized sqlite database with create_all().')
    else:
        # On managed DBs (Postgres/MySQL) prefer explicit migrations (Alembic) in production.
        logger.info('Detected non-sqlite database. Ensure migrations are applied (Alembic / Flyway).')

if __name__ == '__main__':
    app.run(debug=True)