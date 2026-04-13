from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import asyncio
from functools import wraps

from src.pipeline.rag_pipeline import ingest_document, query_document, delete_document, check_if_exists
from config import settings

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.secret_key = settings.API_SECRET_KEY


def run_async(coro):
    return asyncio.run(coro)


def login_required(route_function):
    @wraps(route_function)
    def wrapper(*args, **kwargs):
        if not session.get("authenticated"):
            return redirect(url_for("login"))
        return route_function(*args, **kwargs)
    return wrapper


# --- LOGIN ROUTES ---

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        entered_key = request.form.get('api_secret_key')

        if entered_key == settings.API_SECRET_KEY:
            session["authenticated"] = True
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error="Invalid secret key")

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


# --- PAGE ROUTES ---

@app.route('/')
@login_required
def index():
    return render_template('index.html')


@app.route('/home')
@login_required
def home():
    case_id = request.args.get('case_id')
    if not case_id:
        return redirect(url_for('index'))

    return render_template('home.html', case_id=case_id)


# --- API ROUTES ---

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files or 'case_id' not in request.form:
        return jsonify({"error": "Missing file or case ID"}), 400

    file = request.files['file']
    case_id = request.form['case_id']

    if file and file.filename.lower().endswith('.pdf'):
        pdf_bytes = file.read()
        try:
            run_async(ingest_document(pdf_bytes, case_id))
            return jsonify({
                "success": True,
                "redirect_url": url_for('home', case_id=case_id)
            }), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Invalid file format. Please upload a PDF."}), 400


@app.route('/ask', methods=['POST'])
@login_required
def ask_question():
    data = request.get_json(silent=True) or {}
    case_id = data.get('case_id')
    question = data.get('question')

    try:
        result = run_async(query_document(case_id, question))
        steps_dict = [{"step": s.step, "content": s.content} for s in result.steps]
        return jsonify({"steps": steps_dict, "pages": result.pages}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/delete', methods=['POST'])
@login_required
def delete_case():
    data = request.get_json(silent=True) or {}
    case_id = data.get('case_id')

    success = run_async(delete_document(case_id))
    if success:
        return jsonify({"success": True, "redirect_url": url_for('index')}), 200
    else:
        return jsonify({"error": "Failed to delete case."}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)