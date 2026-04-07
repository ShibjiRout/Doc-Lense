from flask import Flask, render_template, request, jsonify, redirect, url_for
import asyncio
from src.pipeline.rag_pipeline import ingest_document, query_document, delete_document, check_if_exists

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

def run_async(coro):
    return asyncio.run(coro)

# --- PAGE ROUTES ---

@app.route('/')
def index():
    # Page 1: The Landing Page
    return render_template('index.html')

@app.route('/home')
def home():
    # Page 2: The ChatGPT-style interface
    case_id = request.args.get('case_id')
    if not case_id:
        return redirect(url_for('index')) # Force user back if no Case ID is provided
    
    return render_template('home.html', case_id=case_id)


# --- API ROUTES (Used by the HTML pages in the background) ---

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files or 'case_id' not in request.form:
        return jsonify({"error": "Missing file or case ID"}), 400
        
    file = request.files['file']
    case_id = request.form['case_id']
    
    if file and file.filename.lower().endswith('.pdf'):
        pdf_bytes = file.read()
        try:
            run_async(ingest_document(pdf_bytes, case_id))
            # On success, tell the frontend to redirect to the home page
            return jsonify({"success": True, "redirect_url": url_for('home', case_id=case_id)}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
            
    return jsonify({"error": "Invalid file format. Please upload a PDF."}), 400

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    case_id = data.get('case_id')
    question = data.get('question')
    
    try:
        result = run_async(query_document(case_id, question)) 
        steps_dict = [{"step": s.step, "content": s.content} for s in result.steps]
        return jsonify({"steps": steps_dict, "pages": result.pages}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/delete', methods=['POST'])
def delete_case():
    data = request.json
    case_id = data.get('case_id')
    
    success = run_async(delete_document(case_id))
    if success:
        return jsonify({"success": True, "redirect_url": url_for('index')}), 200
    else:
        return jsonify({"error": "Failed to delete case."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)