from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_from_directory
from auth import auth
from db import transcripts
import os
from datetime import datetime

UPLOAD_FOLDER = 'uploads'
app = Flask(__name__)
app.secret_key = "keykeykey"  #temp
app.register_blueprint(auth)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def index():
    if "user_id" not in session:
        return redirect(url_for("auth.login"))
    return render_template("index.html")

#@app.route('/', methods=['POST'])
#def block_spam():
#    print("🔎 Got POST / from:", request.user_agent.string)
#    return '', 204
#def fallback():
#    return jsonify({'error': 'Please use /transcribe instead.'}), 404


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'API is running'}), 200

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400

    audio_file = request.files["audio"]
    filename = f"{datetime.utcnow().timestamp()}_{audio_file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    audio_file.save(filepath)

    # Save to DB
    transcripts.insert_one({
        "user_id": session["user_id"],
        "filename": filename,
        "timestamp": datetime.utcnow(),
        "text": "This is a placeholder transcription"
    })

    return jsonify({
        "transcription": "This is a placeholder transcription",
        "audio_url": f"/uploads/{filename}"
    })

if __name__ == '__main__':
    app.run(debug=True, port=5050, use_reloader=False)
