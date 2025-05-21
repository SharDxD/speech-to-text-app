from flask import Flask, abort, render_template, request, jsonify, session, redirect, url_for, send_from_directory
from auth import auth
from db import transcripts, users
import os
from pathlib import Path
import numpy as np
import torch
import soundfile as sf
import librosa
from datetime import datetime
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from werkzeug.utils import secure_filename
from pydub import AudioSegment
from pymongo import MongoClient
from bson.objectid import ObjectId
from io import BytesIO

UPLOAD_FOLDER = 'uploads'
app = Flask(__name__)
app.secret_key = "keykeykey"  #temp not for prod
app.register_blueprint(auth)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
BASEDIR = os.path.abspath(os.path.dirname(__file__))

MODEL_PATHS = {
    #"poz": "openai/whisper-tiny",
    "rus": Path(BASEDIR) / "model" / "whisper-finetune-my" / "tinyru", #os.path.join(BASEDIR, "models", "tinyru"),
    "eng": Path(BASEDIR) / "model" / "whisper-finetune-my" / "tinyru", #change to Path(BASEDIR) / "model" / "whisper-finetune-my" / "tinyen",
    "hun": Path(BASEDIR) / "model" / "whisper-finetune-my" / "tinyhu"
}
processor = {}
model     = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for key, path in MODEL_PATHS.items():
    #path = os.path.abspath(path).replace("\\", "/")
    path = Path(path)
    if not path.is_dir():
        raise RuntimeError(f"Model directory not found: {path}")
    processor[key] = WhisperProcessor.from_pretrained(path, local_files_only=True)
    model[key]     = WhisperForConditionalGeneration.from_pretrained(path, local_files_only=True).to(device)



def transcribe_audio_array(selected, audio: np.ndarray, sampling_rate: int = 16000) -> str:
    # run the transcription with the chosen model
    proc = processor[selected]
    mdl  = model[selected]

    if sampling_rate != 16000:
        audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    inputs = proc.feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
    inputs = inputs.input_features.to(device)   # use the same device
    with torch.no_grad():
        generated_ids = mdl.generate(inputs, max_length=225)
    return proc.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

def get_transcripts_collection():
    client = MongoClient(
        "mongodb://localhost:27017",
        serverSelectionTimeoutMS=5000,  # fail fast if down
        connectTimeoutMS=5000
    )
    db = client["speech_app"]
    return db["transcripts"]

@app.route("/")
def index():
    if "user_id" not in session:
        return redirect(url_for("auth.login"))

    user_id = session["user_id"]

    # fetch all transcripts for this user, newest first
    cursor = transcripts.find(
        {"user_id": user_id}
    ).sort("timestamp", -1)

    # build history, skipping any doc without timestamp
    history = [
        {"text": d.get("text", ""), "timestamp": d.get("timestamp")}
        for d in cursor
        if d.get("timestamp") is not None
    ]

    return render_template("index.html", history=history)


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'API is running'}), 200


@app.route('/admin/users', methods=['GET'])
def admin_users():
    # only admins may view
    if session.get('role') != 'admin':
        return abort(403)

    # pull all user docs
    all_users = list(users.find({}, {'_id': 1, 'email': 1, 'role': 1}))
    return render_template('admin_users.html', users=all_users)

@app.route('/admin/users/delete/<user_id>', methods=['POST'])
def delete_user(user_id):
    if session.get('role') != 'admin':
        return abort(403)

    try:
        users.delete_one({'_id': ObjectId(user_id)})
    except Exception:
        # you could flash an error here
        pass

    return redirect(url_for('admin_users'))


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    if "audio" not in request.files or "model" not in request.form:
        return jsonify({"error": "Must send both audio file and model"}), 400
    
    selected = request.form["model"]
    if selected not in MODEL_PATHS:
        return jsonify({"error": f"Unknown model '{selected}'"}), 400
    

    audio_file = request.files["audio"]
    raw_bytes  = audio_file.read()

    webm_io = BytesIO(raw_bytes)
    audio_seg = AudioSegment.from_file(webm_io, format="webm")
    audio_seg = audio_seg.set_frame_rate(16000).set_channels(1)
    wav_io   = BytesIO()
    audio_seg.export(wav_io, format="wav")
    wav_io.seek(0)

    # 2) Read that WAV bytes via soundfile
    audio, sr = sf.read(wav_io, dtype="float32")

    # 3) Run your transcription helper, which takes numpy+sr
    try:
        # run the transcription with the chosen model
        transcription = transcribe_audio_array(selected, audio, sampling_rate=sr)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


    transcripts.insert_one({
        "user_id": session["user_id"],
        "model": selected,
        # "filename": filename,
        "timestamp": datetime.now(),
        "text": transcription
    })

    user_id = session["user_id"]
    cursor = transcripts.find(
        {"user_id": user_id},
        {"_id": 0, "text": 1, "timestamp": 1}
    ).sort("timestamp", -1)

    history = [doc["text"] for doc in cursor]

    return jsonify({
        "transcription": transcription,
        "history": history,
        # "audio_url": f"/uploads/{filename}"
    })


if __name__ == '__main__':
    app.run(debug=True, port=5050, use_reloader=False)

