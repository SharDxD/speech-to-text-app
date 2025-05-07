from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_from_directory
from auth import auth
from db import transcripts
import os
import numpy as np
import torch
import soundfile as sf
import librosa
from datetime import datetime
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from werkzeug.utils import secure_filename
from pydub import AudioSegment
from pymongo import MongoClient

UPLOAD_FOLDER = 'uploads'
app = Flask(__name__)
app.secret_key = "keykeykey"  #temp
app.register_blueprint(auth)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_DIR = os.path.join(os.getcwd(), "C:/Users/kirig/Documents/model_0/whisper-finetune/ft_quick_tiny")  

# load processor & model only once
processor = WhisperProcessor.from_pretrained(MODEL_DIR)
# model = WhisperForConditionalGeneration.from_pretrained(MODEL_DIR).to("cuda")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WhisperForConditionalGeneration.from_pretrained(MODEL_DIR).to(device)

def transcribe_audio_array(audio: np.ndarray, sampling_rate: int = 16000) -> str:
    if sampling_rate != 16000:
        audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    inputs = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
    inputs = inputs.input_features.to(device)   # use the same device
    with torch.no_grad():
        generated_ids = model.generate(inputs, max_length=225)
    return processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

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

# @app.route("/transcribe", methods=["POST"])
# def transcribe():
#     if "user_id" not in session:
#         return jsonify({"error": "Unauthorized"}), 401

#     if "audio" not in request.files:
#         return jsonify({"error": "No audio file"}), 400

#     audio_file = request.files["audio"]
#     filename = f"{datetime.utcnow().timestamp()}_{audio_file.filename}"
#     filepath = os.path.join(UPLOAD_FOLDER, filename)
#     audio_file.save(filepath)

#     # Save to DB
#     transcripts.insert_one({
#         "user_id": session["user_id"],
#         "filename": filename,
#         "timestamp": datetime.utcnow(),
#         "text": "This is a placeholder transcription"
#     })

#     return jsonify({
#         "transcription": "This is a placeholder transcription",
#         "audio_url": f"/uploads/{filename}"
#     })

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400

    audio_file = request.files["audio"]
    # secure and timestamp the filename
    filename = f"{datetime.utcnow().timestamp()}_{secure_filename(audio_file.filename)}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    audio_file.save(filepath)

    # read back into numpy array (any format ffmpeg/browser recorder produces)
    #audio, sr = sf.read(filepath, dtype="float32")

    # if it’s a WebM, convert it to WAV first
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".webm":
        wav_path = filepath[:-5] + ".wav"
        audio_segment = AudioSegment.from_file(filepath, format="webm")
        audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
        audio_segment.export(wav_path, format="wav")
        read_path = wav_path
    else:
        read_path = filepath
    #print("test")
    # now read the WAV (or whatever)
    audio, sr = sf.read(read_path, dtype="float32")
    #print("test2")
    try:
        # run your helper
        #print("test3")
        transcription = transcribe_audio_array(audio, sampling_rate=sr)
        #print("test4:", transcription)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    # persist into MongoDB
    # transcripts.insert_one({
    #     "user_id": session["user_id"],
    #     "filename": filename,
    #     "timestamp": datetime.utcnow(),
    #     "text": transcription
    # })

    col = get_transcripts_collection()
    col.insert_one({
         "user_id": session["user_id"],
         "filename": filename,
         "timestamp": datetime.utcnow(),
         "text": transcription
    })


    return jsonify({
        "transcription": transcription,
        "audio_url": f"/uploads/{filename}"
    })


if __name__ == '__main__':
    app.run(debug=True, port=5050, use_reloader=False)
