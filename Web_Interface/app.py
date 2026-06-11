import os
import uuid
from io import BytesIO

import torch
from flask import Flask, jsonify, render_template, request, send_file
from flask_cors import CORS

# This gets the absolute path to your main project folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ==============================================================================
# PyTorch compatibility patch for Coqui TTS checkpoints
# ==============================================================================
# PyTorch 2.6+ defaults can block complex Coqui checkpoint config objects.
# The application keeps the legacy loading behavior for local trusted models.
print("Applying PyTorch checkpoint compatibility patch for Coqui TTS.")

_original_torch_load = torch.load


def patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = patched_torch_load
print("Checkpoint compatibility patch is active.")

# Imports that must run after the torch.load patch.
from TTS.api import TTS
from TTS.utils.synthesizer import Synthesizer

app = Flask(__name__)
CORS(app)

# ==============================================================================
# Model configuration
# ==============================================================================
use_cuda = torch.cuda.is_available()
device_name = "cuda" if use_cuda else "cpu"
print(f"Hardware detected: {device_name.upper()}")

MY_MODEL_PATH = os.path.join(BASE_DIR, "proof_logs", "run_2026-02-12_16-48-52", "run-February-12-2026_04+48PM-0000000", "best_model.pth")
MY_CONFIG_PATH = os.path.join(BASE_DIR, "proof_logs", "run_2026-02-12_16-48-52", "run-February-12-2026_04+48PM-0000000", "config.json")

UPLOAD_FOLDER = "temp_uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def parse_speed(value):
    try:
        speed = float(value)
    except (TypeError, ValueError):
        return 1.0

    return max(0.5, min(speed, 2.0))


def apply_speed_to_waveform(wav, speed):
    if abs(speed - 1.0) < 0.01:
        return wav

    try:
        import librosa
        import numpy as np

        waveform = np.asarray(wav, dtype=np.float32)
        return librosa.effects.time_stretch(waveform, rate=speed)
    except Exception as exc:
        print(f"Speed adjustment was skipped for in-memory synthesis output: {exc}")
        return wav


def apply_speed_to_file(path, speed):
    if abs(speed - 1.0) < 0.01:
        return

    try:
        import librosa
        import soundfile as sf

        waveform, sample_rate = librosa.load(path, sr=None, mono=False)
        adjusted = librosa.effects.time_stretch(waveform, rate=speed)
        sf.write(path, adjusted.T if adjusted.ndim > 1 else adjusted, sample_rate)
    except Exception as exc:
        print(f"Speed adjustment was skipped for generated file output: {exc}")


def remove_if_exists(path):
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError as exc:
        print(f"Temporary file cleanup skipped for {path}: {exc}")


def send_generated_audio(path, download_name):
    with open(path, "rb") as audio_file:
        audio_bytes = audio_file.read()

    remove_if_exists(path)
    return send_file(
        BytesIO(audio_bytes),
        mimetype="audio/wav",
        as_attachment=False,
        download_name=download_name,
        max_age=0,
    )


# ==============================================================================
# Load AI models
# ==============================================================================
print("\n[1/2] Loading custom text-to-speech model.")
try:
    custom_engine = Synthesizer(
        tts_checkpoint=MY_MODEL_PATH,
        tts_config_path=MY_CONFIG_PATH,
        use_cuda=use_cuda,
    )
    print("Custom text-to-speech model loaded.")
except Exception as exc:
    print(f"Custom text-to-speech model failed to load: {exc}")
    custom_engine = None

print("[2/2] Loading XTTS v2 voice cloning model.")
try:
    clone_engine = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device_name)
    print("XTTS v2 voice cloning model loaded.")
except Exception as exc:
    print(f"XTTS v2 voice cloning model failed to load: {exc}")
    clone_engine = None

print("\nServer is ready to accept inference requests.\n")


# ==============================================================================
# API routes
# ==============================================================================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/tts", methods=["POST"])
def generate_tts():
    if not custom_engine:
        return jsonify({"error": "Custom model is offline."}), 500

    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    speed = parse_speed(data.get("speed", 1.0))

    if not text:
        return jsonify({"error": "No text provided."}), 400

    output_path = os.path.join(UPLOAD_FOLDER, f"tts_{uuid.uuid4()}.wav")

    try:
        print(f"Processing text-to-speech request. Speed factor: {speed:.1f}x.")
        wav = custom_engine.tts(text)
        wav = apply_speed_to_waveform(wav, speed)
        custom_engine.save_wav(wav, output_path)

        return send_generated_audio(output_path, "tts_output.wav")
    except Exception as exc:
        print(f"Text-to-speech generation failed: {exc}")
        remove_if_exists(output_path)
        return jsonify({"error": str(exc)}), 500


@app.route("/api/clone", methods=["POST"])
def clone_voice():
    if not clone_engine:
        return jsonify({"error": "Cloning model failed to load."}), 500

    text = request.form.get("text", "").strip()
    file = request.files.get("audio_file")
    speed = parse_speed(request.form.get("speed", 1.0))

    if not text or not file:
        return jsonify({"error": "Missing text or audio file."}), 400

    ref_path = os.path.join(UPLOAD_FOLDER, f"ref_{uuid.uuid4()}.wav")
    output_path = os.path.join(UPLOAD_FOLDER, f"clone_{uuid.uuid4()}.wav")

    try:
        print(f"Processing voice cloning request. Speed factor: {speed:.1f}x.")
        file.save(ref_path)

        try:
            clone_engine.tts_to_file(
                text=text,
                speaker_wav=ref_path,
                language="en",
                file_path=output_path,
                speed=speed,
            )
        except TypeError:
            clone_engine.tts_to_file(
                text=text,
                speaker_wav=ref_path,
                language="en",
                file_path=output_path,
            )
            apply_speed_to_file(output_path, speed)

        remove_if_exists(ref_path)

        return send_generated_audio(output_path, "cloned_output.wav")
    except Exception as exc:
        print(f"Voice cloning generation failed: {exc}")
        remove_if_exists(ref_path)
        remove_if_exists(output_path)
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    # host="0.0.0.0" tells Flask to accept connections from the outside internet
    # port=7860 is required by Hugging Face Docker Spaces
    app.run(host="0.0.0.0", port=7860)