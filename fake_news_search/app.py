from flask import Flask, request, jsonify, render_template
from pydub import AudioSegment
import os
import tempfile
from sarvamai import SarvamAI

app = Flask(__name__)

def convert_audio_to_wav(input_path):
    try:
        audio = AudioSegment.from_file(input_path)
        wav_path = tempfile.mktemp(suffix=".wav")
        audio.export(wav_path, format="wav")
        return wav_path
    except Exception as e:
        print(f"Audio conversion failed: {e}")
        return None

@app.route("/")
def index():
    return render_template("index1.html")

@app.route("/process_audio", methods=["POST"])
def process_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    file = request.files['audio']
    input_path = tempfile.mktemp(suffix=".webm")
    file.save(input_path)
    print(f"📂 Saved: {input_path}")

    wav_path = convert_audio_to_wav(input_path)
    if not wav_path:
        return jsonify({"error": "Audio conversion failed"}), 500
    print(f"🎵 Converted to: {wav_path}")
    print(f"📏 File Size: {os.path.getsize(wav_path)} bytes")

    try:
        client = SarvamAI(api_subscription_key='474af18e-b1e7-4fba-b201-cc4b4f52ed3c')
        # Send using a named file to set correct MIME type
        with open(wav_path, "rb") as f:
            response = client.speech_to_text.transcribe(
                file=("recording.wav", f),
                model="saarika:v2.5",
                language_code="unknown"
            )
        transcript = response.transcript
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        return jsonify({"error": "Transcription failed", "details": str(e)}), 500
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(wav_path):
            os.remove(wav_path)

    return jsonify({"transcript": transcript})

if __name__ == "__main__":
    app.run(debug=True, port=5001)
