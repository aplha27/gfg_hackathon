import os
import json
import requests
from flask import Flask, request, jsonify, render_template, url_for
from dotenv import load_dotenv, find_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# Load environment variables
load_dotenv(find_dotenv())

app = Flask(__name__)

# Configuration
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")

if not all([ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID]):
    print("Warning: Missing environment variables. Please check .env file.")

# Load the RoBERTa model and tokenizer
# Note: Loading model globally means it stays in memory.
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load model: {e}")
    tokenizer = None
    model = None


def polarity_scores_roberta(text):
    if not tokenizer or not model:
        return {'roberta_neg': 0, 'roberta_neu': 0, 'roberta_pos': 0}
        
    encoded_text = tokenizer(text, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg': float(scores[0]),
        'roberta_neu': float(scores[1]),
        'roberta_pos': float(scores[2])
    }
    return scores_dict


def map_to_emotion(positive, negative, neutral):
    if positive > 0.6:
        if positive > 0.8:
            return "Exclamation"
        return "Happiness"
    elif negative > 0.6:
        if neutral > 0.4:
            return "Sadness"
        elif negative > 0.8:
            return "Distress"
        else:
            return "Anger"
    elif neutral > 0.6:
        return "Neutral"
    else:
        return "Mixed Emotion"


def get_emotion_from_text(text):
    scores = polarity_scores_roberta(text)
    emotion = map_to_emotion(scores['roberta_pos'], scores['roberta_neg'], scores['roberta_neu'])
    return scores, emotion


def generate_tts(text, emotion):
    if not ELEVENLABS_API_KEY or not ELEVENLABS_VOICE_ID:
        return None, "Missing API Key or Voice ID"

    # Emotion-based voice settings
    settings = {
        "Happiness": {"stability": 0.8, "similarity_boost": 0.9, "style": 0.7},
        "Sadness": {"stability": 0.5, "similarity_boost": 0.9, "style": 0.3},
        "Anger": {"stability": 0.7, "similarity_boost": 0.9, "style": 0.6},
        "Exclamation": {"stability": 0.6, "similarity_boost": 0.9, "style": 0.9},
        "Distress": {"stability": 0.4, "similarity_boost": 0.8, "style": 0.3},
        "Neutral": {"stability": 0.5, "similarity_boost": 0.5, "style": 0.5},
    }
    
    current_settings = settings.get(emotion, {"stability": 0.5, "similarity_boost": 0.5, "style": 0.5})

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/stream"
    headers = {
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY
    }
    data = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": current_settings["stability"],
            "similarity_boost": current_settings["similarity_boost"],
            "style": current_settings["style"],
            "use_speaker_boost": True
        }
    }

    try:
        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 200:
            # Ensure static directory exists
            static_dir = os.path.join(app.root_path, 'static')
            os.makedirs(static_dir, exist_ok=True)
            
            # Use a fixed name for simplicity as per original, or unique names in prod
            filename = "output_audio.mp3"
            audio_path = os.path.join(static_dir, filename)
            
            with open(audio_path, "wb") as audio_file:
                audio_file.write(response.content)
                
            # Return relative path for frontend
            return f"static/{filename}", None
        else:
            error_msg = f"ElevenLabs API Error: {response.status_code} - {response.text}"
            print(error_msg)
            return None, error_msg
    except Exception as e:
        error_msg = f"Error calling ElevenLabs API: {e}"
        print(error_msg)
        return None, error_msg


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    try:
        text = request.form.get('text')
        emotion = request.form.get('emotion')
        
        if not text or not emotion:
            return jsonify({'error': 'Missing text or emotion'}), 400

        scores, detected_emotion = get_emotion_from_text(text)
        audio_path, error = generate_tts(text, emotion)

        if error:
            return jsonify({'error': error}), 400

        return jsonify({
            'text': text,
            'scores': scores,
            'detected_emotion': detected_emotion,
            'emotion': emotion,
            'audio_path': audio_path
        })
    except Exception as e:
        print(f"Error in /generate route: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)

