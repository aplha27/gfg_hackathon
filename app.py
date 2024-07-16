import os
from flask import Flask, request, jsonify, render_template
import pymongo
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import requests
import json

app = Flask(__name__)

# MongoDB Configuration
try:
    mongo_client = pymongo.MongoClient("mongodb+srv://2713alpha8631:7s8vktQYGHh2Rw4@alpha0.a2ba0xa.mongodb.net/tts?retryWrites=true&w=majority")
    db = mongo_client['tts']
    collection = db['emotions']
    print("Connected to MongoDB successfully!")
except pymongo.errors.OperationFailure as e:
    print(f"OperationFailure: {e.details['errmsg']}")
except Exception as e:
    print(f"Exception: {e}")

# Load the RoBERTa model and tokenizer
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


def polarity_scores_roberta(text):
    encoded_text = tokenizer(text, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2]
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
    api_key = "sk_ae481476b824ee8b41d51eec8d999d75cf184b467411f17a"
    voice_id = "5Q0t7uMcjvnagumLfvZi"

    if emotion == "Happiness":
        stability = 0.8
        similarity_boost = 0.9
        style = 0.7
    elif emotion == "Sadness":
        stability = 0.5
        similarity_boost = 0.9
        style = 0.3
    elif emotion == "Anger":
        stability = 0.7
        similarity_boost = 0.9
        style = 0.6
    elif emotion == "Exclamation":
        stability = 0.6
        similarity_boost = 0.9
        style = 0.9
    elif emotion == "Distress":
        stability = 0.4
        similarity_boost = 0.8
        style = 0.3
    elif emotion == "Neutral":
        stability = 0.5
        similarity_boost = 0.5
        style = 0.5
    else:
        stability = 0.5
        similarity_boost = 0.5
        style = 0.5

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
    headers = {
        "Content-Type": "application/json",
        "xi-api-key": api_key
    }
    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity_boost,
            "style": style,
            "use_speaker_boost": True
        }
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        audio_path = os.path.join("static", "output_audio.mp3")
        with open(audio_path, "wb") as audio_file:
            audio_file.write(response.content)
        return audio_path
    else:
        return None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    try:
        text = request.form['text']
        emotion = request.form['emotion']
        scores, detected_emotion = get_emotion_from_text(text)
        audio_path = generate_tts(text, emotion)

        # Store data in MongoDB
        collection.insert_one({
            'text': text,
            'scores': scores,
            'detected_emotion': detected_emotion,
            'selected_emotion': emotion,
            'audio_path': audio_path
        })

        return jsonify({
            'text': text,
            'scores': scores,
            'detected_emotion': detected_emotion,
            'emotion': emotion,
            'audio_path': audio_path
        })
    except pymongo.errors.OperationFailure as e:
        print(f"OperationFailure in /generate route: {e.details['errmsg']}")
        return jsonify({'error': e.details['errmsg']}), 500
    except Exception as e:
        print(f"Error in /generate route: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)

