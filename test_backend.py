import os
import requests
from dotenv import load_dotenv, find_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# Load Env
load_dotenv(find_dotenv())

def test_model():
    print("--- Testing Model ---")
    try:
        MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL)
        
        text = "I am so happy!"
        encoded_text = tokenizer(text, return_tensors='pt')
        output = model(**encoded_text)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        print(f"✅ Model succesful. Scores: {scores}")
        return True
    except Exception as e:
        print(f"❌ Model Failed: {e}")
        return False

def test_elevenlabs():
    print("\n--- Testing ElevenLabs API ---")
    APP_KEY = os.getenv("ELEVENLABS_API_KEY")
    VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
    
    if not APP_KEY or not VOICE_ID:
        print("❌ Missing Credentials in .env")
        return False

    print(f"Using API Key: {APP_KEY[:5]}...{APP_KEY[-5:]}")

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream"
    headers = {
        "Content-Type": "application/json",
        "xi-api-key": APP_KEY
    }
    data = {
        "text": "Hello, this is a test.",
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }
    
    try:
        print(f"Sending request to {url}...")
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            print(f"✅ API Success. Content length: {len(response.content)} bytes")
            return True
        else:
            print(f"❌ API Failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Exception during API call: {e}")
        return False

if __name__ == "__main__":
    test_elevenlabs()
