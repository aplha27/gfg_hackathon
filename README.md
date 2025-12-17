# Emotion TTS Generator (Flask)

A modern web application that detects emotions in text using the RoBERTa model and generates speech with corresponding emotional intonation using the ElevenLabs API.

## Features
- **Emotion Detection**: Uses `cardiffnlp/twitter-roberta-base-sentiment` to analyze text sentiment/emotion.
- **Adaptive TTS**: Automatically adjusts voice stability, similarity, and style based on the detected emotion (Happiness, Sadness, Anger, etc.).
- **Modern UI**: Clean, responsive interface built with HTML5, CSS3, and JavaScript.
- **Data Persistence**: Stores generation history in MongoDB.

## Prerequisites
- Python 3.8+
- [ElevenLabs API Key](https://elevenlabs.io/)
- [MongoDB Atlas URI](https://www.mongodb.com/atlas)

## Setup

1.  **Clone/Open the project**
    Navigate to the project directory in your terminal.

2.  **Create a Virtual Environment**
    It is recommended to use a virtual environment to manage dependencies.
    ```bash
    python -m venv venv
    ```

3.  **Activate the Virtual Environment**
    - **Windows (PowerShell):**
      ```powershell
      .\venv\Scripts\Activate
      ```
    - **Mac/Linux:**
      ```bash
      source venv/bin/activate
      ```

4.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Configure Environment**
    Ensure you have a `.env` file in the root directory with the following keys:
    ```ini
    MONGO_URI=your_mongodb_uri
    ELEVENLABS_API_KEY=your_api_key
    ELEVENLABS_VOICE_ID=your_voice_id
    ```

## Usage

1.  **Run the Application**
    ```bash
    python app.py
    ```

2.  **Access the Interface**
    Open your browser and navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000).

3.  **Generate Speech**
    - Enter text in the input box.
    - Select a target emotion (or let the AI detect it).
    - Click **Generate Audio** and listen to the result!

## Structure
- `app.py`: Main Flask application logic.
- `templates/index.html`: Frontend user interface.
- `static/`: Generated audio files.
- `.env`: Configuration secrets (not committed to git).

## License
[MIT](https://choosealicense.com/licenses/mit/)