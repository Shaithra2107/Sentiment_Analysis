import os
import joblib
import re
import openai
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import spacy

# Load environment variables from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Flask app setup
app = Flask(__name__)
CORS(app)

# Load your trained model and vectorizer
model = joblib.load("sentiment_nb_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")
stop_words = nlp.Defaults.stop_words

def clean_text(text):
    # Basic cleaning + lemmatization, stopword removal
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'@\w+|#', '', text)
    text = re.sub(r"[^a-zA-Z\s]", '', text)
    text = text.lower()
    doc = nlp(text)
    return " ".join(token.lemma_ for token in doc if token.text not in stop_words and token.is_alpha)

def get_sentiment_label(pred_label):
    return {0: "Negative 😠", 1: "Neutral 😐", 2: "Positive 😊"}.get(pred_label, "Unknown")

def generate_gpt_reply(user_message, sentiment_label):
     # Return a canned reply when quota is exceeded or for testing
    return f"Sentiment detected: {sentiment_label}."
    prompt = f"""You are a friendly chatbot. A user just said: "{user_message}". 
The sentiment detected was: {sentiment_label}. 
Respond like a human chatbot would, keeping the sentiment in mind."""
    
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=100
    )
    return response.choices[0].message.content.strip()

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_input = request.json.get("message", "")
        if not user_input.strip():
            return jsonify({"reply": "Please say something!", "sentiment": "Unknown"})

        # Clean and predict sentiment
        clean = clean_text(user_input)
        vect = vectorizer.transform([clean])
        pred_label = model.predict(vect)[0]
        sentiment_str = get_sentiment_label(pred_label)

        # Get OpenAI GPT reply
        bot_reply = generate_gpt_reply(user_input, sentiment_str)

        return jsonify({
            "sentiment": sentiment_str,
            "reply": bot_reply
        })
    except Exception as e:
        print("Error:", e)
        return jsonify({"reply": "Sorry, something went wrong.", "sentiment": "Unknown"})

if __name__ == "__main__":
    app.run(debug=True)
