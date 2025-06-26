import re
import joblib
import spacy
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load spaCy for text cleaning
nlp = spacy.load("en_core_web_sm")
stop_words = nlp.Defaults.stop_words

# Load sentiment model & vectorizer
model = joblib.load("sentiment_nb_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-1B-distill")
chat_model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-1B-distill")


# Text cleaning
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'@\w+|#', '', text)
    text = re.sub(r"[^a-zA-Z\s]", '', text)
    text = text.lower()
    doc = nlp(text)
    return " ".join(token.lemma_ for token in doc if token.text not in stop_words and token.is_alpha)

# Map predicted labels to emoji
def get_sentiment_label(pred_label):
    return {
        0: "Negative 😠",
        1: "Neutral 😐",
        2: "Positive 😊"
    }.get(pred_label, "Unknown")

# Generate chatbot reply using Hugging Face
def generate_reply(user_input):
    try:
        inputs = tokenizer([user_input], return_tensors="pt")
        reply_ids = chat_model.generate(**inputs, max_length=100)
        reply = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
        return reply.strip() if reply.strip() else "Sorry, can you rephrase that?"
    except Exception as e:
        print("BlenderBot Error:", e)
        return "Sorry, I couldn't generate a response right now."


# Main route
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    if not user_input.strip():
        return jsonify({"reply": "Please say something!", "sentiment": "Unknown"})

    # Sentiment analysis
    clean = clean_text(user_input)
    vect = vectorizer.transform([clean])
    pred_label = model.predict(vect)[0]
    sentiment_str = get_sentiment_label(pred_label)

    # Chatbot reply
    reply = generate_reply(user_input)

    return jsonify({
        "sentiment": sentiment_str,
        "reply": reply
    })

if __name__ == "__main__":
    app.run(debug=True)
