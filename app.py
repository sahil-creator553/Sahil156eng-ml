from flask import Flask, request, jsonify
from flask_cors import CORS
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download VADER lexicon if not already present
nltk.download("vader_lexicon")

app = Flask(__name__)
CORS(app)

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()


@app.route("/")
def home():
    return jsonify({
        "message": "Sentiment Analysis API is running!"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data or "text" not in data:
            return jsonify({
                "error": "Please provide a 'text' field."
            }), 400

        text = data["text"]

        scores = sia.polarity_scores(text)

        compound = scores["compound"]

        if compound >= 0.05:
            sentiment = "Positive"
        elif compound <= -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        confidence = round(max(
            scores["pos"],
            scores["neu"],
            scores["neg"]
        ) * 100, 2)

        return jsonify({
            "text": text,
            "sentiment": sentiment,
            "confidence": confidence,
            "scores": {
                "positive": scores["pos"],
                "neutral": scores["neu"],
                "negative": scores["neg"],
                "compound": scores["compound"]
            }
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)