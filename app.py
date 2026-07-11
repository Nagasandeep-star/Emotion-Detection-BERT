from flask import Flask, render_template, request
from inference.predictor import predict_emotion

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        text = request.form["text"]

        emotion, confidence = predict_emotion(text)

        return render_template(
            "index.html",
            text=text,
            emotion=emotion,
            confidence=round(confidence * 100, 2)
        )

    except Exception as e:
        return render_template(
            "index.html",
            text=text if "text" in locals() else "",
            emotion=f"Error: {e}",
            confidence=0
        )

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )