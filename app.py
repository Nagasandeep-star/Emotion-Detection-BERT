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

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)