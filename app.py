from flask import Flask, render_template, request
from inference.predictor import predict_emotion

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    text = request.form["text"]

    emotion = predict_emotion(text)

    return render_template(
        "index.html",
        text=text,
        emotion=emotion
    )


if __name__ == "__main__":
    app.run(
    host="127.0.0.1",
    port=5000,
    debug=False
)