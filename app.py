print("Step 1")

from flask import Flask, render_template, request
print("Step 2")

from inference.predictor import predict_emotion
print("Step 3")

app = Flask(__name__)
print("Step 4")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        text = request.form["text"]

        emotion = predict_emotion(text)

        return render_template(
            "index.html",
            text=text,
            emotion=emotion
        )

    except Exception as e:
        return render_template(
            "index.html",
            text=text if "text" in locals() else "",
            emotion=f"Error: {e}"
        )

print("Step 5")

if __name__ == "__main__":
    print("Starting Flask...")
    app.run(host="127.0.0.1", port=5000, debug=False)

print("Step 6")