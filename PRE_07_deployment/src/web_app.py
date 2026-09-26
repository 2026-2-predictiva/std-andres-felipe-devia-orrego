import pickle

import pandas as pd
from flask import Flask, render_template, request

FOLDER = "PRE_07_deployment"
FEATURES = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "condition",
]

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        data = {
            "bedrooms": float(request.form["bedrooms"]),
            "bathrooms": float(request.form["bathrooms"]),
            "sqft_living": float(request.form["sqft_living"]),
            "sqft_lot": float(request.form["sqft_lot"]),
            "floors": float(request.form["floors"]),
            "waterfront": 1 if request.form["waterfront"] == "Yes" else 0,
            "condition": int(request.form["condition"]),
        }
        with open(f"{FOLDER}/submission/house_predictor.pkl", "rb") as file:
            estimator = pickle.load(file)
        x = pd.DataFrame([data], columns=FEATURES)
        prediction = f"$ {estimator.predict(x)[0]:,.2f}"
    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
