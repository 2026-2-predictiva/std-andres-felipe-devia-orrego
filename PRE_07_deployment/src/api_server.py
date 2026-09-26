import pickle

import pandas as pd
from flask import Flask, jsonify, request

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


@app.route("/", methods=["POST"])
def predict():
    with open(f"{FOLDER}/submission/house_predictor.pkl", "rb") as file:
        estimator = pickle.load(file)
    x = pd.DataFrame(request.get_json(), columns=FEATURES)
    return jsonify({"prediction": estimator.predict(x).tolist()})


if __name__ == "__main__":
    app.run(debug=True, port=5001)
