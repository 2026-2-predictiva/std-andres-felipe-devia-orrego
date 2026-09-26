import pickle

import pandas as pd
from sklearn.linear_model import LinearRegression

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


def train_model():
    df = pd.read_csv(f"{FOLDER}/data/house_data.csv", sep=",")
    x = df[FEATURES]
    y = df["price"]

    estimator = LinearRegression()
    estimator.fit(x, y)

    with open(f"{FOLDER}/submission/house_predictor.pkl", "wb") as file:
        pickle.dump(estimator, file)


if __name__ == "__main__":
    train_model()
