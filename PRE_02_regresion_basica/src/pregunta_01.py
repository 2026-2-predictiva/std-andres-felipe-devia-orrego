"""
Entrena un MLPRegressor para predecir el consumo (MPG) del dataset auto_mpg
y guarda el modelo y el escalador de caracteristicas en la carpeta submission.
"""

import pickle

import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

FOLDER = "PRE_02_regresion_basica"


def pregunta_01():
    """
    Entrena el modelo y guarda los artefactos en submission/.
    """

    dataset = pd.read_csv(f"{FOLDER}/data/auto_mpg.csv")
    dataset = dataset.dropna()
    dataset["Origin"] = dataset["Origin"].map(
        {1: "USA", 2: "Europe", 3: "Japan"},
    )
    dataset = pd.get_dummies(dataset, columns=["Origin"], prefix="", prefix_sep="")

    y = dataset.pop("MPG")
    x = dataset

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    features_scaler = StandardScaler()
    x_train_scaled = features_scaler.fit_transform(x_train)
    x_test_scaled = features_scaler.transform(x_test)

    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=5000,
        random_state=42,
    )
    mlp.fit(x_train_scaled, y_train)

    mse = mean_squared_error(y_test, mlp.predict(x_test_scaled))

    with open(f"{FOLDER}/submission/mlp.pkl", "wb") as file:
        pickle.dump(mlp, file)

    with open(f"{FOLDER}/submission/features_scaler.pkl", "wb") as file:
        pickle.dump(features_scaler, file)

    return mse


if __name__ == "__main__":
    print(pregunta_01())
