import pickle

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FOLDER = "PRE_09_hiperparametros"
RANDOM_STATE = 12345


def load_data():
    df = pd.read_csv(f"{FOLDER}/data/winequality-red.csv")
    y = df["quality"]
    x = df.drop(columns=["quality"])
    return x, y


def make_grid():
    # El estimador del paso "regressor" tambien es un hiperparametro: la
    # busqueda compara tres familias de modelos junto con sus parametros
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", Ridge()),
        ]
    )

    param_grid = [
        {
            "regressor": [Ridge()],
            "regressor__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
        },
        {
            "regressor": [KNeighborsRegressor()],
            "regressor__n_neighbors": [5, 10, 20, 40],
            "regressor__weights": ["uniform", "distance"],
        },
        {
            "regressor": [RandomForestRegressor(random_state=RANDOM_STATE)],
            "regressor__n_estimators": [100, 300],
            "regressor__max_depth": [None, 8, 16],
            "regressor__min_samples_leaf": [1, 2, 4],
        },
    ]

    return GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        refit=True,
        n_jobs=-1,
    )


def compute_metrics(estimator, x, y, dataset):
    y_pred = estimator.predict(x)
    return {
        "dataset": dataset,
        "mse": mean_squared_error(y, y_pred),
        "mae": mean_absolute_error(y, y_pred),
        "r2": r2_score(y, y_pred),
    }


def pregunta_01():
    x, y = load_data()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=RANDOM_STATE
    )

    estimator = make_grid()
    estimator.fit(x_train, y_train)

    metrics = pd.DataFrame(
        [
            compute_metrics(estimator, x_train, y_train, "train"),
            compute_metrics(estimator, x_test, y_test, "test"),
        ]
    ).round(4)

    print(estimator.best_params_)
    print(metrics.to_string(index=False))

    with open(f"{FOLDER}/submission/estimator.pkl", "wb") as file:
        pickle.dump(estimator, file)


if __name__ == "__main__":
    pregunta_01()
