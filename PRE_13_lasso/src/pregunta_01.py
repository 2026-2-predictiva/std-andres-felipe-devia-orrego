import pickle

import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

FOLDER = "PRE_13_lasso"
RANDOM_STATE = 12345

TARGET = "mpg"
INPUTS = ["cyl", "disp", "hp", "drat", "wt", "qsec", "vs", "am", "gear", "carb"]


def load_data():
    df = pd.read_csv(f"{FOLDER}/data/mtcars.csv")
    y = df[TARGET]
    x = df[INPUTS]
    return x, y


def make_grid():
    # Lasso requiere entradas estandarizadas para que la penalizacion afecte
    # por igual a todos los coeficientes
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("expansion", PolynomialFeatures(include_bias=False)),
            ("regressor", Lasso(max_iter=50000, random_state=RANDOM_STATE)),
        ]
    )

    param_grid = {
        "expansion__degree": [1, 2],
        "regressor__alpha": [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
    }

    # mtcars solo tiene 32 registros: se usa validacion cruzada con pocos
    # pliegues para que cada particion de entrenamiento siga siendo util
    return GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        refit=True,
        n_jobs=-1,
    )


def selected_inputs(estimator):
    pipeline = estimator.best_estimator_
    names = pipeline[:-1].get_feature_names_out()
    coefs = pipeline["regressor"].coef_
    return [name for name, coef in zip(names, coefs) if coef != 0]


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
    print("Entradas con coeficiente no nulo:", selected_inputs(estimator))
    print(metrics.to_string(index=False))

    with open(f"{FOLDER}/submission/estimator.pkl", "wb") as file:
        pickle.dump(estimator, file)


if __name__ == "__main__":
    pregunta_01()
