import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

FOLDER = "PRE_11_selection_inputs_regresion"
RANDOM_STATE = 12345

CATEGORICAL = ["Origin"]
NUMERIC = [
    "Cylinders",
    "Displacement",
    "Horsepower",
    "Weight",
    "Acceleration",
    "Model Year",
]


def load_data():
    df = pd.read_csv(f"{FOLDER}/data/auto_mpg.csv")
    df = df.dropna(subset=["MPG"])
    y = df["MPG"]
    x = df[NUMERIC + CATEGORICAL]
    return x, y


def make_grid():
    # Origin es nominal (1=USA, 2=Europa, 3=Japon): se codifica con one-hot,
    # mientras las variables numericas se imputan y escalan
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", drop="first"),
                CATEGORICAL,
            ),
        ]
    )

    # La seleccion de entradas (SelectKBest con f_regression) se ajusta dentro
    # de la validacion cruzada para no filtrar informacion del conjunto de prueba
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("expansion", PolynomialFeatures(include_bias=False)),
            ("selector", SelectKBest(score_func=f_regression)),
            ("regressor", LinearRegression()),
        ]
    )

    # El rango de k se acota al numero de columnas que genera cada grado
    param_grid = [
        {
            "expansion__degree": [1],
            "selector__k": [2, 4, 6, 8, "all"],
        },
        {
            "expansion__degree": [2, 3],
            "selector__k": [2, 4, 6, 8, 12, 16, 20, 30, "all"],
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


def selected_inputs(estimator):
    pipeline = estimator.best_estimator_
    names = pipeline[:-2].get_feature_names_out()
    return list(names[pipeline["selector"].get_support()])


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
    print("Entradas seleccionadas:", selected_inputs(estimator))
    print(metrics.to_string(index=False))

    with open(f"{FOLDER}/submission/estimator.pkl", "wb") as file:
        pickle.dump(estimator, file)


if __name__ == "__main__":
    pregunta_01()
