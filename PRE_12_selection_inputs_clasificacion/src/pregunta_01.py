import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FOLDER = "PRE_12_selection_inputs_clasificacion"
RANDOM_STATE = 12345

# cp, restecg, slope y thal son nominales; el resto se trata como numerico
CATEGORICAL = ["cp", "restecg", "slope", "thal"]
# Codigos validos segun la documentacion del dataset de Cleveland
THAL_VALUES = ["normal", "fixed", "reversible"]
CP_VALUES = [1, 2, 3, 4]
NUMERIC = [
    "age",
    "sex",
    "trestbps",
    "chol",
    "fbs",
    "thalach",
    "exang",
    "oldpeak",
    "ca",
]


def load_data():
    df = pd.read_csv(f"{FOLDER}/data/heart_disease.csv")
    df = df.dropna(subset=["target"])
    # Se descartan los registros con codigos invalidos: thal ("1" y "2", una
    # sola observacion cada uno) y cp (0, cuatro observaciones)
    df = df[df["thal"].isin(THAL_VALUES) & df["cp"].isin(CP_VALUES)].copy()
    df[CATEGORICAL] = df[CATEGORICAL].astype(str)
    y = df["target"]
    x = df[NUMERIC + CATEGORICAL]
    return x, y


def make_grid():
    preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", drop="first"),
                CATEGORICAL,
            ),
        ]
    )

    # SelectKBest (f_classif) hace parte del pipeline para que la seleccion de
    # entradas se reajuste en cada particion de la validacion cruzada
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("selector", SelectKBest(score_func=f_classif)),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    k_values = [2, 4, 6, 8, 10, 12, 15, 18, "all"]

    param_grid = [
        {
            "selector__k": k_values,
            "classifier": [LogisticRegression(max_iter=1000)],
            "classifier__C": [0.01, 0.1, 1.0, 10.0],
        },
        {
            "selector__k": k_values,
            "classifier": [RandomForestClassifier(random_state=RANDOM_STATE)],
            "classifier__n_estimators": [100, 300],
            "classifier__max_depth": [None, 4, 8],
            "classifier__min_samples_leaf": [1, 2, 4],
        },
    ]

    # El dataset es pequeno (~300 registros), por lo que se repite la
    # validacion cruzada estratificada para estabilizar la comparacion
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)

    return GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="balanced_accuracy",
        refit=True,
        n_jobs=-1,
    )


def selected_inputs(estimator):
    pipeline = estimator.best_estimator_
    names = pipeline["preprocessor"].get_feature_names_out()
    return list(names[pipeline["selector"].get_support()])


def compute_metrics(estimator, x, y, dataset):
    y_pred = estimator.predict(x)
    return {
        "dataset": dataset,
        "accuracy": accuracy_score(y, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "f1": f1_score(y, y_pred),
    }


def pregunta_01():
    x, y = load_data()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
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
    print(confusion_matrix(y_test, estimator.predict(x_test)))
    print(classification_report(y_test, estimator.predict(x_test)))

    with open(f"{FOLDER}/submission/estimator.pkl", "wb") as file:
        pickle.dump(estimator, file)


if __name__ == "__main__":
    pregunta_01()
