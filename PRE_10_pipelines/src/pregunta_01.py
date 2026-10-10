import pickle

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

FOLDER = "PRE_10_pipelines"
RANDOM_STATE = 12345


def load_data():
    df = pd.read_csv(f"{FOLDER}/data/sentences.csv.zip", compression="zip")
    df = df.dropna()
    return df["phrase"], df["target"]


def make_grid():
    # El pipeline encadena vectorizacion, seleccion de terminos y
    # clasificacion; la busqueda ajusta los hiperparametros de los tres pasos
    pipeline = Pipeline(
        [
            ("vectorizer", CountVectorizer()),
            ("selector", SelectKBest(score_func=chi2)),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    param_grid = [
        {
            "vectorizer": [CountVectorizer(), TfidfVectorizer()],
            "vectorizer__ngram_range": [(1, 1), (1, 2)],
            "vectorizer__min_df": [1, 2],
            "selector__k": [500, 1000, "all"],
            "classifier": [LogisticRegression(max_iter=1000)],
            "classifier__C": [0.1, 1.0, 10.0],
        },
        {
            "vectorizer": [CountVectorizer(), TfidfVectorizer()],
            "vectorizer__ngram_range": [(1, 1), (1, 2)],
            "vectorizer__min_df": [1, 2],
            "selector__k": [500, 1000, "all"],
            "classifier": [MultinomialNB()],
            "classifier__alpha": [0.1, 0.5, 1.0],
        },
    ]

    return GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="balanced_accuracy",
        refit=True,
        n_jobs=-1,
    )


def pregunta_01():
    x, y = load_data()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    estimator = make_grid()
    estimator.fit(x_train, y_train)

    print(estimator.best_params_)
    print("Accuracy train:", round(accuracy_score(y_train, estimator.predict(x_train)), 4))
    print("Accuracy test: ", round(accuracy_score(y_test, estimator.predict(x_test)), 4))
    print(classification_report(y_test, estimator.predict(x_test)))

    with open(f"{FOLDER}/submission/estimator.pkl", "wb") as file:
        pickle.dump(estimator, file)


if __name__ == "__main__":
    pregunta_01()
