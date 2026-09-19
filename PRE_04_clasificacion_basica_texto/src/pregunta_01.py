import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

FOLDER = "PRE_04_clasificacion_basica_texto"


def pregunta_01():
    dataframe = pd.read_csv(
        f"{FOLDER}/data/sentences.csv.zip",
        index_col=False,
        compression="zip",
    )
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    x = vectorizer.fit_transform(dataframe.phrase)
    clf = LogisticRegression(C=10, max_iter=1000)
    clf.fit(x, dataframe.target)
    with open(f"{FOLDER}/submission/clf.pkl", "wb") as file:
        pickle.dump(clf, file)
    with open(f"{FOLDER}/submission/vectorizer.pkl", "wb") as file:
        pickle.dump(vectorizer, file)


if __name__ == "__main__":
    pregunta_01()
