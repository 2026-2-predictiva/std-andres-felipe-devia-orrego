import pickle

from sklearn import datasets
from sklearn.svm import SVC

FOLDER = "PRE_03_clasificacion_basica_imagenes"


def pregunta_01():
    data, target = datasets.load_digits(return_X_y=True)
    estimator = SVC(gamma=0.001)
    estimator.fit(data, target)
    with open(f"{FOLDER}/submission/estimator.pkl", "wb") as file:
        pickle.dump(estimator, file)


if __name__ == "__main__":
    pregunta_01()
