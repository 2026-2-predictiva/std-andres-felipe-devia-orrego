import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

FOLDER = "PRE_06_clustering_mercadeo"


def pregunta_01():
    df = pd.read_csv(f"{FOLDER}/data/snsdata.csv")

    df["gender"] = df["gender"].fillna("NA")
    df["female"] = (df["gender"] == "F").astype(int)
    df["no_gender"] = (df["gender"] == "NA").astype(int)

    df.loc[(df["age"] < 13) | (df["age"] >= 20), "age"] = None
    df["age"] = df["age"].fillna(df.groupby("gradyear")["age"].transform("mean"))

    interests = df.columns[4:40]
    scaled = StandardScaler().fit_transform(df[interests])

    kmeans = KMeans(n_clusters=5, n_init=10, random_state=0)
    df["cluster"] = kmeans.fit_predict(scaled)

    df.to_csv(f"{FOLDER}/submission/segmented.csv")


if __name__ == "__main__":
    pregunta_01()
