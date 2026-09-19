import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

FOLDER = "PRE_05_clustering_demanda"


def pregunta_01():
    df = pd.read_csv(f"{FOLDER}/data/demanda_comercial.csv.zip", compression="zip")
    df = df.dropna()
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    df = df.set_index("Fecha")

    plt.figure(figsize=(12, 4))
    plt.plot(df.index, df.sum(axis=1))
    plt.title("Demanda comercial diaria")
    plt.tight_layout()
    plt.savefig(f"{FOLDER}/submission/demanda-comercial.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    for _, row in df.sample(10, random_state=0).iterrows():
        plt.plot(range(1, 25), row.values, alpha=0.7)
    plt.title("Patrones de ejemplo")
    plt.xlabel("Hora")
    plt.tight_layout()
    plt.savefig(f"{FOLDER}/submission/demanda-comercial-patrones-ejemplo.png")
    plt.close()

    normalized = df.div(df.sum(axis=1), axis=0)
    kmeans = KMeans(n_clusters=4, n_init=10, random_state=0)
    labels = kmeans.fit_predict(normalized)

    plt.figure(figsize=(8, 5))
    for k, center in enumerate(kmeans.cluster_centers_):
        plt.plot(range(1, 25), center, label=f"Perfil {k} (n={(labels == k).sum()})")
    plt.title("Perfiles de demanda")
    plt.xlabel("Hora")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{FOLDER}/submission/demanda-comercial-perfiles.png")
    plt.close()


if __name__ == "__main__":
    pregunta_01()
