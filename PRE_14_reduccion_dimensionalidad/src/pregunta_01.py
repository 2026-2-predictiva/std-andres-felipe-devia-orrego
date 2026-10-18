import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from umap import UMAP

FOLDER = "PRE_14_reduccion_dimensionalidad"
RANDOM_STATE = 12345


def load_data():
    digits = load_digits()
    # Los 64 pixeles estan en la misma escala (0-16), pero se estandarizan
    # para que PCA no quede dominado por los pixeles del centro de la imagen
    x = StandardScaler().fit_transform(digits.data)
    return x, digits.target


def plot_embedding(embedding, y, title, filename):
    fig, ax = plt.subplots(figsize=(8, 7))

    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=y,
        cmap="tab10",
        s=8,
        alpha=0.8,
    )

    ax.set_title(title)
    ax.set_xlabel("Componente 1")
    ax.set_ylabel("Componente 2")
    ax.spines[["top", "right"]].set_visible(False)
    fig.colorbar(scatter, ax=ax, ticks=range(10), label="Digito")
    fig.tight_layout()
    fig.savefig(f"{FOLDER}/submission/{filename}", dpi=150)
    plt.close(fig)


def pregunta_01():
    x, y = load_data()

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    plot_embedding(
        pca.fit_transform(x),
        y,
        "Digits - PCA",
        "digits_pca.png",
    )
    print("Varianza explicada por PCA:", pca.explained_variance_ratio_.sum().round(4))

    tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=RANDOM_STATE)
    plot_embedding(
        tsne.fit_transform(x),
        y,
        "Digits - t-SNE",
        "digits_tsne.png",
    )

    umap = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=RANDOM_STATE)
    plot_embedding(
        umap.fit_transform(x),
        y,
        "Digits - UMAP",
        "digits_umap.png",
    )


if __name__ == "__main__":
    pregunta_01()
