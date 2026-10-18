import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from sklearn.cluster import AffinityPropagation
from sklearn.covariance import GraphicalLassoCV
from sklearn.manifold import LocallyLinearEmbedding

FOLDER = "PRE_15_estructura_mercado"

SYMBOLS = {
    "AAPL": "Apple",
    "AIG": "AIG",
    "AMZN": "Amazon",
    "AXP": "American express",
    "BA": "Boeing",
    "BAC": "Bank of America",
    "CAJ": "Canon",
    "CAT": "Caterpillar",
    "CL": "Colgate-Palmolive",
    "CMCSA": "Comcast",
    "COP": "ConocoPhillips",
    "CSCO": "Cisco",
    "CVC": "Cablevision",
    "CVS": "CVS",
    "CVX": "Chevron",
    "DD": "DuPont de Nemours",
    "DELL": "Dell",
    "F": "Ford",
    "GD": "General Dynamics",
    "GE": "General Electrics",
    "GS": "Goldman Sachs",
    "GSK": "GlaxoSmithKline",
    "HD": "Home Depot",
    "HMC": "Honda",
    "HPQ": "HP",
    "IBM": "IBM",
    "JPM": "JPMorgan Chase",
    "K": "Kellogg",
    "KMB": "Kimberly-Clark",
    "KO": "Coca Cola",
    "MAR": "Marriott",
    "MCD": "McDonald's",
    "MMM": "3M",
    "MSFT": "Microsoft",
    "NAV": "Navistar",
    "NOC": "Northrop Grumman",
    "NVS": "Novartis",
    "PEP": "Pepsi",
    "PFE": "Pfizer",
    "PG": "Procter Gamble",
    "R": "Ryder",
    "RTN": "Raytheon",
    "SAP": "SAP",
    "SNE": "Sony",
    "SNY": "Sanofi-Aventis",
    "TM": "Toyota",
    "TOT": "Total",
    "TWX": "Time Warner",
    "TXN": "Texas Instruments",
    "UN": "Unilever",
    "VLO": "Valero Energy",
    "WFC": "Wells Fargo",
    "WMT": "Wal-Mart",
    "XOM": "Exxon",
    "XRX": "Xerox",
    "YHOO": "Yahoo",
}


def load_data():
    # La estructura del mercado se estima sobre la variacion diaria de cada
    # accion (cierre - apertura), no sobre el precio
    variations = {}
    for symbol in SYMBOLS:
        df = pd.read_csv(f"{FOLDER}/data/{symbol}.csv.zip", index_col="date")
        variations[symbol] = df["close"] - df["open"]

    variations = pd.DataFrame(variations).dropna()

    # Cada serie se lleva a varianza unitaria para que las acciones con
    # precios altos no dominen la covarianza
    x = variations.to_numpy()
    x = x / x.std(axis=0)
    return x, variations.columns.to_numpy()


def build_model(x):
    # GraphicalLassoCV estima una covarianza inversa dispersa: cada arco que
    # sobrevive indica correlacion condicional entre dos acciones
    edge_model = GraphicalLassoCV()
    edge_model.fit(x)

    # Affinity propagation agrupa las acciones sin fijar el numero de grupos
    labels = AffinityPropagation(random_state=0).fit_predict(edge_model.covariance_)

    # El embedding 2D se calcula sobre las correlaciones para ubicar en el
    # plano las acciones que se mueven juntas
    embedding = LocallyLinearEmbedding(
        n_components=2,
        eigen_solver="dense",
        n_neighbors=6,
    ).fit_transform(x.T)

    return edge_model, labels, embedding.T


def plot_market(edge_model, labels, embedding, names):
    n_labels = labels.max() + 1

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_axis_off()

    partial_correlations = -edge_model.precision_.copy()
    d = 1 / np.sqrt(np.diag(edge_model.precision_))
    partial_correlations *= d
    partial_correlations *= d[:, np.newaxis]
    non_zero = np.abs(np.triu(partial_correlations, k=1)) > 0.02

    ax.scatter(
        embedding[0],
        embedding[1],
        s=100 * d**2,
        c=labels,
        cmap="nipy_spectral",
    )

    start_idx, end_idx = np.where(non_zero)
    segments = [
        [embedding[:, start], embedding[:, stop]]
        for start, stop in zip(start_idx, end_idx)
    ]
    values = np.abs(partial_correlations[non_zero])
    lines = LineCollection(
        segments,
        zorder=0,
        cmap="hot_r",
        norm=plt.Normalize(0, 0.7 * values.max()),
    )
    lines.set_array(values)
    lines.set_linewidths(15 * values)
    ax.add_collection(lines)

    for index, (name, (x, y)) in enumerate(zip(names, embedding.T)):
        dx = x - embedding[0]
        dx[index] = 1
        dy = y - embedding[1]
        dy[index] = 1
        this_dx = dx[np.argmin(np.abs(dy))]
        this_dy = dy[np.argmin(np.abs(dx))]

        horizontalalignment = "left" if this_dx > 0 else "right"
        x = x + 0.002 if this_dx > 0 else x - 0.002
        verticalalignment = "bottom" if this_dy > 0 else "top"
        y = y + 0.002 if this_dy > 0 else y - 0.002

        ax.text(
            x,
            y,
            SYMBOLS[name],
            size=10,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            bbox=dict(
                facecolor="w",
                edgecolor=plt.cm.nipy_spectral(labels[index] / float(n_labels)),
                alpha=0.6,
            ),
        )

    ax.set_xlim(
        embedding[0].min() - 0.15 * np.ptp(embedding[0]),
        embedding[0].max() + 0.10 * np.ptp(embedding[0]),
    )
    ax.set_ylim(
        embedding[1].min() - 0.03 * np.ptp(embedding[1]),
        embedding[1].max() + 0.03 * np.ptp(embedding[1]),
    )
    ax.set_title("Estructura del mercado")

    fig.tight_layout()
    fig.savefig(f"{FOLDER}/submission/stocks.png", dpi=150)
    plt.close(fig)


def pregunta_01():
    x, names = load_data()
    edge_model, labels, embedding = build_model(x)

    for label in range(labels.max() + 1):
        members = [SYMBOLS[name] for name in names[labels == label]]
        print(f"Grupo {label + 1}: {', '.join(members)}")

    plot_market(edge_model, labels, embedding, names)


if __name__ == "__main__":
    pregunta_01()
