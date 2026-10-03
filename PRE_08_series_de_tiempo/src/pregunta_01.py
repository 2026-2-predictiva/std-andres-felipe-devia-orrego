import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

FOLDER = "PRE_08_series_de_tiempo"
N_TEST = 24


def load_data():
    df = pd.read_csv(f"{FOLDER}/data/sutter.csv")
    df = df.set_index("date")
    df = df.assign(trend=list(range(len(df))))
    df = df.assign(month=df.index.str[5:7].astype(int))
    for period in [12, 6, 4, 3]:
        df[f"sin_{period}m"] = np.sin(2 * np.pi * df.month / period)
        df[f"cos_{period}m"] = np.cos(2 * np.pi * df.month / period)
    return df


def fit_predict(X, y):
    X_train, y_train = X.iloc[:-N_TEST], y.iloc[:-N_TEST]
    mask = X_train.notna().all(axis=1) & y_train.notna()
    model = LinearRegression().fit(X_train[mask], y_train[mask])
    pred = pd.Series(np.nan, index=X.index)
    valid = X.notna().all(axis=1)
    pred[valid] = model.predict(X[valid])
    return pred


def compute_metrics(df):
    results = {"Metrics": ["MSE Train", "MSE Test", "MAE Train", "MAE Test"]}
    for col in [c for c in df.columns if c.startswith("yt_pred")]:
        train = df.iloc[:-N_TEST][["yt_true", col]].dropna()
        test = df.iloc[-N_TEST:][["yt_true", col]].dropna()
        results[col] = [
            mean_squared_error(train.yt_true, train[col]),
            mean_squared_error(test.yt_true, test[col]),
            mean_absolute_error(train.yt_true, train[col]),
            mean_absolute_error(test.yt_true, test[col]),
        ]
    return pd.DataFrame(results).round(2)


def pregunta_01():
    df = load_data()
    y = df["yt_true"]

    # Tendencia lineal
    df["yt_pred_trend"] = fit_predict(df[["trend"]], y)

    # Tendencia lineal + variables dummy por mes
    months = pd.get_dummies(df.month, prefix="m", drop_first=True, dtype=float)
    df["yt_pred_trend_month"] = fit_predict(
        pd.concat([df[["trend"]], months], axis=1), y
    )

    # Tendencia lineal + componentes seno/coseno
    sincos = [c for c in df.columns if c.startswith(("sin_", "cos_"))]
    df["yt_pred_trend_sincos"] = fit_predict(df[["trend"] + sincos], y)

    # Modelo autorregresivo sobre la serie diferenciada (d1, d12),
    # pronostico a un paso reconstruido a la escala original
    d1d12 = y.diff(1).diff(12)
    lags = pd.concat({f"lag_{i}m": d1d12.shift(i) for i in range(1, 14)}, axis=1)
    d1d12_pred = fit_predict(lags, d1d12)
    df["yt_pred_ar_d1d12"] = y.shift(1) + y.shift(12) - y.shift(13) + d1d12_pred

    forecasts = df[[c for c in df.columns if c.startswith("yt_pred")] + ["yt_true"]]
    forecasts.to_csv(f"{FOLDER}/submission/forecasts.csv", index=True)

    compute_metrics(df).to_csv(f"{FOLDER}/submission/metrics.csv", index=False)


if __name__ == "__main__":
    pregunta_01()
