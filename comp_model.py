import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.statespace.tools import diff
import patsy
from patsy.highlevel import dmatrices

from data_processing import load_data_cumulative, get_datasets_cumulative

ignore_na = patsy.missing.NAAction(NA_types=[])

param_grid = [1, 2, 3]


def make_m1(df):
    df["case_squared"] = df["case_count_cumulative"] ** 2
    df["case_x_death"] = df["case_count_cumulative"] * df["death_count_cumulative"]
    df = df[["case_count", "case_count_cumulative", "death_count_cumulative", "case_squared", "case_x_death"]]
    return df


def make_m1_stationary(df):
    df = make_m1(df)

    for row in ["case_count_cumulative", "death_count_cumulative", "case_squared", "case_x_death"]:
        df[row] = np.insert(diff(df[row].to_numpy(), k_diff=1), 0, 0)
    return df


def make_m2(df):
    df["case_squared"] = df["case_count_cumulative"] ** 2
    df["d^2/c"] = df["death_count_cumulative"] ** 2 / df["case_count_cumulative"]
    df["death_squared"] = df["death_count_cumulative"] ** 2
    df = df[["case_count", "case_count_cumulative", "case_squared", "d^2/c", "death_squared"]]
    return df


def make_m2_stationary(df):
    df = make_m2(df)

    for row in ["case_count_cumulative", "case_squared", "d^2/c", "death_squared"]:
        df[row] = np.insert(diff(df[row].to_numpy(), k_diff=1), 0, 0)
    return df


N = 332_000_000
initial_d = 0.05


def compute_a(df):
    C_cumulative = df["case_count_cumulative"].to_numpy()[:-1]
    C = df["case_count"].to_numpy()[1:]
    D_cumulative = df["death_count_cumulative"].to_numpy()[:-1]
    d = np.ones_like(C) * initial_d
    a = 0
    steps = 0
    while True:
        old_a = a
        old_d = d
        denom = (N - C_cumulative) * (C_cumulative - D_cumulative / d)
        nonzero = np.nonzero(denom)[0]
        # print("Nonzero count:", len(nonzero))
        A = C[nonzero] / denom[nonzero]
        a = A.mean()
        d = D_cumulative / (C_cumulative - (C / (a * (N - C_cumulative))))
        # print(d)
        # print(a)

        steps += 1
        if abs(old_a - a) <= 1e-10 and np.linalg.norm(old_d - d) <= 1e-10:
            break
    # print(f"Steps taken: {steps}")
    return a


def make_m3(df):
    a = compute_a(df)
    df["case_squared"] = df["case_count_cumulative"] ** 2
    dt = df["death_count_cumulative"] / (df["case_count_cumulative"] - (df["case_count"] / (a * (N - df["case_count_cumulative"]))))
    df["D/d"] = df["death_count_cumulative"] / dt
    df["CD/d"] = df["D/d"] * df["case_count_cumulative"]
    df = df[["case_count", "case_count_cumulative", "case_squared", "D/d", "CD/d"]]
    return df


def make_m3_stationary(df):
    df = make_m3(df)

    for row in ["case_count_cumulative", "case_squared", "D/d", "CD/d"]:
        df[row] = np.insert(diff(df[row].to_numpy(), k_diff=1), 0, 0)
    return df


def make_m4(df):
    a = compute_a(df)
    df["dt"] = df["death_count_cumulative"] / (df["case_count_cumulative"] - (df["case_count"] / (a * (N - df["case_count_cumulative"]))))
    df = df[["case_count", "case_count_cumulative", "death_count_cumulative", "dt"]]
    return df


def make_m4_stationary(df):
    df = make_m4(df)

    for row in ["case_count_cumulative", "death_count_cumulative", "dt"]:
        df[row] = np.insert(diff(df[row].to_numpy(), k_diff=1), 0, 0)
    return df


if __name__ == "__main__":
    model_name = "var"
    train_size = 0.25  # percentage of data for training only
    group = "day"
    forecast_steps = 3 if group == "week" else 21

    df = load_data_cumulative()
    datasets = get_datasets_cumulative(df)

    features = list(df.columns[2:].values)  # explicitly specify this
    reg_eq = "case_count ~ " + " + ".join(features)
    print(reg_eq)

    preds = []
    for strain, data in datasets.items():
        print(f"modeling strain {strain}")
        for train_idx in range(int(len(data) * train_size), len(data)):
            if train_idx % 7 != 6:
                continue
            train_data = data.iloc[:train_idx].reset_index()
            y, X = dmatrices(reg_eq, data=train_data, return_type="dataframe", NA_action=ignore_na)

            for hyp in param_grid:
                endog = y.to_frame() if X is None else y.join(X)
                endog.drop("Intercept", axis=1, inplace=True, errors="ignore")
                old_endog_case_count = endog["case_count"].to_numpy()[-1]
                endog["case_count"] = np.insert(diff(endog["case_count"].to_numpy(), k_diff=1), 0, 0)

                endog = make_m4_stationary(endog)  # THIS LINE CHOOSES METHOD

                model = sm.tsa.VAR(endog, missing="drop")
                result = model.fit(maxlags=hyp)
                lag_order = result.k_ar
                pred = result.forecast(np.array(endog[-lag_order:]), steps=forecast_steps)
                pred = old_endog_case_count + np.cumsum(pred[:, 0])
                pred = pd.DataFrame(pred, columns=["predicted_mean"], index=range(train_idx, train_idx+forecast_steps))

                pred.reset_index(names="forecast_index", inplace=True)
                pred["strain"] = strain
                pred["train_idx"] = train_idx
                pred["hyp"] = [hyp] * forecast_steps
                preds.append(pred)

    preds = pd.concat(preds).reset_index(names="steps")
    preds["steps"] += 1
    preds.to_csv(f"results/{model_name}_{group}_predictions.csv", index=False)
