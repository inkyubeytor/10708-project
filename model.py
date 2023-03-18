import argparse
import pandas as pd
import numpy as np
import statsmodels.api as sm
import patsy
from patsy.highlevel import dmatrices
from itertools import product

from data_processing import load_data, get_datasets
from sma import SMA

ignore_na = patsy.missing.NAAction(NA_types=[])

param_grid = {"arima": list(product([1, 2, 3], [1, 2], [1, 2])),
              "var": [1, 2, 3],
              "sma": [1, 2, 3]}


if __name__ == "__main__":
    model_name = "var"
    include_exog = True
    train_size = 0.25  # percentage of data for training only

    df = load_data()
    datasets = get_datasets(df)

    reg_eq = "case_count ~ " + " + ".join(list(df.columns[2:].values))
    print(reg_eq)

    preds = []
    for strain, data in datasets.items():
        for train_idx in range(int(len(data) * train_size), len(data)):
            train_data = data.iloc[:train_idx].reset_index()
            y, X = dmatrices(reg_eq, data=train_data, return_type="dataframe", NA_action=ignore_na)
            if not include_exog:
                X = None

            for hyp in param_grid[model_name]:
                if model_name == "arima":
                    model = sm.tsa.arima.ARIMA(y, order=hyp,
                                               missing="drop",
                                               enforce_stationarity=False, enforce_invertibility=False)
                    result = model.fit()
                    pred = result.forecast(steps=3).to_frame()
                elif model_name == "var":
                    endog = y.to_frame() if X is None else y.join(X)
                    endog.drop("Intercept", axis=1, inplace=True, errors="ignore")

                    # model = sm.tsa.VAR(endog, missing="drop")
                    # result = model.fit(maxlags=hyp)
                    # lag_order = result.k_ar
                    # pred = result.forecast(np.array(endog[-lag_order:]), steps=3)
                    # pred = pred[:, 0]
                    # print(pred)

                    B_est = np.tile('E', (endog.shape[1], endog.shape[1]))
                    for i in range(B_est.shape[0]):
                        for j in range(B_est.shape[1]):
                            if i == j:
                                B_est[i, j] = 1
                            elif i < j:
                                B_est[i, j] = 0
                    endog_np = endog.to_numpy()
                    model = sm.tsa.SVAR(endog_np,
                                        svar_type="B", B=B_est,
                                        missing="drop")
                    result = model.fit(maxlags=hyp, B_guess=np.zeros(55).astype(int))
                    result.exog = None
                    result.coefs_exog = np.zeros(0)
                    result.trend = "c"
                    pred = result.forecast(endog_np[-hyp:], steps=3)
                    print(pred)

                elif model_name == "sma":
                    model = SMA(y["case_count"], hyp)
                    pred = model.forecast(3)
                else:
                    raise NotImplementedError

                pred.reset_index(names="forecast_index", inplace=True)
                pred["strain"] = strain
                pred["train_idx"] = train_idx
                pred["hyp"] = [hyp] * 3
                preds.append(pred)

    preds = pd.concat(preds).reset_index(names="steps")
    preds["steps"] += 1
    preds.to_csv(f"results/{model_name}_predictions.csv", index=False)
