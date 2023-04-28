import argparse
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.statespace.tools import diff
import patsy
from patsy.highlevel import dmatrices
from itertools import product

from data_processing import load_data, get_datasets
from sma import SMA

ignore_na = patsy.missing.NAAction(NA_types=[])

param_grid = {"arima": list(product([1, 2, 3], [1, 2], [1, 2])),
              "var": [1, 2, 3],
              "svar": [1, 2, 3],
              "sma": [1, 2, 3]}


CAUSAL1 = ['case_count', 'fever_query_index', 'sore_throat_query_index']
CAUSAL2 = ['case_count', 'covid_query_index', 'sore_throat_query_index', 'cough_query_index', 'fever_query_index', 'flight_query_index', 'administered_raw']
CAUSAL3 = ['case_count', 'covid_query_index', 'vacation_query_index', 'administered_raw']
ALL = [
    "case_count",
    "administered_raw",
    "covid_query_index",
    "covid_symptoms_query_index",
    "cough_query_index",
    "sore_throat_query_index",
    "fever_query_index",
    "flight_query_index",
    "vacation_query_index",
    "flight_seats_domestic",
    "flight_seats_international",
]

if __name__ == "__main__":
    model_name = "var"
    include_exog = True
    train_size = 0.25  # percentage of data for training only
    group = "day"
    forecast_steps = 3 if group == "week" else 21

    df = load_data(group=group, interpolation="ffill")
    datasets = get_datasets(df, group=group)

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
            if not include_exog:
                X = None

            for hyp in param_grid[model_name]:
                if model_name == "arima":
                    model = sm.tsa.arima.ARIMA(y, order=hyp,
                                               missing="drop",
                                               enforce_stationarity=False, enforce_invertibility=False)
                    result = model.fit()
                    pred = result.forecast(steps=forecast_steps).to_frame()
                elif model_name == "var":
                    endog = y.to_frame() if X is None else y.join(X)
                    endog.drop("Intercept", axis=1, inplace=True, errors="ignore")
                    old_endog_case_count = endog["case_count"].to_numpy()[-1]
                    endog["case_count"] = np.insert(diff(endog["case_count"].to_numpy(), k_diff=1), 0, 0)

                    params = CAUSAL3

                    model = sm.tsa.VAR(endog[params], missing="drop")
                    result = model.fit(maxlags=hyp)
                    lag_order = result.k_ar
                    pred = result.forecast(np.array(endog[params][-lag_order:]), steps=forecast_steps)
                    pred = old_endog_case_count + np.cumsum(pred[:, 0])
                    pred = pd.DataFrame(pred, columns=["predicted_mean"], index=range(train_idx, train_idx+forecast_steps))
                elif model_name == "svar":
                    endog = y.to_frame() if X is None else y.join(X)
                    endog.drop("Intercept", axis=1, inplace=True, errors="ignore")

                    old_endog_case_count = endog["case_count"].to_numpy()[-1]
                    endog["case_count"] = np.insert(diff(endog["case_count"].to_numpy(), k_diff=1), 0, 0)

                    params = [
                        "case_count",
                        "administered_raw",
                        "covid_query_index",
                        "covid_symptoms_query_index",
                        "cough_query_index",
                        "sore_throat_query_index",
                        "fever_query_index",
                        "flight_query_index",
                        "vacation_query_index",
                        "flight_seats_domestic",
                        "flight_seats_international",
                    ]
                    endog = endog[params]

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
                    result = model.fit(maxlags=hyp)
                    result.exog = None
                    result.coefs_exog = np.zeros(0)
                    result.trend = "c"
                    pred = result.forecast(endog_np[-hyp:], steps=forecast_steps)
                    pred = pd.DataFrame(pred[:, 0], columns=["predicted_mean"], index=range(train_idx, train_idx + forecast_steps))
                elif model_name == "sma":
                    model = SMA(y["case_count"], hyp)
                    pred = model.forecast(forecast_steps)
                else:
                    raise NotImplementedError

                pred.reset_index(names="forecast_index", inplace=True)
                pred["strain"] = strain
                pred["train_idx"] = train_idx
                pred["hyp"] = [hyp] * forecast_steps
                preds.append(pred)

    preds = pd.concat(preds).reset_index(names="steps")
    preds["steps"] += 1
    preds.to_csv(f"results/{model_name}_{group}_predictions.csv", index=False)
