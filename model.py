import argparse
import pandas as pd
import statsmodels.api as sm
import patsy
from patsy.highlevel import dmatrices
from itertools import product

from data_processing import load_data, get_datasets

ignore_na = patsy.missing.NAAction(NA_types=[])

param_grid = {"arima": list(product([1, 2, 3], [1, 2], [1, 2]))}


if __name__ == "__main__":
    model_name = "arima"
    include_exog = False
    include_exog_str = "includeX" if include_exog else "noX"
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

            for hyps in param_grid[model_name]:
                if model_name == "arima":
                    model = sm.tsa.arima.ARIMA(y, exog=X, order=hyps,
                                               missing="drop",
                                               enforce_stationarity=False, enforce_invertibility=False)
                    # filename = f"models/{model_name}/result_{model_name}_{strain}_train{train_idx}_order{hyps}.pkl"
                elif model_name == "var":
                    model = sm.tsa.VAR(y)
                    # filename = f"models/{model_name}/result_{model_name}_{include_exog_str}_{strain}_train{train_idx}_hyps{hyps}.pkl"
                else:
                    raise NotImplementedError
                result = model.fit()

                pred = result.forecast(3).to_frame()
                pred.reset_index(names="forecast_index", inplace=True)
                pred["strain"] = strain
                pred["train_idx"] = train_idx
                pred["hyp"] = [hyps] * 3
                preds.append(pred)

    preds = pd.concat(preds).reset_index(names="steps")
    preds["steps"] += 1
    preds.to_csv(f"results/{model_name}_predictions.csv", index=False)
