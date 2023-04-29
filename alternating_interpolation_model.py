import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from data_processing import load_data, get_datasets

import statsmodels.api as sm
from statsmodels.tsa.statespace.tools import diff
from itertools import product


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
features_dict = {"all": ALL,
                 "causal1": CAUSAL1,
                 "causal2": CAUSAL2,
                 "causal3": CAUSAL3}


def get_datasets_alt_interpolate(window_size, num_epochs=10, interpolation="ffill"):
    df_nan = load_data(group="day", interpolation=None)
    datasets_nan = get_datasets(df_nan, group="day")

    df_int = load_data(group="day", interpolation=interpolation)
    datasets_int = get_datasets(df_int, group="day")

    terms = ["covid", "covid_symptoms", "cough", "sore_throat", "fever", "flight", "vacation"]

    interpolate_columns = [f"{term}_query_index" for term in terms] \
                          + ["flight_seats_domestic", "flight_seats_international", "administered_cum"]
    np.random.shuffle(interpolate_columns)

    for strain, df in datasets_nan.items():
        df.reset_index(drop=True, inplace=True)

    for strain, df in datasets_int.items():
        df.reset_index(drop=True, inplace=True)
        for epoch in range(num_epochs):
            for col in interpolate_columns:
                df_tmp = df.drop("date", axis=1)
                X = []
                for i in range(len(df)):
                    if i < window_size:
                        vals = df_tmp.iloc[0:i].values
                        fill = np.ones((window_size - i, df_tmp.shape[1])) * -1
                        X.append(np.vstack([fill, vals]).flatten())
                    else:
                        X.append(df_tmp.iloc[i - window_size:i].values.flatten())
                X = np.stack(X)
                y = df[col]
                scaler = StandardScaler()
                X = scaler.fit_transform(X)
                model = LinearRegression()
                model.fit(X, y)
                fill_y = pd.Series(model.predict(X))
                df[col] = datasets_nan[strain][col].fillna(fill_y)

            delta = df["administered_cum"][1:].values - df["administered_cum"][:-1].values
            df["administered_raw"] = np.append(0, delta).astype(np.int64)

    return datasets_int


if __name__ == "__main__":
    model_name = "var"
    train_size = 0.25  # percentage of data for training only
    group = "day"
    forecast_steps = 21
    window_size = 7

    datasets = get_datasets_alt_interpolate(window_size=window_size, num_epochs=10, interpolation="ffill")

    for params_name, params in features_dict.items():
        reg_eq = "case_count ~ " + " + ".join(params)
        print(reg_eq)

        preds = []
        for strain, data in datasets.items():
            print(f"modeling strain {strain}")
            for train_idx in range(int(len(data) * train_size), len(data)):
                if group != "week":
                    if train_idx % 7 != 6:
                        continue
                train_data = data.iloc[:train_idx].reset_index()

                for hyp in param_grid[model_name]:
                    endog = train_data
                    old_endog_case_count = endog["case_count"].to_numpy()[-1]
                    endog["case_count"] = np.insert(diff(endog["case_count"].to_numpy(), k_diff=1), 0, 0)

                    model = sm.tsa.VAR(endog[params], missing="drop")
                    result = model.fit(maxlags=hyp)
                    lag_order = result.k_ar
                    pred = result.forecast(np.array(endog[params][-lag_order:]), steps=forecast_steps)
                    pred = old_endog_case_count + np.cumsum(pred[:, 0])
                    pred = pd.DataFrame(pred, columns=["predicted_mean"],
                                        index=range(train_idx, train_idx + forecast_steps))

                    pred.reset_index(names="forecast_index", inplace=True)
                    pred["strain"] = strain
                    pred["train_idx"] = train_idx
                    pred["hyp"] = [hyp] * forecast_steps
                    preds.append(pred)

        preds = pd.concat(preds).reset_index(names="steps")
        preds["steps"] += 1
        preds.to_csv(f"results/alternating/alternating_window{window_size}_{params_name}.csv", index=False)
