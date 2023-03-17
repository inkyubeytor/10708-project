import argparse
import statsmodels.api as sm
import patsy
from patsy.highlevel import dmatrices
from itertools import product

from data_processing import load_data, get_folds

ignore_na = patsy.missing.NAAction(NA_types=[])

if __name__ == "__main__":
    df = load_data()
    folds = get_folds(df)

    param_grid = {"arima": product([1, 2, 3], [1, 2], [1, 2])}

    for strain, data in folds.items():
        train_data, val_data, test_data = data[:round(0.6 * len(data))], \
            data[round(0.6 * len(data)):round(0.8 * len(data))], data[round(0.8 * len(data)):]

        reg_eq = "case_count ~ " + " + ".join(list(train_data.columns[2:].values))
        print(reg_eq)
        y, X = dmatrices(reg_eq, data=train_data, return_type="dataframe", NA_action=ignore_na)
        print(X.shape, y.shape)

        for p, d, q in param_grid["arima"]:
            model = sm.tsa.arima.ARIMA(y, exog=X, order=(p, d, q))
            # model = sm.tsa.VAR()
            result = model.fit()
            print(result.summary())
            print(result.mse)
            break

        break
