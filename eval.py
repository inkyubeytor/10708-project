import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

from data_processing import load_data, get_datasets

if __name__ == "__main__":
    model_name = "arima"
    include_exog = True
    val_size = 0.25
    test_size = 0.25

    df = load_data()
    datasets = get_datasets(df)

    preds = pd.read_csv(f"results/{model_name}_predictions.csv")
    hyps = preds["hyp"].unique()

    for strain, data in datasets.items():
        val_idx = list(range(len(data)))[round(len(data) * (1 - val_size - test_size)):round(len(data) * (1 - test_size))]
        test_idx = list(range(len(data)))[round(len(data) * (1 - test_size)):]

        preds_val = preds[preds["strain"] == strain][preds["forecast_index"].isin(val_idx)]
        preds_test = preds[preds["strain"] == strain][preds["forecast_index"].isin(test_idx)]
        truth_val = data.iloc[val_idx]["case_count"]
        truth_test = data.iloc[test_idx]["case_count"]

        for step in [1, 2, 3]:
            errors = []
            for hyp in hyps:
                pred = preds_val[(preds_val["hyp"] == hyp) & (preds_val["steps"] == step)]["predicted_mean"]
                assert len(truth_val) == len(pred)
                pred = pred.fillna(-1e7)
                rmse = mean_squared_error(truth_val, pred, squared=False)
                errors.append(rmse)
            rmse_val = min(errors)
            best_hyp = hyps[np.argmin(errors)]

            pred_test = preds_test[(preds_test["hyp"] == best_hyp) & (preds_test["steps"] == step)]["predicted_mean"]
            rmse_test = mean_squared_error(truth_test, pred_test, squared=False)
            print(strain, step, best_hyp, rmse_val, rmse_test)
