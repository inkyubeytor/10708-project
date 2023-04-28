import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

from data_processing import load_data, get_datasets


def evaluate(model_name, datasets, include_exog=True, val_size=0.25, test_size=0.25):
    preds = pd.read_csv(f"results/{model_name}_week_predictions.csv")
    hyps = preds["hyp"].unique()

    results = []
    for strain, data in datasets.items():
        val_idx = list(range(len(data)))[
                  round(len(data) * (1 - val_size - test_size)):round(len(data) * (1 - test_size))]
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
                # err = mean_squared_error(truth_val, pred, squared=False)
                err = mean_absolute_percentage_error(truth_val, pred)
                errors.append(err)
            err_val = min(errors)
            best_hyp = hyps[np.argmin(errors)]

            pred_test = preds_test[(preds_test["hyp"] == best_hyp) & (preds_test["steps"] == step)]["predicted_mean"]
            # err_test = mean_squared_error(truth_test, pred_test, squared=False)
            err_test = mean_absolute_percentage_error(truth_test, pred_test)
            result = {"strain": strain,
                      "step": step,
                      "best_hyp": best_hyp,
                      "err_val": err_val,
                      "err_test": err_test}
            results.append(result)

    return pd.DataFrame(results)


def evaluate_day(model_name, datasets, include_exog=True, val_size=0.25, test_size=0.25, path=None):
    if path is None:
        preds = pd.read_csv(f"results/{model_name}_{group}_predictions.csv")
    else:
        preds = pd.read_csv(path)
    hyps = preds["hyp"].unique()
    steps = [7, 14, 21]

    results = []
    for strain, data in datasets.items():
        data = data[["case_count", "death_count"]]

        val_idx = list(range(len(data)))[
                  round(len(data) * (1 - val_size - test_size)):round(len(data) * (1 - test_size))]
        test_idx = list(range(len(data)))[round(len(data) * (1 - test_size)):]
        val_idx = [list(range(i - 6, i + 1)) for i in val_idx if i % 7 == 5]
        test_idx = [list(range(i - 6, i + 1)) for i in test_idx if i % 7 == 5]

        preds_val = []
        truth_val = []
        for idx_set in val_idx:
            preds_val.append(preds[preds["strain"] == strain][preds["forecast_index"].isin(idx_set)]
                             .groupby(by=["strain", "train_idx", "hyp"])
                             .agg({"steps": "max", "forecast_index": "max", "predicted_mean": "sum"})
                             .reset_index())
            truth_val.append(data.iloc[idx_set]["case_count"].sum())
        preds_val = pd.concat(preds_val, ignore_index=True)
        truth_val = np.array(truth_val)

        preds_test = []
        truth_test = []
        for idx_set in test_idx:
            preds_test.append(preds[preds["strain"] == strain][preds["forecast_index"].isin(idx_set)]
                              .groupby(by=["strain", "train_idx", "hyp"])
                              .agg({"steps": "max", "forecast_index": "max", "predicted_mean": "sum"})
                              .reset_index())
            truth_test.append(data.iloc[idx_set]["case_count"].sum())
        preds_test = pd.concat(preds_test, ignore_index=True)
        truth_test = np.array(truth_test)

        for step in steps:
            errors = []
            for hyp in hyps:
                pred = preds_val[(preds_val["hyp"] == hyp) & (preds_val["steps"] == step)]["predicted_mean"]
                assert len(truth_val) == len(pred)
                pred = pred.fillna(-1e7)
                # err = mean_squared_error(truth_val, pred, squared=False)
                err = mean_absolute_percentage_error(truth_val, pred)
                errors.append(err)
            err_val = min(errors)
            best_hyp = hyps[np.argmin(errors)]

            pred_test = preds_test[(preds_test["hyp"] == best_hyp) & (preds_test["steps"] == step)]["predicted_mean"]
            # err_test = mean_squared_error(truth_test, pred_test, squared=False)
            err_test = mean_absolute_percentage_error(truth_test, pred_test)
            result = {"strain": strain,
                      "step": step,
                      "best_hyp": best_hyp,
                      "err_val": err_val,
                      "err_test": err_test}
            results.append(result)

    return pd.DataFrame(results)


if __name__ == "__main__":
    group = "week"

    df = load_data(group=group)
    datasets = get_datasets(df, group=group)

    results = evaluate("var", datasets)
    print(" ".join(map(str, results["err_test"])))
