# 10708-project

`data_processing.py` contains functions to process the raw data into pandas DataFrames.

To train the baseline and interpolation (except alternating) models, change `model_name` and `group` in `model.py` to the desired model (`"sma"`, `"arima"`, `"var"`) and data frequency (`"week"`, `"day"`) and run. To train the alternating interpolation models, do the same (and set `window_size` to desired value) in `alternating_interpolation_model.py`. To train the compartmental models, do the same in `comp_model.py`.

To run causal discovery, set `data` to the desired strain in `causal_discovery.py` and run.

To evaluate a model, pass in the path of the predictions csv to the relevant function (evaluate or evaluate_day) in `eval.py`.
