from data_processing import load_data, get_datasets

df_nan = load_data(group="day", interpolation=None)
datasets_nan = get_datasets(df_nan, group="day")

df_ffill = load_data(group="day", interpolation="ffill")
datasets_ffill = get_datasets(df_ffill, group="day")
