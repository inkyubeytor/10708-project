import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from data_processing import load_data, get_datasets


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

    return datasets_int


if __name__ == "__main__":
    model_name = "var"
    include_exog = True
    train_size = 0.25  # percentage of data for training only
    group = "day"
    forecast_steps = 21

    datasets = get_datasets_alt_interpolate(window_size=2, num_epochs=10)
    for strain, data in datasets.items():
        print(strain)
        print(data.isna().sum())
