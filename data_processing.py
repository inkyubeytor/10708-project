import numpy as np
import pandas as pd
import datetime
from functools import reduce


def load_data(group="week", interpolation=None):
    df_covid = pd.merge(load_covid_data(count_type="case"),
                        load_covid_data(count_type="death"),
                        on="date")

    terms = ["covid", "covid_symptoms", "cough", "sore_throat", "fever", "flight", "vacation"]
    df_search = reduce(lambda x, y: pd.merge(x, y, on="week_start"), [load_search_data(term) for term in terms])

    df_flight = pd.merge(load_flight_data(terminal="domestic"),
                         load_flight_data(terminal="international"),
                         on=["year", "week"])

    df_vaccination = load_vaccination_data(interpolation=interpolation)

    if group == "week":
        # standardize dates using ISO calendar weeks (start on Monday)
        # note that this is not the same as MMWR weeks used by the CDC (start on Sunday)
        # also note that flight data weeks are ambiguous (no indication of dates by the source)
        df_covid["year"] = df_covid["date"].dt.isocalendar().year
        df_covid["week"] = df_covid["date"].dt.isocalendar().week
        df_search["year"] = df_search["week_start"].dt.isocalendar().year
        df_search["week"] = df_search["week_start"].dt.isocalendar().week
        df_vaccination["year"] = df_vaccination["Date"].dt.isocalendar().year
        df_vaccination["week"] = df_vaccination["Date"].dt.isocalendar().week

        # sum COVID counts and vaccine doses administered by week
        df_covid = df_covid.groupby(["year", "week"]).sum(numeric_only=True)
        df_vaccination = df_vaccination.groupby(["year", "week"]).sum(numeric_only=True)
        df_search.set_index(["year", "week"], inplace=True)
        df_search.drop(["week_start"], axis=1, inplace=True)
        df_flight.set_index(["year", "week"], inplace=True)

        df = df_covid.join(df_search).join(df_flight).join(df_vaccination)
    elif group == "day":
        df_vaccination.rename(columns={"Date": "date"}, inplace=True)

        df_search["week_end"] = df_search["week_start"] + datetime.timedelta(days=6)
        df_flight["week_end"] = df_flight.apply(lambda x: datetime.date.fromisocalendar(x.year, x.week, 7), axis=1) \
                                         .astype('datetime64[ns]')

        if interpolation is None:
            df_search["date"] = df_search["week_end"]
            df_flight["date"] = df_flight["week_end"]
        elif interpolation == "ffill":
            df_search["date"] = df_search["week_end"].apply(lambda d: [d + datetime.timedelta(days=i+1) for i in range(7)])
            df_search = df_search.explode("date", ignore_index=True)

            df_flight["date"] = df_flight["week_end"].apply(lambda d: [d + datetime.timedelta(days=i+1) for i in range(7)])
            df_flight = df_flight.explode("date", ignore_index=True)
        elif interpolation == "linear":
            # new_index_search = pd.date_range(start=df_search["week_end"].min(),
            #                                  end=df_search["week_end"].max(),
            #                                  name="date")
            # df_search = df_search.set_index("week_end").reindex(new_index_search).reset_index()
            # df_search_cols = df_search.select_dtypes(include=np.number)
            # for col in df_search_cols:
            #     df_search[col] = df_search[col].interpolate()
            #
            # new_index_flight = pd.date_range(start=df_flight["week_end"].min(),
            #                                  end=df_flight["week_end"].max(),
            #                                  name="date")
            # df_flight = df_flight.set_index("week_end").reindex(new_index_flight).reset_index()
            # df_flight_cols = df_flight.select_dtypes(include=np.number)
            # for col in df_flight_cols:
            #     df_flight[col] = df_flight[col].interpolate()
            #
            # df_search.drop(["week_start"], axis=1, inplace=True)
            # df_flight.drop(["year", "week"], axis=1, inplace=True)
            raise NotImplementedError
        else:
            raise NotImplementedError

        df_search.drop(["week_start", "week_end"], axis=1, inplace=True)
        df_flight.drop(["year", "week", "week_end"], axis=1, inplace=True)

        df = df_covid.merge(df_vaccination, on="date", how="left") \
                     .merge(df_search, on="date", how="left") \
                     .merge(df_flight, on="date", how="left")
        df.set_index("date")
    else:
        raise NotImplementedError

    df["administered_raw"] = df["administered_raw"].fillna(value=0, limit=326)
    df["administered_cum"] = df["administered_cum"].fillna(value=0, limit=326)
    return df


def get_datasets(df, split="strain", group="week"):
    if split == "strain":
        if group == "week":
            folds = {"alpha": df.iloc[:75], "delta": df.iloc[75:99], "omicron": df.iloc[99:]}
        elif group == "day":
            folds = {"alpha": df.iloc[:522], "delta": df.iloc[522:690], "omicron": df.iloc[690:]}
        else:
            raise NotImplementedError
        return folds
    else:
        raise NotImplementedError


def load_covid_data(count_type="case", path=""):
    # default filepaths
    if not path:
        if count_type == "case":
            path = "data/covid/truth-Incident Cases.csv"
        elif count_type == "death":
            path = "data/covid/truth-Incident Deaths.csv"

    df = pd.read_csv(path, parse_dates=["date"])
    df = df[df["location"] == "US"].reset_index(drop=True)  # only use nationwide counts
    df.drop(["location", "location_name"], axis=1, inplace=True)
    df.rename(columns={"value": f"{count_type}_count"}, inplace=True)
    return df


# topic trends (as opposed to search term trends)
def load_search_data(term, path=""):
    query_index_name = f"{term}_query_index"

    # definitely not elegant but oh well
    if not path:
        try:
            path = f"data/search_trends/{term}_topic.csv"
            df = pd.read_csv(path, skiprows=3, names=["week_start", query_index_name], parse_dates=["week_start"])
        except FileNotFoundError:
            path = f"data/search_trends/{term}_searchterm.csv"
            df = pd.read_csv(path, skiprows=3, names=["week_start", query_index_name], parse_dates=["week_start"])
    else:
        df = pd.read_csv(path, skiprows=3, names=["week_start", query_index_name], parse_dates=["week_start"])

    df.replace("<1", "0.5", inplace=True)
    df[query_index_name] = pd.to_numeric(df[query_index_name])
    return df


def load_flight_data(terminal="global", path=""):
    if not path:
        if terminal == "global":
            path = "data/flight/OAG_Covid19_Aviation_Data_06 March 2023 - 1. Weekly Global Scheduled Seats.csv"
        elif terminal == "domestic":
            path = "data/flight/OAG_Covid19_Aviation_Data_06 March 2023 - 2. Weekly Domestic Scheduled Seats.csv"
        elif terminal == "international":
            path = "data/flight/OAG_Covid19_Aviation_Data_06 March 2023 - 3. Weekly International Scheduled Seats.csv"

    df = pd.read_csv(path)
    years = [2019, 2020, 2021, 2022, 2023]
    seats = pd.concat([df[str(year)] for year in years], ignore_index=True).dropna()
    # for some reason only 52 weeks are provided for 2020 even though it should have a leap week
    year_col = reduce(lambda x, y: x + y, [[year] * 52 for year in years])[:len(seats)]
    week_index = (list(range(1, 53)) * 5)[:len(seats)]
    df = seats.apply(lambda s: s.replace(',', '')).astype(np.int64).to_frame(name=f"flight_seats_{terminal}")
    df["week"] = week_index
    df["year"] = year_col
    return df


def load_vaccination_data(path="", interpolation=None):
    if not path:
        path = "data/vaccination/COVID-19_Vaccinations_in_the_United_States_Jurisdiction.csv"

    df = pd.read_csv(path, parse_dates=["Date"])
    df = df[df["Location"] == "US"].reset_index(drop=True)
    df = df[["Date", "Administered"]]
    df.sort_values(by="Date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    if interpolation is not None:
        new_index = pd.date_range(start=df["Date"].min(), end=df["Date"].max(), name="Date")
        df = df.set_index("Date").reindex(new_index).reset_index()

        if interpolation == "ffill":
            df.fillna(method="ffill", inplace=True)
        elif interpolation == "linear":
            df["Administered"] = df["Administered"].interpolate()
        else:
            raise NotImplementedError

    # subtract series shifted by one to convert from cumulative to raw
    delta = df["Administered"][1:].values - df["Administered"][:-1].values
    df.rename(columns={"Administered": "administered_cum"}, inplace=True)
    df["administered_raw"] = np.append(0, delta).astype(np.int64)
    return df
