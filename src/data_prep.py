import pandas as pd
from pathlib import Path

def load_raw_data(filepath):
    df = pd.read_csv(filepath)
    return df.copy()

def outlier_thresholds(dataframe, variable, q1=0.01, q3=0.99):
    quantile1 = dataframe[variable].quantile(q1)
    quantile3 = dataframe[variable].quantile(q3)
    iqr = quantile3 - quantile1

    low_limit = round(quantile1 - 1.5 * iqr)
    up_limit = round(quantile3 + 1.5 * iqr)

    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable, q1=0.01, q3=0.99):
    dataframe = dataframe.copy()
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1, q3)
    dataframe.loc[dataframe[variable] < low_limit, variable] = low_limit
    dataframe.loc[dataframe[variable] > up_limit, variable] = up_limit

    return dataframe

def handle_outliers(dataframe, target_columns=None, q1=0.01, q3=0.99):
    if target_columns is None:
        target_columns = [
            "order_num_total_ever_online",
            "order_num_total_ever_offline",
            "customer_value_total_ever_offline",
            "customer_value_total_ever_online"
        ]
    df = dataframe.copy()
    for col in target_columns:
        if col in df.columns:
            df = replace_with_thresholds(df, col, q1, q3)
    return df

def create_omnichannel_features(dataframe):
    df = dataframe.copy()
    df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
    df["customer_value_total"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]
    return df

def convert_date_columns(dataframe):
    df = dataframe.copy()
    date_cols = [col for col in df.columns if "_date" in col]

    for col in date_cols:
        df[col] = pd.to_datetime(df[col])

    return df

def prepare_flo_data(filepath):
    df = load_raw_data(filepath)
    df = handle_outliers(df, q1=0.01, q3=0.99)
    df = create_omnichannel_features(df)
    df = convert_date_columns(df)
    return df

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_PATH = BASE_DIR / "data" / "raw" / "flo_data_20k.csv"
    try:
        processed_df = prepare_flo_data(DATA_PATH)
        print("Data preparation pipeline successfully executed.")
        print(f"Dataframe Shape: {processed_df.shape}")
        print(f"Columns: {processed_df.columns.tolist()}")
    except FileNotFoundError:
        print(f"File not found at: {DATA_PATH}. Please verify the path.")
