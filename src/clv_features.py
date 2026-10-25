import pandas as pd
from pathlib import Path

def get_analysis_data(dataframe, time_buffer_days=2):
    latest_order_date = dataframe["last_order_date"].max()
    return latest_order_date + pd.Timedelta(days=time_buffer_days)

def build_clv_features(dataframe, analysis_date=None, min_frequency=1):
    df = dataframe.copy()
    if analysis_date is None:
        analysis_date = get_analysis_data(df, 2)

    cltv_df = pd.DataFrame()
    cltv_df["master_id"] = df["master_id"]
    cltv_df["recency_cltv_weekly"] = (df["last_order_date"] - df["first_order_date"]).dt.days / 7
    cltv_df["T_weekly"] = (analysis_date - df["first_order_date"]).dt.days / 7
    cltv_df["frequency"] = (df["order_num_total"]).astype(int)
    cltv_df["monetary_cltv_avg"] = df["customer_value_total"] / df["order_num_total"]
    filtered_df = cltv_df[cltv_df["frequency"] > min_frequency].copy()
    return filtered_df

if __name__ == "__main__":
    from src.data_prep import prepare_flo_data

    BASE_DIR = Path(__file__).resolve().parent.parent
    RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "flo_data_20k.csv"
    try:
        cleaned_data = prepare_flo_data(RAW_DATA_PATH)
        features_df = build_clv_features(cleaned_data)
        print("CLV features successfully extracted.")
        print(f"Eligible Customer Count: {features_df.shape[0]}")
        print(features_df.head())
    except FileNotFoundError:
        print("Data file not found. Ensure raw data exists in data/raw/.")