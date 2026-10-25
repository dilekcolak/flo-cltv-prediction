import pandas as pd
import pytest
from src.data_prep import (
    outlier_thresholds,
    replace_with_thresholds,
    create_omnichannel_features,
    convert_date_columns
)
from src.clv_features import build_clv_features, get_analysis_data

@pytest.fixture
def sample_raw_dataframe():
    data = {
        "master_id": ["cust_1", "cust_2", "cust_3"],
        "order_num_total_ever_online": [2.0, 1.0, 5.0],
        "order_num_total_ever_offline": [1.0, 0.0, 2.0],
        "customer_value_total_ever_offline": [150.0, 0.0, 300.0],
        "customer_value_total_ever_online": [250.0, 100.0, 700.0],
        "first_order_date": ["2020-01-01", "2020-05-10", "2020-03-15"],
        "last_order_date": ["2021-01-01", "2020-05-10", "2021-05-20"]
    }
    return pd.DataFrame(data)

def test_create_omnichannel_features(sample_raw_dataframe):
    df = create_omnichannel_features(sample_raw_dataframe)

    assert "order_num_total" in df.columns
    assert "customer_value_total" in df.columns
    assert df.loc[df["master_id"] == "cust_1", "order_num_total"].values[0] == 3.0
    assert df.loc[df["master_id"] == "cust_1", "customer_value_total"].values[0] == 400.0

def test_convert_date_columns(sample_raw_dataframe):
    df = convert_date_columns(sample_raw_dataframe)
    assert pd.api.types.is_datetime64_any_dtype(df["first_order_date"])
    assert pd.api.types.is_datetime64_any_dtype(df["last_order_date"])

def test_replace_with_thresholds(sample_raw_dataframe):
    df = pd.DataFrame({"metric": [10.0, 12.0, 11.0, 10.5, 1000.0]})
    df_capped = replace_with_thresholds(df, "metric", q1=0.25, q3=0.75)

    assert df_capped["metric"].max() < 1000.0

def test_build_clv_features(sample_raw_dataframe):
    df_prep = create_omnichannel_features(sample_raw_dataframe)
    df_prep = convert_date_columns(df_prep)
    features_df = build_clv_features(df_prep, min_frequency=0)

    expected_cols = ["master_id", "recency_cltv_weekly", "T_weekly", "frequency", "monetary_cltv_avg"]
    for col in expected_cols:
        assert col in features_df.columns

    cust_1_freq = features_df.loc[features_df["master_id"] == "cust_1", "frequency"].values[0]
    assert cust_1_freq == 3
