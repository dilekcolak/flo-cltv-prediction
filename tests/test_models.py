import pandas as pd
import pytest
from src.models import CLVPredictionEngine

@pytest.fixture
def mock_clv_features_df():
    data = {
        "master_id": [f"cust_{i}" for i in range(1, 11)],
        "frequency": [2, 3, 5, 2, 4, 6, 3, 2, 8, 4],
        "recency_cltv_weekly": [10.5, 20.0, 35.2, 8.0, 22.1, 40.0, 15.0, 12.3, 50.0, 28.4],
        "T_weekly": [25.0, 30.0, 42.0, 18.0, 32.0, 45.0, 22.0, 20.0, 55.0, 35.0],
        "monetary_cltv_avg": [120.0, 85.5, 210.0, 60.0, 145.0, 190.0, 110.0, 75.0, 300.0, 130.0]
    }
    return pd.DataFrame(data)

def test_model_fit_and_generate_predictions(mock_clv_features_df):
    engine = CLVPredictionEngine(bg_penalizer=0.001, gg_penalizer=0.01)
    engine.fit(mock_clv_features_df)

    results_df = engine.genarate_full_clv_dataframe(mock_clv_features_df, clv_time_months=6)
    assert "p_alive" in results_df.columns
    assert "exp_sales_3_months" in results_df.columns
    assert "exp_sales_6_months" in results_df.columns
    assert "exp_average_value" in results_df.columns
    assert "clv" in results_df.columns

    assert results_df["p_alive"].between(0.0, 1.0).all()

    assert (results_df["exp_sales_3_months"] >= 0).all()
    assert (results_df["exp_sales_6_months"] >= 0).all()
    assert (results_df["exp_average_value"] >= 0).all()
    assert (results_df["clv"] >= 0).all()

    assert (results_df["exp_sales_6_months"] >= results_df["exp_sales_3_months"]).all()

def test_model_unfitted_raises_error(mock_clv_features_df):
    unfitted_engine = CLVPredictionEngine()

    with pytest.raises(ValueError):
        unfitted_engine.predict_purchases(mock_clv_features_df)
