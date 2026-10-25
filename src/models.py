from pathlib import Path
import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter

class CLVPredictionEngine:
    def __init__(self, bg_penalizer=0.001, gg_penalizer=0.01):
        self.bg_penalizer = bg_penalizer
        self.gg_penalizer = gg_penalizer
        self.bgf = None
        self.ggf = None

    def fit(self, dataframe):
        self.bgf = BetaGeoFitter(penalizer_coef=self.bg_penalizer)
        self.bgf.fit(
            dataframe["frequency"],
            dataframe["recency_cltv_weekly"],
            dataframe["T_weekly"]
        )

        self.ggf = GammaGammaFitter(penalizer_coef=self.gg_penalizer)
        self.ggf.fit(
            dataframe["frequency"],
            dataframe["monetary_cltv_avg"]
        )
        return self

    def predict_purchases(
            self,
            dataframe,
            time_months=3
    ):
        if self.bgf is None:
            raise ValueError("BG-NBD modeli henüz eğitilmedi. Önce fit() metodunu çalıştırın.")

        weeks = time_months * 4
        return self.bgf.predict(
            weeks,
            dataframe["frequency"],
            dataframe["recency_cltv_weekly"],
            dataframe["T_weekly"]
        )

    def predict_expected_average_value(self, dataframe):
        if self.ggf is None:
            raise ValueError("Gamma-Gamma modeli henüz eğitilmedi. Önce fit() metodunu çalıştırın.")

        return self.ggf.conditional_expected_average_profit(
            dataframe["frequency"],
            dataframe["monetary_cltv_avg"]
        )

    def calculate_probability_alive(self, dataframe):
        if self.bgf is None:
            raise ValueError("BG-NBD modeli henüz eğitilmedi.")

        return self.bgf.conditional_probability_alive(
            dataframe["frequency"],
            dataframe["recency_cltv_weekly"],
            dataframe["T_weekly"]
        )

    def calculate_clv(
            self,
            dataframe,
            time_months=6,
            discount_rate=0.01,
            freq_type="W"
    ):
        if self.bgf is None or self.ggf is None:
            raise ValueError("Modeller henüz eğitilmedi. Önce fit() metodunu çalıştırın.")

        return self.ggf.customer_lifetime_value(
            self.bgf,
            dataframe["frequency"],
            dataframe["recency_cltv_weekly"],
            dataframe["T_weekly"],
            dataframe["monetary_cltv_avg"],
            time=time_months,
            discount_rate=discount_rate,
            freq=freq_type
        )

    def genarate_full_clv_dataframe(self, dataframe, clv_time_months=6, discount_rate=0.01):
        df_results = dataframe.copy()
        df_results["p_alive"] = self.calculate_probability_alive(df_results)
        df_results["exp_sales_3_months"] = self.predict_purchases(df_results, time_months=3)
        df_results["exp_sales_6_months"] = self.predict_purchases(df_results, time_months=6)

        df_results["exp_average_value"] = self.predict_expected_average_value(df_results)

        df_results["clv"] = self.calculate_clv(
            df_results,
            time_months=clv_time_months,
            discount_rate=discount_rate
        )
        return df_results

if __name__ == "__main__":
    from src.data_prep import prepare_flo_data
    from src.clv_features import build_clv_features

    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_PATH = BASE_DIR / "data" / "raw" / "flo_data_20k.csv"

    try:
        raw_df = prepare_flo_data(str(DATA_PATH))
        clv_input_df = build_clv_features(raw_df)

        engine = CLVPredictionEngine(bg_penalizer=0.001, gg_penalizer=0.01)
        engine.fit(clv_input_df)

        final_df = engine.genarate_full_clv_dataframe(clv_input_df, clv_time_months=6)
        print("Model training and predictions completed successfully.")
        print("\nTop 5 Customers by CLV:")
        print(final_df.sort_values(by="clv", ascending=False).head())
    except FileNotFoundError:
        print(f"File not found at: {DATA_PATH}")
