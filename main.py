import argparse
from pathlib import Path
import pandas as pd

from src.data_prep import prepare_flo_data
from src.clv_features import build_clv_features
from src.hypothesis_tests import test_gamma_gamma_assumptions, print_assumption_report
from src.models import CLVPredictionEngine
from src.segmentation import (
    create_clv_segments,
    create_risk_adjusted_segments,
    get_segment_profiles,
    get_action_recommendations
)

def run_pipeline(
        raw_data_path=Path,
        output_path=Path,
        clv_months=6,
        discount_rate=0.01,
        bg_penalizer=0.001,
        gg_penalizer=0.01
):
    print("\n" + "*" * 70)
    print("FLO CUSTOMER LIFETIME VALUE PREDICTION PIPELINE")
    print("*" * 70)

    print("\n[Adım 1/5] Ham veri okunuyor ve aykırı değerler baskılanıyor...")
    df_prepared = prepare_flo_data(str(raw_data_path))
    print(f"-> Veri hazırlandı. Satır sayısı {df_prepared.shape[0]}")

    print("\n[Adım 2/5] RFM ve Lifetimes öznitelikleri (x, t_x, T, M) hesaplanıyor...")
    clv_features_df = build_clv_features(df_prepared, min_frequency=1)
    print(f"-> Model için uygun müşteri sayısı: {clv_features_df.shape[0]}")

    print("\n[Adım 3/5] Gamma-Gamma bağımsızlık varsayımı test ediliyor...")
    hypothesis_results = test_gamma_gamma_assumptions(clv_features_df)
    print_assumption_report(hypothesis_results)

    print(f"\n[Adım 4/5] Modeller eğitiliyor ve {clv_months} aylık CLV hesaplanıyor...")
    engine = CLVPredictionEngine(bg_penalizer=bg_penalizer, gg_penalizer=gg_penalizer)
    engine.fit(clv_features_df)

    scored_df = engine.genarate_full_clv_dataframe(
        clv_features_df,
        clv_time_months=clv_months,
        discount_rate=discount_rate
    )
    print("-> P(Alive), beklenen satın alımlar ve CLV metrikleri üretildi.")

    print("\n[Adım 5/5] Müşteri segmentleri ve aksiyon stratejileri oluşturuluyor...")
    segmented_df = create_clv_segments(scored_df, n_segments=4)
    segmented_df = create_risk_adjusted_segments(segmented_df)

    profiles = get_segment_profiles(segmented_df, group_col="clv_segment")
    print("\nSegment Özet Tablosu:")
    print(profiles.to_string(index=False))

    get_action_recommendations(profiles)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    segmented_df.to_csv(output_path, index=False)
    print(f"\n[Tamamlandı] Sonuçlar başarıyla kaydedildi: {output_path}")
    print("*" * 70 + "\n")

    return segmented_df

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    DEFAULT_DATA_PATH = BASE_DIR / "data" / "raw" / "flo_data_20k.csv"
    DEFAULT_OUTPUT_PATH = BASE_DIR / "data" / "processed" / "flo_clv_segmented_customers.csv"

    parser = argparse.ArgumentParser(description="FLO CLV Prediction & Segmentation Pipeline")
    parser.add_argument("--data-path", type=str, default=str(DEFAULT_DATA_PATH), help="Ham veri dosya yolu")
    parser.add_argument("--output-path", type=str, default=str(DEFAULT_OUTPUT_PATH), help="İşlenmiş veri kayıt yolu")
    parser.add_argument("--months", type=int, default=6, help="CLV tahmin periyodu (ay cinsinden)")
    parser.add_argument("--discount-rate", type=float, default=0.01, help="Aylık iskonto oranı")

    args = parser.parse_args()

    run_pipeline(
        raw_data_path=Path(args.data_path),
        output_path=Path(args.output_path),
        clv_months=args.months,
        discount_rate=args.discount_rate
    )