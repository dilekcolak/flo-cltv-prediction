from pathlib import Path
import pandas as pd

def create_clv_segments(
        dataframe,
        clv_col="clv",
        n_segments=4,
        labels=None
):
    df = dataframe.copy()
    if labels is None:
        labels = ["D", "C", "B", "A"]

    df["clv_segment"] = pd.qcut(df[clv_col], n_segments, labels)
    return df

def create_risk_adjusted_segments(
        dataframe,
        clv_col="clv",
        p_alive_col="p_alive"
):
    df = dataframe.copy()
    clv_median = df[clv_col].median()
    p_alive_median = df[p_alive_col].median()

    def categorize(row):
        high_clv = row[clv_col] >= clv_median
        high_alive = row[p_alive_col] >= p_alive_median

        if high_clv and high_alive:
            return "Champions (VIP)"
        elif high_clv and not high_alive:
            return "At-Risk High Value"
        elif not high_clv and high_alive:
            return "Potential Loyals"
        else:
            return "Hibernating"

    df["strategic_segment"] = df.apply(categorize, axis=1)
    return df
def get_segment_profiles(
        dataframe,
        group_col="clv_segment"
):
    summary = dataframe.groupby(group_col).agg(
        customer_count=("master_id", "count"),
        recency_weekly_mean=("recency_cltv_weekly", "mean"),
        tenure_weekly_mean=("T_weekly", "mean"),
        frequency_mean=("frequency", "mean"),
        monetary_mean=("monetary_cltv_avg", "mean"),
        exp_sales_6m_mean=("exp_sales_6_months", "mean"),
        p_alive_mean=("p_alive", "mean"),
        clv_mean=("clv", "mean"),
        clv_total=("clv", "sum")
    ).reset_index()
    return summary

def get_action_recommendations(profile_df):
    print("\n" + "=" * 70)
    print("YÖNETİM İÇİN 6 AYLIK STRATEJİK CLV AKSİYON ÖNERİLERİ")
    print("=" * 70)

    print("\n1. SEGMENT A (VIP / En Yüksek Değerli Müşteriler):")
    print("   - Hedef: Müşteri kaybını sıfırlamak ve marka bağlılığını maksimize etmek.")
    print("   - Aksiyon: Kişiselleştirilmiş VIP müşteri temsilcisi atanması, yeni sezon ürünlerinde")
    print("     öncelikli erişim hakkı ve sadakat ödül puanlarının iki katına çıkarılması.")
    print("   - Kanal: Doğrudan SMS/Arama ve mobil uygulama özel bildirimleri.")

    print("\n2. SEGMENT C (Geliştirilmeye Açık / Churn Riski Olabilecek Kitle):")
    print("   - Hedef: Satın alma frekansını artırarak B segmentine taşımak.")
    print("   - Aksiyon: Son ilgilendikleri kategorilere yönelik 'Sepette İndirim' veya")
    print("     'Belirli limit üzeri ücretsiz kargo' gibi sepet teşvikleri sunulması.")
    print("   - Kanal: Dinamik e-posta bültenleri ve push bildirimler.")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    from src.data_prep import prepare_flo_data
    from src.clv_features import build_clv_features
    from src.models import CLVPredictionEngine

    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_PATH = BASE_DIR / "data" / "raw" / "flo_data_20k.csv"

    try:
        raw_df = prepare_flo_data(str(DATA_PATH))
        clv_input_df = build_clv_features(raw_df)

        engine = CLVPredictionEngine(bg_penalizer=0.001, gg_penalizer=0.01)
        engine.fit(clv_input_df)
        predicted_df = engine.genarate_full_clv_dataframe(clv_input_df)

        segmented_df = create_clv_segments(predicted_df)
        segmented_df = create_risk_adjusted_segments(segmented_df)

        profiles = get_segment_profiles(segmented_df, group_col="clv_segment")
        print("Segment Özet Tablosu:")
        print(profiles.to_string(index=False))

        get_action_recommendations(profiles)
    except FileNotFoundError:
        print(f"Dosya Bulunamadı: {DATA_PATH}")

