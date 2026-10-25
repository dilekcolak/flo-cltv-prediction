"""
hypothesis_tests.py
-------------------
Gamma-Gamma modeli öncesinde 'Frequency' ve 'Monetary' değişkenleri arasındaki
bağımsızlık varsayımını test eden istatistiksel analiz fonksiyonlarını içerir.
"""

#Bu modülü sadece değerlendirmek amacıyla yapay zekaya hazırlattım. Proje için kullanmadım.

from typing import Dict, Any
from pathlib import Path
import pandas as pd
from scipy import stats


def test_gamma_gamma_assumptions(
        dataframe: pd.DataFrame,
        freq_col: str = "frequency",
        monetary_col: str = "monetary_cltv_avg",
        alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Gamma-Gamma modelinin çalışması için gereken temel varsayımı (Frequency ve
    Monetary değerleri arasında anlamlı bir korelasyon olmaması durumu) test eder.

    Parameters:
    - dataframe: 'frequency' ve 'monetary_cltv_avg' içeren DataFrame
    - freq_col: Satın alma sıklığı sütun adı
    - monetary_col: Ortalama sepet tutarı sütun adı
    - alpha: İstatistiksel anlamlılık eşiği (varsayılan: 0.05)

    Returns:
    - Test sonuçları ve varsayımın geçerlilik durumunu içeren sözlük.
    """
    freq = dataframe[freq_col]
    monetary = dataframe[monetary_col]

    # Pearson korelasyon katsayısı ve p-değeri (Doğrusal ilişki testi)
    pearson_corr, pearson_p = stats.pearsonr(freq, monetary)

    # Spearman rank korelasyon katsayısı ve p-değeri (Monotonik ilişki testi)
    spearman_corr, spearman_p = stats.spearmanr(freq, monetary)

    # Hipotez:
    # H0: Frequency ve Monetary arasında ilişki yoktur (rho = 0).
    # H1: Frequency ve Monetary arasında anlamlı bir ilişki vardır (rho != 0).
    # Not: Büyük örneklemlerde çok zayıf korelasyonlar bile p < 0.05 verebilir.
    # Bu nedenle korelasyon katsayısının mutlak büyüklüğü (|r| < 0.15) ana kriterdir.
    is_practically_independent = abs(pearson_corr) < 0.15

    results = {
        "pearson_corr": round(pearson_corr, 4),
        "pearson_p_value": round(pearson_p, 6),
        "spearman_corr": round(spearman_corr, 4),
        "spearman_p_value": round(spearman_p, 6),
        "is_assumption_valid": is_practically_independent,
        "sample_size": len(dataframe)
    }

    return results


def print_assumption_report(results: Dict[str, Any]) -> None:
    """
    Hipotez testi sonuçlarını okunabilir bir iş/istatistik raporu formatında yazdırır.
    """
    print("\n" + "=" * 60)
    print("GAMMA-GAMMA MODEL VARSAYIM TESTİ RAPORU")
    print("=" * 60)
    print(f"İncelenen Müşteri Sayısı (Sample Size): {results['sample_size']}")
    print(f"Pearson Korelasyon Katsayısı: {results['pearson_corr']} (p-value: {results['pearson_p_value']})")
    print(f"Spearman Korelasyon Katsayısı: {results['spearman_corr']} (p-value: {results['spearman_p_value']})")
    print("-" * 60)

    if results["is_assumption_valid"]:
        print("[BAŞARILI] Frequency ve Monetary arasındaki korelasyon sıfıra oldukça yakın.")
        print(">> Gamma-Gamma alt modeli için gereken bağımsızlık varsayımı SAĞLANDI.")
    else:
        print("[UYARI] Frequency ve Monetary arasında pratikte dikkate değer bir ilişki var (|r| >= 0.15).")
        print(">> Gamma-Gamma tahminleri yanlılık içerebilir; penalizer_coef ayarlanmalıdır.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    from src.data_prep import prepare_flo_data
    from src.clv_features import build_clv_features

    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_PATH = BASE_DIR / "data" / "raw" / "flo_data_20k.csv"

    try:
        raw_df = prepare_flo_data(str(DATA_PATH))
        clv_df = build_clv_features(raw_df)

        test_results = test_gamma_gamma_assumptions(clv_df)
        print_assumption_report(test_results)
    except FileNotFoundError:
        print(f"Dosya bulunamadı: {DATA_PATH}. Lütfen yolu kontrol edin.")