# FLO Müşteri Yaşam Boyu Değeri (CLTV) Tahmini ve Segmentasyonu

Bu proje, **BG-NBD** ve **Gamma-Gamma** modellerini kullanarak müşteri yaşam boyu değerini (Customer Lifetime Value, CLV/CLTV) tahmin eden ve müşterileri hem tahmini değerlerine hem de aktif kalma olasılıklarına göre segmentlere ayıran modüler bir Python uygulamasıdır.

FLO CLTV Prediction vaka çalışmasından hareketle eğitim amacıyla geliştirilmiştir. Temel modelleme sürecine yeniden kullanılabilir modüller, istatistiksel tanı analizleri, birim testleri ve segment bazlı aksiyon önerileri eklenmiştir.

## Projenin özellikleri

- Online ve offline sipariş sayılarının ve harcama tutarlarının birleştirilmesi, aykırı değerlerin baskılanması
- Müşteri bazında recency, müşteri yaşı (T), tekrar satın alma sıklığı ve ortalama sipariş tutarı değişkenlerinin oluşturulması
- **BG-NBD** ile gelecekteki satın alma sayısının ve müşterinin aktif kalma olasılığının (`p_alive`) tahmin edilmesi
- **Gamma-Gamma** ile beklenen ortalama işlem tutarının hesaplanması
- Varsayılan olarak 6 aylık, değiştirilebilir tahmin ufkunda indirgenmiş CLV hesaplanması
- CLV çeyrekliklerine göre **A, B, C, D** segmentlerinin oluşturulması
- CLV ve `p_alive` birlikte değerlendirilerek risk odaklı stratejik segmentasyon yapılması
- Frequency ve monetary değişkenleri için Pearson ve Spearman korelasyon analizleri
- Sonuçların CSV dosyasına aktarılması ve `pytest` ile birim testleri

## Proje dizin yapısı

```text
flo-cltv-prediction/
├── data/
│   ├── raw/                  # Ham veri seti (Git'e eklenmez)
│   └── processed/            # Analiz çıktıları (Git'e eklenmez)
├── src/
│   ├── clv_features.py       # CLV model değişkenleri
│   ├── data_prep.py          # Veri ön işleme
│   ├── hypothesis_tests.py   # İstatistiksel analizler
│   ├── models.py             # BG-NBD ve Gamma-Gamma
│   └── segmentation.py       # Müşteri segmentasyonu
├── tests/
│   ├── test_data_prep.py
│   └── test_models.py
├── main.py                   # Ana analiz pipeline'ı
├── requirements.txt          # Python bağımlılıkları
├── .gitignore                # Git hariç tutma kuralları
└── README.md                 # Proje dokümantasyonu
```

## Kurulum

`lifetimes` paketiyle uyumlu bir Python ortamı kullanın. Projenin ana dizininde sanal ortam oluşturun:

```bash
python -m venv .venv
```

Windows PowerShell'de sanal ortamı etkinleştirin:

```powershell
.\.venv\Scripts\Activate.ps1
```

Bağımlılıkları yükleyin:

```bash
python -m pip install -r requirements.txt
```

FLO veri setini `data/raw/flo_data_20k.csv` konumuna yerleştirin. Veri seti GitHub deposuna dahil edilmemiştir.

## Projeyi çalıştırma

Projenin ana dizininden:

```bash
python main.py
```

İsteğe bağlı komut satırı parametreleriyle:

```bash
python main.py --data-path data/raw/flo_data_20k.csv --output-path data/processed/flo_clv_segmented_customers.csv --months 6 --discount-rate 0.01
```

Analiz akışı sırasıyla veriyi hazırlar, CLV değişkenlerini oluşturur, frequency–monetary korelasyonlarını raporlar, iki olasılıksal modeli eğitir, müşteri bazında tahminleri hesaplar ve segmentlenmiş sonuçları CSV olarak kaydeder.

## Çıktılar

Varsayılan çıktı dosyası: `data/processed/flo_clv_segmented_customers.csv`

| Sütun | Açıklama |
| --- | --- |
| `master_id` | Müşteri kimliği |
| `frequency` | Tekrar satın alma sayısı (toplam sipariş sayısı eksi bir) |
| `recency_cltv_weekly` | İlk ve son satın alma arasındaki hafta sayısı |
| `T_weekly` | İlk satın alma ile analiz tarihi arasındaki hafta sayısı |
| `monetary_cltv_avg` | Geçmiş sipariş başına ortalama gelir |
| `p_alive` | Müşterinin hâlen aktif olma olasılığı tahmini |
| `exp_sales_3_months` / `exp_sales_6_months` | Gelecekte beklenen satın alma sayıları |
| `exp_average_value` | Beklenen ortalama işlem tutarı |
| `clv` | İndirgenmiş tahmini müşteri yaşam boyu değeri |
| `clv_segment` | CLV çeyreklik segmenti (A en yüksek, D en düşük) |
| `strategic_segment` | CLV ve aktiflik olasılığına dayalı stratejik segment |

Analiz tarihi, veri setindeki en son sipariş tarihinden **iki gün sonrası** olarak belirlenir. Mevcut uygulama, birden fazla tekrar satın alması olan müşterileri (`frequency > 1`) modele dahil eder.

## Müşteri segmentasyonu

**CLV segmentleri:** Müşteriler tahmini CLV değerlerine göre dört çeyreklik gruba ayrılır: **A** (en yüksek), **B**, **C** ve **D** (en düşük).

**Risk odaklı stratejik segmentler:** CLV ve `p_alive` değişkenlerinin medyanlarına göre dört grup oluşturulur:

- **Champions (VIP):** Yüksek CLV, yüksek aktiflik olasılığı
- **At-Risk High Value:** Yüksek CLV, düşük aktiflik olasılığı
- **Potential Loyals:** Düşük CLV, yüksek aktiflik olasılığı
- **Hibernating:** Düşük CLV, düşük aktiflik olasılığı

Mevcut kod, komut satırında A ve C segmentleri için örnek pazarlama aksiyonları sunar. Bunlar deneysel olarak doğrulanmış sonuçlar değil, örnek strateji önerileridir.

## Testler

```bash
python -m pytest -q
```

**Bilinen düzeltme:** `tests/test_models.py` içindeki `test_model_unfitted_raises_error` fonksiyonuna `mock_clv_fatures_df` fixture parametresi eklenmelidir. Mevcut proje durumunda tüm testlerin başarıyla geçtiği doğrulanmamıştır.

## Metodolojik notlar ve sınırlılıklar

- BG-NBD ve Gamma-Gamma modellerinin varsayımları, tahminler gerçek iş kararlarında kullanılmadan önce değerlendirilmelidir.
- Pearson ve Spearman korelasyonları **tanısal göstergelerdir**; istatistiksel bağımsızlığı tek başlarına kanıtlamaz.
- Kodda tekrar satın alma sıklığı, toplam sipariş sayısının bir eksiği olarak tanımlanmıştır. Vaka çalışmasındaki sonuçlarla karşılaştırmadan önce bu tanımın uyumu kontrol edilmelidir.
- Monetary değişkeni olarak işlem başına gelir kullanıldığı için CLV tahmini **gelir bazlıdır**; net kâr tahmini değildir.
- Mevcut analiz akışında model kalibrasyonu, örneklem dışı doğrulama ve belirsizlik tahminleri bulunmamaktadır.

## Kullanılan teknolojiler

**Python**, **pandas**, **NumPy**, **SciPy**, **lifetimes**, **pytest**.

## Veri seti ve kaynak

Bu çalışma, FLO CLTV Prediction vaka çalışmasını temel alan eğitim amaçlı bir uygulamadır. Kaynak veri seti, ilgili kullanım koşullarına uygun biçimde temin edilmelidir. Orijinal vaka çalışması PDF'i bu GitHub deposuna dahil edilmemiştir.
