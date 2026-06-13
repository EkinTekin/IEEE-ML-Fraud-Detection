import pandas as pd
from sklearn.preprocessing import LabelEncoder
import warnings

# Terminaldeki gereksiz pembe uyarıları kapatalım
warnings.filterwarnings('ignore')

INPUT_PATH = 'data/cleaned_train_step1.csv'
OUTPUT_PATH = 'data/train_final.csv'

def feature_engineering():
    print("1. Temizlenmiş veri (Step 1) yükleniyor...")
    df = pd.read_csv(INPUT_PATH)

    print("2. Kalan Eksik Veriler (NaN) Akıllıca Dolduruluyor...")
    # Verileri türlerine göre (Metin vs Sayı) ikiye ayırıyoruz
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    numeric_cols = df.select_dtypes(exclude=['object', 'category']).columns

    # Kaggle Taktiği: Sayısal boşluklara -999 basıyoruz (XGBoost bunu çok sever)
    df[numeric_cols] = df[numeric_cols].fillna(-999)

    # Metin boşluklarına ise basitçe 'unknown' (bilinmeyen) yazıyoruz
    df[categorical_cols] = df[categorical_cols].fillna('unknown')

    print("3. Kategorik Veriler (Metinler) Sayılara Çevriliyor (Label Encoding)...")
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = df[col].astype(str) # Hata almamak için her şeyi metne zorluyoruz
        df[col] = le.fit_transform(df[col]) # 'Visa' -> 0, 'Mastercard' -> 1 vb.

    print(f"Dönüştürülen metin tabanlı sütun sayısı: {len(categorical_cols)}")

    print("4. Modele Girecek Nihai Veri Kaydediliyor...")
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Başarılı! Eğitime hazır, kusursuz nihai veri '{OUTPUT_PATH}' konumuna kaydedildi.")

if __name__ == "__main__":
    feature_engineering()