import pandas as pd
import os

# Dosya yollarını tanımlıyoruz (Verilerin 'data' klasöründe olduğunu varsayıyoruz)
TRANSACTION_PATH = 'data/train_transaction.csv'
IDENTITY_PATH = 'data/train_identity.csv'
OUTPUT_PATH = 'data/cleaned_train_step1.csv'

def prepare_data():
    print("1. Veriler yükleniyor (Bu işlem veri devasa olduğu için 1-2 dakika sürebilir)...")
    train_transaction = pd.read_csv(TRANSACTION_PATH)
    train_identity = pd.read_csv(IDENTITY_PATH)

    print("2. Transaction ve Identity tabloları birleştiriliyor...")
    # İşlem ve cihaz bilgilerini TransactionID üzerinden yan yana yapıştırıyoruz (Left Join)
    train_df = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
    print(f"Birleştirilmiş İlk Boyut: {train_df.shape}")

    print("3. Çöp (Eksik) sütunlar temizleniyor...")
    # Her sütunun yüzde kaçı boş (NaN) hesaplıyoruz
    missing_percentages = train_df.isnull().sum() / len(train_df) * 100
    
    # %80'den fazlası boş olan sütunları tespit ediyoruz
    threshold = 80
    columns_to_drop = missing_percentages[missing_percentages > threshold].index
    
    # Bu gereksiz sütunları veri setinden atıyoruz
    train_df_cleaned = train_df.drop(columns=columns_to_drop)
    
    print(f"Silinen Sütun Sayısı: {len(columns_to_drop)}")
    print(f"Temizlik Sonrası Kalan Boyut: {train_df_cleaned.shape}")

    print("4. Temizlenmiş veri 2. aşama için kaydediliyor...")
    # İşlem bitince RAM'i boşaltmak ve bir sonraki aşamaya hazır etmek için kaydediyoruz
    train_df_cleaned.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Başarılı! Temiz veri '{OUTPUT_PATH}' konumuna kaydedildi.")

if __name__ == "__main__":
    prepare_data()