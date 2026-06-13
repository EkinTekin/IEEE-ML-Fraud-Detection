import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

def train_advanced_model():
    print("1. Nihai veri yükleniyor...")
    df = pd.read_csv('data/train_final.csv')
    
    print("2. Zaman Mühendisliği (Feature Engineering) yapılıyor...")
    # Saniyeleri güne ve saate çeviriyoruz (Dolandırıcıların saatlik alışkanlıklarını bulmak için)
    # 3600 saniye = 1 saat | 3600*24 saniye = 1 gün
    df['TransactionHour'] = (df['TransactionDT'] / 3600) % 24
    df['TransactionDay'] = (df['TransactionDT'] / (3600 * 24)) % 7
    
    y = df['isFraud']
    # Artık TransactionDT'yi silebiliriz çünkü içindeki cevheri Saat ve Gün olarak aldık
    X = df.drop(columns=['isFraud', 'TransactionID', 'TransactionDT'])

    print("3. Sınıf Dengesizliği (Imbalance) hesaplanıyor...")
    # Masumların sayısını, dolandırıcıların sayısına bölüyoruz
    scale_weight = len(y[y == 0]) / len(y[y == 1])
    print(f"   -> Modele dolandırıcıları kaçırmaması için {scale_weight:.2f} kat ekstra dikkat cezası verildi!")

    # Geliştirilmiş Süper Modelimiz
    xgb_model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=7,
        learning_rate=0.1,
        subsample=1.0,
        colsample_bytree=0.8,
        missing=-999,
        tree_method='hist',
        scale_pos_weight=scale_weight,  # YENİ EKLENEN KRİTİK PARAMETRE
        random_state=42,
        eval_metric='aucpr'
    )

    print("\n4. 5-Fold Cross-Validation ve Eğitim Başlıyor...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    auc_scores = []
    prc_scores = []
    
    for fold, (t_idx, v_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X.iloc[t_idx], X.iloc[v_idx]
        y_train, y_val = y.iloc[t_idx], y.iloc[v_idx]
        
        xgb_model.fit(X_train, y_train)
        probs = xgb_model.predict_proba(X_val)[:, 1]
        
        auc = roc_auc_score(y_val, probs)
        prc = average_precision_score(y_val, probs)
        
        auc_scores.append(auc)
        prc_scores.append(prc)
        print(f"   -> Fold {fold} | AUC: {auc:.4f} | AUPRC: {prc:.4f}")

    print("\n--- YENİ SÜPER MODEL SONUÇLARI ---")
    print(f"Ortalama AUC-ROC: {np.mean(auc_scores):.4f}")
    print(f"Ortalama AUPRC: {np.mean(prc_scores):.4f}")

    print("\n5. Model kaydediliyor ve Grafik çiziliyor...")
    xgb_model.fit(X, y)
    joblib.dump(xgb_model, 'models/best_fraud_model_v2.pkl')

    fig, ax = plt.subplots(figsize=(12, 8))
    xgb.plot_importance(xgb_model, max_num_features=20, ax=ax, height=0.5, 
                        title='En Önemli 20 Özellik (Yeni Sütunlarla Birlikte)')
    plt.tight_layout()
    plt.savefig('feature_importance_v2.png')
    print("✅ İşlem Tamam! Yeni grafik 'feature_importance_v2.png' olarak kaydedildi.")

if __name__ == "__main__":
    train_advanced_model()