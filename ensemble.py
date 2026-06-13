import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
import joblib
import warnings

warnings.filterwarnings('ignore')

def train_ultimate_ensemble():
    print("1. Veriler yükleniyor ve Zaman Mühendisliği yapılıyor...")
    df = pd.read_csv('data/train_final.csv')
    
    # Zaman Mühendisliği
    df['TransactionHour'] = (df['TransactionDT'] / 3600) % 24
    df['TransactionDay'] = (df['TransactionDT'] / (3600 * 24)) % 7
    
    y = df['isFraud']
    X = df.drop(columns=['isFraud', 'TransactionID', 'TransactionDT'])

    print("2. Veriler MLP (Sinir Ağı) için ölçeklendiriliyor (StandardScaling)...")
    # Ağaç modelleri (XGB, RF) ölçeklendirme istemez ama Sinir Ağları (MLP) için şarttır!
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    joblib.dump(scaler, 'models/scaler.pkl')

    print("3. Üç Büyük Uzman (Modeller) Hazırlanıyor...")
    scale_weight = len(y[y == 0]) / len(y[y == 1])

    # Uzman 1: XGBoost (Şampiyon Keskin Nişancı)
    xgb_model = xgb.XGBClassifier(
        n_estimators=400, max_depth=7, learning_rate=0.1,
        subsample=1.0, colsample_bytree=0.8, missing=-999,
        tree_method='hist', scale_pos_weight=scale_weight, 
        random_state=42, eval_metric='aucpr'
    )

    # Uzman 2: Random Forest (Stabil ve Bilge - Sınıf dengesizliği için class_weight='balanced')
    rf_model = RandomForestClassifier(
        n_estimators=200, max_depth=10, 
        class_weight='balanced', random_state=42, n_jobs=-1
    )

    # Uzman 3: Multi-layer Perceptron / Sinir Ağı (Gizli örüntü avcısı)
    # Hız için gizli katmanları 64 ve 32 nöronla sınırladık
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(64, 32), activation='relu', 
        learning_rate_init=0.001, max_iter=50, random_state=42
    )

    print("4. Ortak Akıl (Soft Voting Ensemble) Kuruluyor...")
    # 'soft' parametresi, modellerin evet/hayır demesini değil, olasılıkları ortalamasını sağlar
    ensemble_model = VotingClassifier(
        estimators=[
            ('XGBoost', xgb_model),
            ('RandomForest', rf_model),
            ('NeuralNetwork', mlp_model)
        ],
        voting='soft',
        n_jobs=-1
    )

    print("\n5. 5-Fold Cross-Validation Başlıyor (Bu işlem 20-40 dakika sürebilir, fan sesine hazırlıklı ol!)...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    auc_scores = []
    prc_scores = []
    
    for fold, (t_idx, v_idx) in enumerate(skf.split(X_scaled, y), 1):
        X_train, X_val = X_scaled.iloc[t_idx], X_scaled.iloc[v_idx]
        y_train, y_val = y.iloc[t_idx], y.iloc[v_idx]
        
        # Ensemble modeli bu fold için eğit
        ensemble_model.fit(X_train, y_train)
        
        # Olasılıkları tahmin et
        probs = ensemble_model.predict_proba(X_val)[:, 1]
        
        # Metrikler
        auc = roc_auc_score(y_val, probs)
        prc = average_precision_score(y_val, probs)
        
        auc_scores.append(auc)
        prc_scores.append(prc)
        print(f"   -> Fold {fold} | Ensemble AUC: {auc:.4f} | Ensemble AUPRC: {prc:.4f}")

    print("\n--- ULTIMATE ENSEMBLE SONUÇLARI ---")
    print(f"Ortalama AUC-ROC: {np.mean(auc_scores):.4f}")
    print(f"Ortalama AUPRC: {np.mean(prc_scores):.4f}")

    print("\n6. Model tüm veriyle eğitilip kaydediliyor...")
    ensemble_model.fit(X_scaled, y)
    joblib.dump(ensemble_model, 'models/ultimate_ensemble_model.pkl')
    print("✅ Başarılı! Efsanevi model 'models/ultimate_ensemble_model.pkl' olarak kaydedildi.")

if __name__ == "__main__":
    train_ultimate_ensemble()