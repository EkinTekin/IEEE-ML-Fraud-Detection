import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
import warnings

warnings.filterwarnings('ignore')

INPUT_PATH = 'data/train_final.csv'

def tune_xgboost():
    print("Veri yükleniyor...")
    df = pd.read_csv(INPUT_PATH)
    y = df['isFraud']
    X = df.drop(columns=['isFraud', 'TransactionID', 'TransactionDT'])

    # Temel XGBoost Modeli
    xgb_model = xgb.XGBClassifier(tree_method='hist', missing=-999, random_state=42, eval_metric='aucpr')

    # Modül 7 ve 9-1'de öğretilen Parametre Uzayı
    param_dist = {
        'max_depth': [5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [200, 400, 600],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    # Çapraz Doğrulama (Modül 8) - Hız için 3 fold yapıyoruz
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # Modül 9-1'deki RandomizedSearchCV uygulaması
    # Hedefimiz AUPRC skorunu ('average_precision') maksimize etmek
    random_search = RandomizedSearchCV(
        xgb_model, param_distributions=param_dist, n_iter=10, 
        scoring='average_precision', cv=skf, verbose=3, random_state=42, n_jobs=-1
    )

    print("Hiperparametre optimizasyonu başlıyor (Bu işlem 15-30 dk sürebilir)...")
    random_search.fit(X, y)

    print("\n--- BULUNAN EN İYİ PARAMETRELER ---")
    print(random_search.best_params_)
    print(f"En İyi AUPRC Skoru: {random_search.best_score_:.4f}")

if __name__ == "__main__":
    tune_xgboost()