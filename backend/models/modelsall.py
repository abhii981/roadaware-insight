import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix,
                             roc_auc_score)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import warnings
import joblib
import os
import json
warnings.filterwarnings('ignore')

print("=" * 60)
print("INDIAN ROAD ACCIDENT RISK ANALYZER - MODEL TRAINING")
print("=" * 60)

def remove_outliers_iqr(df, exclude_cols=None):
    """
    Remove outliers using IQR method.
    Only applied to numeric columns that make sense for outlier removal.
    Categorical encoded columns and binary columns are excluded.
    """
    if exclude_cols is None:
        exclude_cols = []

    original_size = len(df)
    numeric_cols  = df.select_dtypes(include=['int64', 'float64']).columns
    cols_to_check = [c for c in numeric_cols if c not in exclude_cols]

    outlier_summary = {}
    for col in cols_to_check:
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers_count = ((df[col] < lower) | (df[col] > upper)).sum()
        if outliers_count > 0:
            outlier_summary[col] = outliers_count
        df = df[(df[col] >= lower) & (df[col] <= upper)]

    removed = original_size - len(df)
    print(f"\n IQR Outlier Detection Results:")
    print(f"   Original rows  : {original_size:,}")
    print(f"   Rows removed   : {removed:,}")
    print(f"   Remaining rows : {len(df):,}")
    print(f"   Columns checked: {len(cols_to_check)}")
    if outlier_summary:
        print(f"   Outliers found per column:")
        for col, count in outlier_summary.items():
            print(f"     {col}: {count}")
    return df

print("\n Loading dataset...")
df = pd.read_csv("data/indian_roads_dataset.csv")
print(f"   Loaded {len(df):,} rows, {len(df.columns)} columns")

exclude = [
    'accident_id', 'is_weekend', 'is_peak_hour',
    'traffic_signal', 'lanes'
]
df = remove_outliers_iqr(df, exclude_cols=exclude)


df['festival'] = df['festival'].fillna('none')
df['date']     = pd.to_datetime(df['date'])
df['month']    = df['date'].dt.month


print("\n Encoding categorical features...")
cat_cols = ['city', 'state', 'weather', 'road_type', 'visibility',
            'traffic_density', 'festival', 'cause', 'day_of_week']
le = LabelEncoder()
for col in cat_cols:
    df[col + '_enc'] = le.fit_transform(df[col])

severity_map = {'minor': 0, 'major': 1, 'fatal': 2}
df['severity_label'] = df['accident_severity'].map(severity_map)

print("\n Class distribution:")
print(df['severity_label'].value_counts().to_string())

print("\n Applying Normalization (MinMaxScaler)...")

continuous_cols = ['temperature', 'risk_score', 'casualties',
                   'vehicles_involved', 'hour']

normalizer = MinMaxScaler()
df_normalized = df.copy()
df_normalized[continuous_cols] = normalizer.fit_transform(df[continuous_cols])

print(f"   Normalized columns: {continuous_cols}")
print(f"   All values scaled to [0, 1] range")


for col in continuous_cols:
    print(f"   {col}: min={df_normalized[col].min():.3f}, max={df_normalized[col].max():.3f}")


FEATURES = [
    'hour', 'is_weekend', 'is_peak_hour', 'temperature', 'lanes',
    'traffic_signal', 'vehicles_involved', 'casualties', 'risk_score',
    'month', 'weather_enc', 'road_type_enc', 'visibility_enc',
    'traffic_density_enc', 'festival_enc', 'cause_enc',
    'day_of_week_enc', 'city_enc', 'state_enc'
]

X = df_normalized[FEATURES]
y = df_normalized['severity_label']

print(f"\n Feature matrix shape : {X.shape}")
print(f" Target variable shape: {y.shape}")

print("\n Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train size: {len(X_train):,}")
print(f"   Test size : {len(X_test):,}")

print("\n Applying SMOTE for class balancing...")
print(f"   Before SMOTE: {dict(y_train.value_counts().sort_index())}")

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"   After SMOTE : {dict(pd.Series(y_train_balanced).value_counts().sort_index())}")
print(f"   Train size after SMOTE: {len(X_train_balanced):,}")


print("\n Fitting StandardScaler on balanced training data...")
scaler = StandardScaler()
X_train_balanced_sc = scaler.fit_transform(X_train_balanced)
X_test_sc           = scaler.transform(X_test)
print("   StandardScaler fitted — mean=0, std=1 transformation applied")

X_train_balanced_raw = X_train_balanced.copy()


def evaluate_model(model, X_test, y_test):
    y_pred  = model.predict(X_test)
    metrics = {
        'accuracy':           round(float(accuracy_score(y_test, y_pred)), 4),
        'precision_macro':    round(float(precision_score(y_test, y_pred, average='macro')), 4),
        'recall_macro':       round(float(recall_score(y_test, y_pred, average='macro')), 4),
        'f1_macro':           round(float(f1_score(y_test, y_pred, average='macro')), 4),
        'precision_weighted': round(float(precision_score(y_test, y_pred, average='weighted')), 4),
        'recall_weighted':    round(float(recall_score(y_test, y_pred, average='weighted')), 4),
        'f1_weighted':        round(float(f1_score(y_test, y_pred, average='weighted')), 4),
        'confusion_matrix':   confusion_matrix(y_test, y_pred).tolist()
    }
    try:
        y_proba = model.predict_proba(X_test)
        metrics['roc_auc'] = round(float(roc_auc_score(
            y_test, y_proba, multi_class='ovr')), 4)
    except:
        metrics['roc_auc'] = None
    return metrics

all_metrics = {}


print("\n" + "=" * 45)
print(" Random Forest Classifier")
print("=" * 45)
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_balanced_raw, y_train_balanced)
y_pred_rf = rf.predict(X_test)
print(f"Accuracy : {accuracy_score(y_test, y_pred_rf):.4f}")
print(classification_report(y_test, y_pred_rf,
      target_names=['minor', 'major', 'fatal']))
all_metrics['Random Forest'] = evaluate_model(rf, X_test, y_test)

print("\n" + "=" * 45)
print(" XGBoost Classifier")
print("=" * 45)
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss',
    n_jobs=-1
)
xgb.fit(X_train_balanced_raw, y_train_balanced)
y_pred_xgb = xgb.predict(X_test)
print(f"Accuracy : {accuracy_score(y_test, y_pred_xgb):.4f}")
print(classification_report(y_test, y_pred_xgb,
      target_names=['minor', 'major', 'fatal']))
all_metrics['XGBoost'] = evaluate_model(xgb, X_test, y_test)


print("\n" + "=" * 45)
print(" K-Nearest Neighbors")
print("=" * 45)
knn = KNeighborsClassifier(
    n_neighbors=7,
    weights='distance',
    n_jobs=-1
)
knn.fit(X_train_balanced_sc, y_train_balanced)
y_pred_knn = knn.predict(X_test_sc)
print(f"Accuracy : {accuracy_score(y_test, y_pred_knn):.4f}")
print(classification_report(y_test, y_pred_knn,
      target_names=['minor', 'major', 'fatal']))
all_metrics['KNN'] = evaluate_model(knn, X_test_sc, y_test)


print("\n" + "=" * 45)
print(" Logistic Regression")
print("=" * 45)
lr = LogisticRegression(
    max_iter=2000,
    C=1.0,
    n_jobs=-1
)
lr.fit(X_train_balanced_sc, y_train_balanced)
y_pred_lr = lr.predict(X_test_sc)
print(f"Accuracy : {accuracy_score(y_test, y_pred_lr):.4f}")
print(classification_report(y_test, y_pred_lr,
      target_names=['minor', 'major', 'fatal']))
all_metrics['Logistic Regression'] = evaluate_model(lr, X_test_sc, y_test)

print("\n" + "=" * 45)
print(" Clustering Analysis")
print("=" * 45)

kmeans_features = ['latitude', 'longitude', 'risk_score', 'hour', 'traffic_density_enc']
X_km     = df[kmeans_features] 
scaler_km = StandardScaler()
X_km_sc  = scaler_km.fit_transform(X_km)


kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
df['cluster_kmeans'] = kmeans.fit_predict(X_km_sc)


dbscan = DBSCAN(eps=0.3, min_samples=10)
df['cluster_dbscan'] = dbscan.fit_predict(X_km_sc)

print(f"KMeans clusters  : {len(df['cluster_kmeans'].unique())}")
print(f"DBSCAN clusters  : {len(df[df['cluster_dbscan'] != -1]['cluster_dbscan'].unique())}")
print(f"DBSCAN noise pts : {(df['cluster_dbscan'] == -1).sum()}")

print("\n" + "=" * 45)
print(" Feature Importance (Random Forest)")
print("=" * 45)
feature_importance = pd.DataFrame({
    'feature':    FEATURES,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance.head(10).to_string(index=False))

print("\n" + "=" * 45)
print(" Cross-Validation (5-fold)")
print("=" * 45)
cv_rf  = cross_val_score(rf,  X_train_balanced_raw,
                          y_train_balanced, cv=5,
                          scoring='accuracy', n_jobs=-1)
cv_xgb = cross_val_score(xgb, X_train_balanced_raw,
                          y_train_balanced, cv=5,
                          scoring='accuracy', n_jobs=-1)
print(f"Random Forest CV : {cv_rf.mean():.4f}  (+/- {cv_rf.std()*2:.4f})")
print(f"XGBoost CV       : {cv_xgb.mean():.4f}  (+/- {cv_xgb.std()*2:.4f})")

print("\n" + "=" * 45)
print(" Saving models & artifacts")
print("=" * 45)

# ── Save confusion matrices for UI display ────────────────────────
print("\n Computing confusion matrices...")

confusion_matrices = {}

# Random Forest
cm_rf = confusion_matrix(y_test, y_pred_rf)
confusion_matrices['Random Forest'] = {
    'matrix': cm_rf.tolist(),
    'accuracy': round(accuracy_score(y_test, y_pred_rf)*100, 1)
}

# XGBoost
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
confusion_matrices['XGBoost'] = {
    'matrix': cm_xgb.tolist(),
    'accuracy': round(accuracy_score(y_test, y_pred_xgb)*100, 1)
}

# KNN
cm_knn = confusion_matrix(y_test, y_pred_knn)
confusion_matrices['KNN'] = {
    'matrix': cm_knn.tolist(),
    'accuracy': round(accuracy_score(y_test, y_pred_knn)*100, 1)
}

# Logistic Regression
cm_lr = confusion_matrix(y_test, y_pred_lr)
confusion_matrices['Logistic Regression'] = {
    'matrix': cm_lr.tolist(),
    'accuracy': round(accuracy_score(y_test, y_pred_lr)*100, 1)
}

# Save as JSON
with open('models/confusion_matrices.json', 'w') as f:
    json.dump(confusion_matrices, f, indent=2)

print("   Confusion matrices saved to models/confusion_matrices.json")

os.makedirs('models', exist_ok=True)

joblib.dump(rf,           'models/rf_model.pkl')
joblib.dump(xgb,          'models/xgb_model.pkl')
joblib.dump(knn,          'models/knn_model.pkl')
joblib.dump(lr,           'models/lr_model.pkl')
joblib.dump(kmeans,       'models/kmeans_model.pkl')
joblib.dump(scaler,       'models/scaler.pkl')
joblib.dump(scaler_km,    'models/scaler_km.pkl')
joblib.dump(normalizer,   'models/normalizer.pkl')
joblib.dump(FEATURES,     'models/feature_names.pkl')

feature_importance.to_csv('models/feature_importance.csv', index=False)


def convert_metrics(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

with open('models/model_metrics.json', 'w') as f:
    json.dump(all_metrics, f, indent=2, default=convert_metrics)


clustered_data = df[['latitude', 'longitude', 'cluster_kmeans',
                      'cluster_dbscan', 'accident_severity',
                      'risk_score', 'city']]
joblib.dump(clustered_data, 'models/clustered_data.pkl')

print("\n All models saved:")
print("   models/rf_model.pkl")
print("   models/xgb_model.pkl")
print("   models/knn_model.pkl")
print("   models/lr_model.pkl")
print("   models/kmeans_model.pkl")
print("   models/scaler.pkl         — StandardScaler for KNN/LR")
print("   models/normalizer.pkl     — MinMaxScaler for continuous cols")
print("   models/scaler_km.pkl      — Scaler for clustering")
print("   models/feature_names.pkl")
print("   models/feature_importance.csv")
print("   models/model_metrics.json")
print("   models/clustered_data.pkl")

print("\n" + "=" * 60)
print(" MODEL PERFORMANCE SUMMARY")
print("=" * 60)
for name, m in all_metrics.items():
    print(f"\n {name}:")
    print(f"   Accuracy  : {m['accuracy']:.4f}")
    print(f"   Precision : {m['precision_weighted']:.4f}")
    print(f"   Recall    : {m['recall_weighted']:.4f}")
    print(f"   F1-Score  : {m['f1_weighted']:.4f}")
    if m['roc_auc']:
        print(f"   ROC-AUC   : {m['roc_auc']:.4f}")

print("\n" + "=" * 60)
print(" TRAINING COMPLETE!")
print("=" * 60)

print("\n DATA ANALYTICS TECHNIQUES APPLIED:")
print("   1. IQR Outlier Detection & Removal")
print("   2. Missing Value Imputation (festival → none)")
print("   3. Label Encoding (9 categorical columns)")
print("   4. MinMaxScaler Normalization (continuous features)")
print("   5. Stratified Train/Test Split (80/20)")
print("   6. SMOTE Oversampling (class balancing)")
print("   7. StandardScaler Standardization (KNN, LR)")
print("   8. Random Forest Classification")
print("   9. XGBoost Classification")
print("   10. KNN Classification")
print("   11. Logistic Regression Classification")
print("   12. K-Means Clustering (8 clusters)")
print("   13. DBSCAN Clustering (anomaly detection)")
print("   14. Feature Importance Analysis")
print("   15. 5-Fold Cross Validation")
print("   16. ROC-AUC, F1, Precision, Recall Evaluation")