import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MinMaxScaler
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
print("INDIAN ROAD ACCIDENT RISK ANALYZER - MODEL TRAINING (v3)")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# 1. OUTLIER REMOVAL (IQR)
# ─────────────────────────────────────────────────────────────
def remove_outliers_iqr(df, exclude_cols=None):
    """Remove outliers using IQR method on numeric columns only."""
    if exclude_cols is None:
        exclude_cols = []
    original_size = len(df)
    numeric_cols  = df.select_dtypes(include=['int64', 'float64']).columns
    cols_to_check = [c for c in numeric_cols if c not in exclude_cols]

    outlier_summary = {}
    for col in cols_to_check:
        Q1    = df[col].quantile(0.25)
        Q3    = df[col].quantile(0.75)
        IQR   = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        count = ((df[col] < lower) | (df[col] > upper)).sum()
        if count > 0:
            outlier_summary[col] = count
        df = df[(df[col] >= lower) & (df[col] <= upper)]

    removed = original_size - len(df)
    print(f"\n IQR Outlier Detection:")
    print(f"   Original rows  : {original_size:,}")
    print(f"   Rows removed   : {removed:,}")
    print(f"   Remaining rows : {len(df):,}")
    if outlier_summary:
        print(f"   Outliers per column:")
        for col, cnt in outlier_summary.items():
            print(f"     {col}: {cnt}")
    return df


# ─────────────────────────────────────────────────────────────
# 2. LOAD & CLEAN
# ─────────────────────────────────────────────────────────────
print("\n Loading dataset...")
df = pd.read_csv("data/indian_roads_dataset.csv")
print(f"   Loaded {len(df):,} rows, {len(df.columns)} columns")

# Save city column BEFORE get_dummies removes it (needed for clustered_data)
df['city_label'] = df['city'].copy()

# Fill missing festival
df['festival'] = df['festival'].fillna('none')

# Parse date and extract month
df['date']  = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month

exclude_from_iqr = [
    'accident_id', 'is_weekend', 'is_peak_hour',
    'traffic_signal', 'lanes', 'month'
]
df = remove_outliers_iqr(df, exclude_cols=exclude_from_iqr)


# ─────────────────────────────────────────────────────────────
# 3. PROPER ENCODING
# ─────────────────────────────────────────────────────────────
print("\n Encoding categorical features...")

# 3a. Ordinal — only where a real order exists
ordinal_map = {
    'visibility':      ['low', 'medium', 'high'],
    'traffic_density': ['low', 'medium', 'high'],
}
for col, order in ordinal_map.items():
    enc = OrdinalEncoder(categories=[order])
    df[col + '_enc'] = enc.fit_transform(df[[col]])
    print(f"   OrdinalEncoder  → {col}_enc  {order}")

# 3b. One-hot — all nominal categoricals (low cardinality)
nominal_cols = ['weather', 'road_type', 'cause', 'festival',
                'city', 'state', 'day_of_week']
df = pd.get_dummies(df, columns=nominal_cols, drop_first=False, dtype=int)
ohe_cols = [c for c in df.columns
            if any(c.startswith(n + '_') for n in nominal_cols)
            and df[c].dtype != object
            and c != 'city_label']
print(f"   OneHotEncoding  → {len(ohe_cols)} columns for {nominal_cols}")

# 3c. Target label
severity_map     = {'minor': 0, 'major': 1, 'fatal': 2}
df['severity_label'] = df['accident_severity'].map(severity_map)

print("\n Class distribution:")
print(df['severity_label'].value_counts().sort_index().to_string())


# ─────────────────────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
print("\n Engineering new features...")

# Ratio / interaction features
df['casualty_per_vehicle'] = (
    df['casualties'] / df['vehicles_involved'].clip(lower=1)
)
df['risk_x_casualty']  = df['risk_score'] * df['casualties']
df['risk_x_density']   = df['risk_score'] * df['traffic_density_enc']
df['speed_proxy']       = df['lanes']      * df['risk_score']

# Time-of-day bin (right=False → [0,6) is night, etc.)
df['hour_bin'] = pd.cut(
    df['hour'],
    bins=[0, 6, 10, 16, 20, 24],
    labels=[0, 1, 2, 3, 4],
    right=False
).astype(np.int32)

# Amplified weights for the two strongest predictors
df['casualties_w'] = df['casualties']  * 3
df['risk_score_w'] = df['risk_score']  * 3

# Binary flag for clearest fatal signature
df['fatal_risk_flag'] = (
    (df['risk_score'] > 0.6) & (df['casualties'] >= 2)
).astype(int)

# NEW: major-class discriminators
# casualties==1 AND risk_score < 0.5 → most likely minor
df['minor_flag'] = (
    (df['casualties'] <= 1) & (df['risk_score'] < 0.5)
).astype(int)
# vehicles>3 AND casualties>1 → major signal
df['major_proxy'] = (
    (df['vehicles_involved'] > 3) & (df['casualties'] > 1) &
    (df['risk_score'] < 0.6)
).astype(int)
# risk_score bucket: 0=low(0-0.3), 1=mid(0.3-0.6), 2=high(0.6-1.0)
df['risk_bucket'] = pd.cut(
    df['risk_score'],
    bins=[0, 0.3, 0.6, 1.01],
    labels=[0, 1, 2],
    right=False
).astype(np.int32)

new_feats = [
    'casualty_per_vehicle', 'risk_x_casualty', 'risk_x_density',
    'speed_proxy', 'hour_bin', 'casualties_w', 'risk_score_w',
    'fatal_risk_flag', 'minor_flag', 'major_proxy', 'risk_bucket'
]
print(f"   {len(new_feats)} new features created:")
for f in new_feats:
    print(f"     {f}")


# ─────────────────────────────────────────────────────────────
# 5. NORMALIZATION  (MinMaxScaler — continuous cols only)
# ─────────────────────────────────────────────────────────────
print("\n Applying MinMaxScaler normalization...")

continuous_cols = [
    'temperature', 'risk_score', 'casualties',
    'vehicles_involved', 'hour',
    'casualty_per_vehicle', 'risk_x_casualty', 'risk_x_density',
    'speed_proxy', 'casualties_w', 'risk_score_w'
]
normalizer = MinMaxScaler()
df[continuous_cols] = normalizer.fit_transform(df[continuous_cols])

print(f"   Normalized {len(continuous_cols)} continuous columns to [0, 1]")
for col in continuous_cols:
    print(f"   {col}: min={df[col].min():.3f}, max={df[col].max():.3f}")


# ─────────────────────────────────────────────────────────────
# 6. FEATURE LIST
# ─────────────────────────────────────────────────────────────
core_features = [
    'hour', 'is_weekend', 'is_peak_hour', 'temperature', 'lanes',
    'traffic_signal', 'vehicles_involved', 'casualties', 'risk_score',
    'month', 'visibility_enc', 'traffic_density_enc',
]
engineered_features = new_feats   # all 11 engineered columns

FEATURES = core_features + engineered_features + ohe_cols

# Safety check — cast every feature to float32 before SMOTE.
# pd.cut produces Categorical dtype even after .astype(int) on some
# pandas versions, and any stray object column will crash SMOTE.
X = df[FEATURES].copy().astype(np.float32)
y = df['severity_label']

print(f"\n Feature matrix : {X.shape[0]:,} rows × {X.shape[1]} features")
print(f"   Core       : {len(core_features)}")
print(f"   Engineered : {len(engineered_features)}")
print(f"   OHE        : {len(ohe_cols)}")


# ─────────────────────────────────────────────────────────────
# 7. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────
print("\n Splitting data (80/20 stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train : {len(X_train):,}  |  Test : {len(X_test):,}")


# ─────────────────────────────────────────────────────────────
# 8. SMOTE  (applied to training set ONLY)
# ─────────────────────────────────────────────────────────────
print("\n Applying SMOTE (training set only)...")
print(f"   Before: {dict(y_train.value_counts().sort_index())}")
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print(f"   After : {dict(pd.Series(y_train_bal).value_counts().sort_index())}")
print(f"   Size  : {len(X_train_bal):,}")


# ─────────────────────────────────────────────────────────────
# 9. STANDARDSCALER  (KNN and LR only — fit on SMOTE train)
# ─────────────────────────────────────────────────────────────
print("\n Fitting StandardScaler (KNN / LR)...")
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_bal)
X_test_sc  = scaler.transform(X_test)
print("   Done — mean=0, std=1")

# Tree models use raw balanced data (scale-invariant)
X_train_raw = X_train_bal.copy()


# ─────────────────────────────────────────────────────────────
# 10. EVALUATION HELPER
# ─────────────────────────────────────────────────────────────
def evaluate_model(model, X_te, y_te):
    y_pred = model.predict(X_te)
    metrics = {
        'accuracy':           round(float(accuracy_score(y_te, y_pred)), 4),
        'precision_macro':    round(float(precision_score(y_te, y_pred, average='macro')), 4),
        'recall_macro':       round(float(recall_score(y_te, y_pred, average='macro')), 4),
        'f1_macro':           round(float(f1_score(y_te, y_pred, average='macro')), 4),
        'precision_weighted': round(float(precision_score(y_te, y_pred, average='weighted')), 4),
        'recall_weighted':    round(float(recall_score(y_te, y_pred, average='weighted')), 4),
        'f1_weighted':        round(float(f1_score(y_te, y_pred, average='weighted')), 4),
        'confusion_matrix':   confusion_matrix(y_te, y_pred).tolist()
    }
    try:
        y_proba = model.predict_proba(X_te)
        metrics['roc_auc'] = round(float(roc_auc_score(
            y_te, y_proba, multi_class='ovr')), 4)
    except Exception:
        metrics['roc_auc'] = None
    return metrics


all_metrics = {}


# ─────────────────────────────────────────────────────────────
# 11. RANDOM FOREST
#     — reduced depth + more estimators to reduce overfitting
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print(" Random Forest Classifier")
print("=" * 50)
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=12,            # reduced from 20 → less overfitting on SMOTE data
    min_samples_split=10,    # increased → smoother decision boundaries
    min_samples_leaf=5,      # increased → prevents tiny over-specific leaves
    max_features='sqrt',
    class_weight='balanced', # extra push for major class
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_raw, y_train_bal)
y_pred_rf = rf.predict(X_test)
print(f"Accuracy : {accuracy_score(y_test, y_pred_rf):.4f}")
print(classification_report(y_test, y_pred_rf,
      target_names=['minor', 'major', 'fatal']))
all_metrics['Random Forest'] = evaluate_model(rf, X_test, y_test)


# ─────────────────────────────────────────────────────────────
# 12. XGBOOST
#     — scale_pos_weight per class via sample_weight trick
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print(" XGBoost Classifier")
print("=" * 50)

# Compute per-sample weights to give major class extra emphasis
class_counts   = pd.Series(y_train_bal).value_counts()
total          = len(y_train_bal)
sample_weights = pd.Series(y_train_bal).map({
    0: total / (3 * class_counts[0]),   # minor  — normal weight
    1: total / (3 * class_counts[1]) * 1.5,  # major  — 1.5× boost
    2: total / (3 * class_counts[2]),   # fatal  — normal weight
}).values

xgb = XGBClassifier(
    n_estimators=400,
    max_depth=6,             # reduced from 8 → less overfit
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,      # increased → avoids memorising SMOTE noise
    gamma=0.2,
    reg_alpha=0.1,           # L1 regularisation
    reg_lambda=1.5,          # L2 regularisation
    random_state=42,
    eval_metric='mlogloss',
    n_jobs=-1
)
xgb.fit(X_train_raw, y_train_bal, sample_weight=sample_weights)
y_pred_xgb = xgb.predict(X_test)
print(f"Accuracy : {accuracy_score(y_test, y_pred_xgb):.4f}")
print(classification_report(y_test, y_pred_xgb,
      target_names=['minor', 'major', 'fatal']))
all_metrics['XGBoost'] = evaluate_model(xgb, X_test, y_test)


# ─────────────────────────────────────────────────────────────
# 13. KNN
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print(" K-Nearest Neighbors")
print("=" * 50)
knn = KNeighborsClassifier(
    n_neighbors=7,
    weights='distance',
    n_jobs=-1
)
knn.fit(X_train_sc, y_train_bal)
y_pred_knn = knn.predict(X_test_sc)
print(f"Accuracy : {accuracy_score(y_test, y_pred_knn):.4f}")
print(classification_report(y_test, y_pred_knn,
      target_names=['minor', 'major', 'fatal']))
all_metrics['KNN'] = evaluate_model(knn, X_test_sc, y_test)


# ─────────────────────────────────────────────────────────────
# 14. LOGISTIC REGRESSION
#     — class_weight to force it to predict major
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print(" Logistic Regression")
print("=" * 50)
lr = LogisticRegression(
    max_iter=2000,
    C=1.0,
    class_weight='balanced',  # fixes the 0-recall on major class
    solver='lbfgs',
    n_jobs=-1
)
lr.fit(X_train_sc, y_train_bal)
y_pred_lr = lr.predict(X_test_sc)
print(f"Accuracy : {accuracy_score(y_test, y_pred_lr):.4f}")
print(classification_report(y_test, y_pred_lr,
      target_names=['minor', 'major', 'fatal']))
all_metrics['Logistic Regression'] = evaluate_model(lr, X_test_sc, y_test)


# ─────────────────────────────────────────────────────────────
# 15. CLUSTERING  (KMeans + DBSCAN)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print(" Clustering Analysis")
print("=" * 50)

kmeans_features = ['latitude', 'longitude', 'risk_score', 'hour',
                   'traffic_density_enc']
X_km      = df[kmeans_features]
scaler_km = StandardScaler()
X_km_sc   = scaler_km.fit_transform(X_km)

kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
df['cluster_kmeans'] = kmeans.fit_predict(X_km_sc)

dbscan = DBSCAN(eps=0.3, min_samples=10)
df['cluster_dbscan'] = dbscan.fit_predict(X_km_sc)

print(f"KMeans clusters  : {df['cluster_kmeans'].nunique()}")
print(f"DBSCAN clusters  : {df[df['cluster_dbscan'] != -1]['cluster_dbscan'].nunique()}")
print(f"DBSCAN noise pts : {(df['cluster_dbscan'] == -1).sum()}")


# ─────────────────────────────────────────────────────────────
# 16. FEATURE IMPORTANCE  (Random Forest — top 15)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print(" Feature Importance (Random Forest) — top 15")
print("=" * 50)
feature_importance = pd.DataFrame({
    'feature':    FEATURES,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance.head(15).to_string(index=False))


# ─────────────────────────────────────────────────────────────
# 17. CROSS-VALIDATION  (5-fold stratified)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print(" Cross-Validation (5-fold Stratified)")
print("=" * 50)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_rf  = cross_val_score(rf,  X_train_raw, y_train_bal,
                          cv=skf, scoring='accuracy', n_jobs=-1)
cv_xgb = cross_val_score(xgb, X_train_raw, y_train_bal,
                          cv=skf, scoring='accuracy', n_jobs=-1)
print(f"Random Forest CV : {cv_rf.mean():.4f}  (+/- {cv_rf.std()*2:.4f})")
print(f"XGBoost CV       : {cv_xgb.mean():.4f}  (+/- {cv_xgb.std()*2:.4f})")


# ─────────────────────────────────────────────────────────────
# 18. CONFUSION MATRICES
# ─────────────────────────────────────────────────────────────
print("\n Computing confusion matrices...")
confusion_matrices = {
    'Random Forest': {
        'matrix':   confusion_matrix(y_test, y_pred_rf).tolist(),
        'accuracy': round(accuracy_score(y_test, y_pred_rf) * 100, 1)
    },
    'XGBoost': {
        'matrix':   confusion_matrix(y_test, y_pred_xgb).tolist(),
        'accuracy': round(accuracy_score(y_test, y_pred_xgb) * 100, 1)
    },
    'KNN': {
        'matrix':   confusion_matrix(y_test, y_pred_knn).tolist(),
        'accuracy': round(accuracy_score(y_test, y_pred_knn) * 100, 1)
    },
    'Logistic Regression': {
        'matrix':   confusion_matrix(y_test, y_pred_lr).tolist(),
        'accuracy': round(accuracy_score(y_test, y_pred_lr) * 100, 1)
    },
}
print("   Done.")


# ─────────────────────────────────────────────────────────────
# 19. SAVE ALL ARTIFACTS
# ─────────────────────────────────────────────────────────────
os.makedirs('models', exist_ok=True)

joblib.dump(rf,               'models/rf_model.pkl')
joblib.dump(xgb,              'models/xgb_model.pkl')
joblib.dump(knn,              'models/knn_model.pkl')
joblib.dump(lr,               'models/lr_model.pkl')
joblib.dump(kmeans,           'models/kmeans_model.pkl')
joblib.dump(scaler,           'models/scaler.pkl')
joblib.dump(scaler_km,        'models/scaler_km.pkl')
joblib.dump(normalizer,       'models/normalizer.pkl')
joblib.dump(FEATURES,         'models/feature_names.pkl')
joblib.dump(ohe_cols,         'models/ohe_cols.pkl')

feature_importance.to_csv('models/feature_importance.csv', index=False)

def convert_metrics(obj):
    if isinstance(obj, (np.integer,  np.int64)):   return int(obj)
    if isinstance(obj, (np.floating, np.float64)): return float(obj)
    if isinstance(obj, np.ndarray):                return obj.tolist()
    return obj

with open('models/model_metrics.json', 'w') as f:
    json.dump(all_metrics, f, indent=2, default=convert_metrics)

with open('models/confusion_matrices.json', 'w') as f:
    json.dump(confusion_matrices, f, indent=2)

# Use city_label (saved before get_dummies) for clustered_data
clustered_data = df[['latitude', 'longitude', 'cluster_kmeans',
                      'cluster_dbscan', 'accident_severity',
                      'risk_score', 'city_label']].rename(
                          columns={'city_label': 'city'})
joblib.dump(clustered_data, 'models/clustered_data.pkl')

print("\n All models and artifacts saved:")
print("   models/rf_model.pkl")
print("   models/xgb_model.pkl")
print("   models/knn_model.pkl")
print("   models/lr_model.pkl")
print("   models/kmeans_model.pkl")
print("   models/scaler.pkl             — StandardScaler (KNN / LR)")
print("   models/normalizer.pkl         — MinMaxScaler (continuous cols)")
print("   models/scaler_km.pkl          — StandardScaler (clustering)")
print("   models/feature_names.pkl      — full FEATURES list")
print("   models/ohe_cols.pkl           — one-hot column names")
print("   models/feature_importance.csv")
print("   models/model_metrics.json")
print("   models/confusion_matrices.json")
print("   models/clustered_data.pkl")


# ─────────────────────────────────────────────────────────────
# 20. FINAL SUMMARY
# ─────────────────────────────────────────────────────────────
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

print("""
 DATA ANALYTICS TECHNIQUES APPLIED:
   1.  IQR Outlier Detection & Removal
   2.  Missing Value Imputation (festival → none)
   3a. OrdinalEncoder    — visibility, traffic_density (true order)
   3b. OneHotEncoder     — weather, road_type, cause, festival,
                           city, state, day_of_week (nominal)
   4.  Feature Engineering (11 new features)
         casualty_per_vehicle  = casualties / vehicles_involved
         risk_x_casualty       = risk_score × casualties
         risk_x_density        = risk_score × traffic_density_enc
         speed_proxy           = lanes × risk_score
         hour_bin              = time-of-day bucket [0–4]
         casualties_w          = casualties × 3  (amplified)
         risk_score_w          = risk_score  × 3  (amplified)
         fatal_risk_flag       = risk>0.6 AND casualties≥2
         minor_flag            = casualties≤1 AND risk<0.5
         major_proxy           = vehicles>3 AND casualties>1 AND risk<0.6
         risk_bucket           = risk discretized [low/mid/high]
   5.  MinMaxScaler Normalization (continuous features)
   6.  Stratified Train/Test Split (80/20)
   7.  SMOTE Oversampling (training set only)
   8.  StandardScaler Standardization (KNN, LR only)
   9.  Random Forest   (500 trees, depth 12, class_weight=balanced)
   10. XGBoost         (400 trees, depth 6, sample_weight for major boost)
   11. KNN             (k=7, distance weights)
   12. Logistic Regression (class_weight=balanced, multinomial)
   13. K-Means Clustering  (8 clusters)
   14. DBSCAN Clustering   (anomaly detection)
   15. Feature Importance Analysis (RF)
   16. 5-Fold Stratified Cross-Validation
   17. ROC-AUC, F1, Precision, Recall Evaluation
""")