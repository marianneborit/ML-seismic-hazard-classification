"""
Machine Learning-Based Seismic Hazard Classification and Spatial Clustering
Analysis of Peruvian Earthquakes: A 25-Year USGS Catalog Study (2000-2025)

Author: Marianne Borit
Institution: Universidad del Pacifico, Lima, Peru
Conference: LACCI 2026
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_squared_error, r2_score,
                             accuracy_score, classification_report,
                             ConfusionMatrixDisplay)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from xgboost import XGBClassifier, XGBRegressor
from imblearn.over_sampling import SMOTE

# Create figures directory
os.makedirs('figures', exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("Loading data from USGS API...")
url = (
    "https://earthquake.usgs.gov/fdsnws/event/1/query.csv"
    "?starttime=2000-01-01%2000:00:00"
    "&endtime=2025-12-31%2023:59:59"
    "&maxlatitude=-0.5&minlatitude=-18.5"
    "&maxlongitude=-68.5&minlongitude=-81.5"
    "&minmagnitude=3&maxmagnitude=9.9&orderby=time"
)
df = pd.read_csv(url)
print(f"Dataset loaded: {df.shape[0]} events, {df.shape[1]} features")

# ============================================================
# 2. PREPROCESSING
# ============================================================
print("\nPreprocessing...")

df['time'] = pd.to_datetime(df['time'])
df['year']  = df['time'].dt.year
df['month'] = df['time'].dt.month
df['hour']  = df['time'].dt.hour

df_clean = df[['latitude', 'longitude', 'depth',
               'mag', 'year', 'month', 'hour']].dropna()

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
print("Engineering features...")

# Distance to Nazca subduction reference point
nazca_lat, nazca_lon = -12.0, -77.5
df_clean = df_clean.copy()
df_clean['dist_nazca'] = np.sqrt(
    (df_clean['latitude']  - nazca_lat)**2 +
    (df_clean['longitude'] - nazca_lon)**2
)

# Latitudinal zone
def zona(lat):
    if lat > -8:    return 0   # North
    elif lat > -14: return 1   # Central
    else:           return 2   # South

df_clean['zona'] = df_clean['latitude'].apply(zona)

# Depth category
def tipo_profundidad(d):
    if d < 70:    return 0   # Shallow
    elif d < 300: return 1   # Intermediate
    else:         return 2   # Deep

df_clean['tipo_prof'] = df_clean['depth'].apply(tipo_profundidad)

# Zone seismic density
df_clean['densidad'] = df_clean.groupby('zona')['zona'].transform('count')

# Polynomial and interaction terms
df_clean['lat_lon_interaction']  = df_clean['latitude']  * df_clean['longitude']
df_clean['depth_squared']        = df_clean['depth']     ** 2
df_clean['dist_nazca_squared']   = df_clean['dist_nazca']** 2

# Zone-level magnitude statistics
df_clean['mag_mean_zona'] = df_clean.groupby('zona')['mag'].transform('mean')
df_clean['mag_std_zona']  = df_clean.groupby('zona')['mag'].transform('std')

print(f"Features engineered. Final shape: {df_clean.shape}")

# ============================================================
# 4. HAZARD LABELS
# ============================================================
def hazard_level(mag):
    if mag < 4.0: return 0   # Low
    elif mag < 5.0: return 1  # Moderate
    else: return 2            # High

df_clean['hazard'] = df_clean['mag'].apply(hazard_level)

print("\nClass distribution:")
print(df_clean['hazard'].value_counts())
print(df_clean['hazard'].value_counts(normalize=True).round(3))

# ============================================================
# 5. REGRESSION BASELINE
# ============================================================
print("\n--- Regression Baseline ---")

X_reg = df_clean[['latitude', 'longitude', 'depth', 'year', 'month', 'hour']]
y_reg = df_clean['mag']
X_tr, X_te, y_tr, y_te = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

reg_models = {
    'Ridge Regression': Pipeline([('scaler', StandardScaler()), ('model', Ridge())]),
    'Random Forest':    RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting':GradientBoostingRegressor(n_estimators=100, random_state=42),
    'SVR':              Pipeline([('scaler', StandardScaler()), ('model', SVR(kernel='rbf'))])
}

reg_results = {}
for name, model in reg_models.items():
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    rmse = np.sqrt(mean_squared_error(y_te, pred))
    r2   = r2_score(y_te, pred)
    reg_results[name] = {'RMSE': round(rmse, 4), 'R2': round(r2, 4)}
    print(f"{name}: RMSE={rmse:.4f}, R2={r2:.4f}")

# Best regression model for actual vs predicted plot
rf_reg = reg_models['Random Forest']
y_pred_reg = rf_reg.predict(X_te)

# ============================================================
# 6. CLASSIFICATION
# ============================================================
print("\n--- Classification ---")

FEATURES = ['latitude', 'longitude', 'depth', 'year', 'month', 'hour',
            'dist_nazca', 'zona', 'tipo_prof', 'densidad',
            'lat_lon_interaction', 'depth_squared', 'dist_nazca_squared',
            'mag_mean_zona', 'mag_std_zona']

X_clf = df_clean[FEATURES]
y_clf = df_clean['hazard']

# SMOTE balancing
sm = SMOTE(random_state=42)
X_bal, y_bal = sm.fit_resample(X_clf, y_clf)
print(f"After SMOTE: {pd.Series(y_bal).value_counts().to_dict()}")

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_bal, y_bal, test_size=0.2, random_state=42)

classifiers = {
    'Random Forest':    RandomForestClassifier(n_estimators=200, random_state=42),
    'Gradient Boosting':GradientBoostingClassifier(n_estimators=200, random_state=42),
    'XGBoost':          XGBClassifier(n_estimators=200, learning_rate=0.05,
                                      random_state=42, eval_metric='mlogloss')
}

clf_results = {}
for name, clf in classifiers.items():
    clf.fit(X_train_c, y_train_c)
    pred = clf.predict(X_test_c)
    acc  = accuracy_score(y_test_c, pred)
    clf_results[name] = acc
    print(f"\n{name}: Accuracy={acc:.4f}")
    print(classification_report(y_test_c, pred,
          target_names=['Low', 'Moderate', 'High']))

best_clf  = classifiers['Random Forest']
best_pred = best_clf.predict(X_test_c)

# ============================================================
# 7. CLUSTERING
# ============================================================
print("\n--- Clustering ---")

X_cluster = df_clean[['latitude', 'longitude', 'depth', 'mag']].copy()
scaler    = StandardScaler()
X_scaled  = scaler.fit_transform(X_cluster)

inertias    = []
silhouettes = []
K = range(2, 10)
for k in K:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, km.labels_))

print("Silhouette scores:", [round(s, 3) for s in silhouettes])

km_final = KMeans(n_clusters=3, random_state=42, n_init=10)
df_clean['cluster'] = km_final.fit_predict(X_scaled)

print("\nCluster profiles:")
print(df_clean.groupby('cluster')[['latitude','longitude','depth','mag']].mean().round(3))

# ============================================================
# 8. FIGURES
# ============================================================
print("\nGenerating figures...")

COLORS        = ['steelblue', 'coral', 'green']
CLUSTER_LABELS = ['Southern Zone', 'Northern Zone', 'Deep Amazon']

# --- Fig 1: Exploratory Analysis ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.hist(df_clean['mag'], bins=30, color='steelblue', edgecolor='black')
ax1.set_xlabel('Magnitude')
ax1.set_ylabel('Frequency')
ax1.set_title('(a) Magnitude Distribution')
sc = ax2.scatter(df_clean['longitude'], df_clean['latitude'],
                 c=df_clean['mag'], cmap='hot_r', alpha=0.4, s=10)
plt.colorbar(sc, ax=ax2, label='Magnitude')
ax2.set_xlabel('Longitude')
ax2.set_ylabel('Latitude')
ax2.set_title('(b) Geographic Distribution')
plt.tight_layout()
plt.savefig('figures/Fig1_exploratory.png', dpi=150)
plt.show()
print("Fig 1 saved")

# --- Fig 2: Classification Results ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
model_names = ['Gradient\nBoosting', 'XGBoost', 'Random\nForest']
accuracies  = [clf_results['Gradient Boosting'],
               clf_results['XGBoost'],
               clf_results['Random Forest']]
bars = ax1.bar(model_names, accuracies, color=['lightgreen', 'coral', 'steelblue'])
ax1.set_ylim(0.5, 1.0)
ax1.set_ylabel('Accuracy')
ax1.set_title('(a) Model Comparison')
for bar, acc in zip(bars, accuracies):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{acc:.1%}', ha='center', fontweight='bold')
ConfusionMatrixDisplay.from_predictions(
    y_test_c, best_pred,
    display_labels=['Low', 'Moderate', 'High'],
    cmap='Blues', ax=ax2)
ax2.set_title('(b) Confusion Matrix — Random Forest')
plt.tight_layout()
plt.savefig('figures/Fig2_classification.png', dpi=150)
plt.show()
print("Fig 2 saved")

# --- Fig 3: Feature Importance ---
imp_clf = best_clf.feature_importances_
idx_clf = np.argsort(imp_clf)
plt.figure(figsize=(10, 6))
plt.barh([FEATURES[i] for i in idx_clf], imp_clf[idx_clf], color='steelblue')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('figures/Fig3_feature_importance.png', dpi=150)
plt.show()
print("Fig 3 saved")

# --- Fig 4: Clustering ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.plot(K, inertias, 'bo-', label='Inertia')
ax1_twin = ax1.twinx()
ax1_twin.plot(K, silhouettes, 'ro-', label='Silhouette')
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Inertia', color='blue')
ax1_twin.set_ylabel('Silhouette Score', color='red')
ax1.set_title('(a) Optimal Cluster Selection')
for i in range(3):
    mask = df_clean['cluster'] == i
    ax2.scatter(df_clean[mask]['longitude'], df_clean[mask]['latitude'],
                c=COLORS[i], label=CLUSTER_LABELS[i], alpha=0.4, s=10)
ax2.set_xlabel('Longitude')
ax2.set_ylabel('Latitude')
ax2.set_title('(b) Seismic Clusters in Peru')
ax2.legend()
plt.tight_layout()
plt.savefig('figures/Fig4_clustering.png', dpi=150)
plt.show()
print("Fig 4 saved")

# --- Fig 5: Cluster Analysis ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
df_clean.boxplot(column='mag', by='cluster', ax=ax1)
ax1.set_title('(a) Magnitude by Cluster')
ax1.set_xlabel('Cluster')
ax1.set_ylabel('Magnitude')
plt.suptitle('')
for i in range(3):
    mask = df_clean['cluster'] == i
    yearly = df_clean[mask].groupby('year').size()
    ax2.plot(yearly.index, yearly.values, marker='o',
             label=CLUSTER_LABELS[i], linewidth=2, color=COLORS[i])
ax2.set_title('(b) Cluster Activity Over Time')
ax2.set_xlabel('Year')
ax2.set_ylabel('Number of Events')
ax2.legend()
plt.tight_layout()
plt.savefig('figures/Fig5_cluster_analysis.png', dpi=150)
plt.show()
print("Fig 5 saved")

# --- Fig 6: Temporal Analysis ---
yearly_counts = df_clean.groupby('year').size()
yearly_hazard = df_clean.groupby(['year','hazard']).size().unstack(fill_value=0)
yearly_hazard.columns = ['Low','Moderate','High']
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
ax1.bar(yearly_counts.index, yearly_counts.values, color='steelblue', alpha=0.7)
ax1.plot(yearly_counts.index, yearly_counts.values, 'ro-', linewidth=2)
ax1.set_ylabel('Number of Events')
ax1.set_title('(a) Annual Seismic Event Frequency')
yearly_hazard.plot(kind='bar', stacked=True, colormap='RdYlGn_r', ax=ax2)
ax2.set_xlabel('Year')
ax2.set_ylabel('Number of Events')
ax2.set_title('(b) Annual Hazard Class Distribution')
plt.tight_layout()
plt.savefig('figures/Fig6_temporal.png', dpi=150)
plt.show()
print("Fig 6 saved")

print("\n✅ All figures saved to /figures/")
print("\n✅ Pipeline complete!")
