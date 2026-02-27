"""
F1 Full Stack ML Project — Model Training Pipeline
====================================================
Trains 3 ML models on F1 data:
  1. REGRESSION   — Predict lap time (milliseconds) using Random Forest
  2. CLASSIFICATION — Predict podium finish (top-3) using Gradient Boosting
  3. CLUSTERING   — Profile F1 drivers into clusters using KMeans

Outputs:
  - Trained model files (.pkl) in models/
  - Evaluation plots (.png) in static/plots/
  - Metrics JSON in static/metrics.json
"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, silhouette_score
)
from sklearn.decomposition import PCA
import joblib

warnings.filterwarnings('ignore')

# ── paths ──────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
DATA      = os.path.join(BASE, 'datasets')
MODEL_DIR = os.path.join(BASE, 'models')
PLOT_DIR  = os.path.join(BASE, 'static', 'plots')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ── plot styling ───────────────────────────────────────────────
plt.style.use('dark_background')
COLORS = {
    'primary': '#E10600',   # F1 red
    'accent':  '#00D2BE',   # Mercedes teal
    'gold':    '#FFD700',
    'silver':  '#C0C0C0',
    'bg':      '#1A1A2E',
    'grid':    '#333355',
}

all_metrics = {}

# ================================================================
#  1. REGRESSION — Predict Lap Time (ms)
# ================================================================
print("\n" + "=" * 70)
print("  TASK 1: REGRESSION — Predicting F1 Lap Times")
print("=" * 70)

# Load & merge data
lap_times    = pd.read_csv(os.path.join(DATA, 'f1_world_championship', 'lap_times.csv'))
races        = pd.read_csv(os.path.join(DATA, 'f1_world_championship', 'races.csv'))
circuits     = pd.read_csv(os.path.join(DATA, 'f1_world_championship', 'circuits.csv'))
results      = pd.read_csv(os.path.join(DATA, 'f1_world_championship', 'results.csv'))

# Merge lap_times ← races ← circuits
df_reg = lap_times.merge(races[['raceId', 'year', 'circuitId', 'round']], on='raceId', how='left')
df_reg = df_reg.merge(circuits[['circuitId', 'circuitRef']], on='circuitId', how='left')

# Get qualifying grid position per driver per race from results
grid_info = results[['raceId', 'driverId', 'grid', 'constructorId']].drop_duplicates()
df_reg = df_reg.merge(grid_info, on=['raceId', 'driverId'], how='left')

# Encode circuit
le_circuit = LabelEncoder()
df_reg['circuit_enc'] = le_circuit.fit_transform(df_reg['circuitRef'].fillna('unknown'))

# Drop rows with missing target
df_reg = df_reg.dropna(subset=['milliseconds'])
df_reg['milliseconds'] = df_reg['milliseconds'].astype(int)

# Feature engineering
features_reg = ['year', 'round', 'lap', 'position', 'circuit_enc', 'grid', 'constructorId']
X_reg = df_reg[features_reg].fillna(0)
y_reg = df_reg['milliseconds']

print(f"  Dataset size: {X_reg.shape[0]:,} rows, {X_reg.shape[1]} features")

# Train / test split
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Scale
scaler_reg = StandardScaler()
X_train_rs = scaler_reg.fit_transform(X_train_r)
X_test_rs  = scaler_reg.transform(X_test_r)

# Train Random Forest
print("  Training Random Forest Regressor ...")
rf = RandomForestRegressor(
    n_estimators=200, max_depth=18, min_samples_split=5,
    n_jobs=-1, random_state=42
)
rf.fit(X_train_rs, y_train_r)

# Evaluate
y_pred_r = rf.predict(X_test_rs)
rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
mae  = mean_absolute_error(y_test_r, y_pred_r)
r2   = r2_score(y_test_r, y_pred_r)

print(f"  RMSE : {rmse:,.0f} ms  ({rmse/1000:.2f} sec)")
print(f"  MAE  : {mae:,.0f} ms  ({mae/1000:.2f} sec)")
print(f"  R²   : {r2:.4f}")

all_metrics['regression'] = {
    'rmse_ms': round(rmse, 2),
    'rmse_sec': round(rmse / 1000, 2),
    'mae_ms': round(mae, 2),
    'mae_sec': round(mae / 1000, 2),
    'r2': round(r2, 4),
    'train_size': int(X_train_r.shape[0]),
    'test_size': int(X_test_r.shape[0]),
    'features': features_reg,
}

# Save model
joblib.dump(rf, os.path.join(MODEL_DIR, 'regression_rf.pkl'))
joblib.dump(scaler_reg, os.path.join(MODEL_DIR, 'scaler_regression.pkl'))
joblib.dump(le_circuit, os.path.join(MODEL_DIR, 'le_circuit.pkl'))

# ── Regression Plots ──────────────────────────────────────────

# Plot 1: Actual vs Predicted (scatter)
fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor(COLORS['bg'])
ax.set_facecolor(COLORS['bg'])
sample_idx = np.random.choice(len(y_test_r), size=min(5000, len(y_test_r)), replace=False)
ax.scatter(
    y_test_r.values[sample_idx] / 1000,
    y_pred_r[sample_idx] / 1000,
    alpha=0.3, s=8, color=COLORS['accent'], edgecolors='none'
)
lims = [40, 180]
ax.plot(lims, lims, '--', color=COLORS['primary'], linewidth=2, label='Perfect prediction')
ax.set_xlabel('Actual Lap Time (sec)', fontsize=12, color='white')
ax.set_ylabel('Predicted Lap Time (sec)', fontsize=12, color='white')
ax.set_title('Regression: Actual vs Predicted Lap Times', fontsize=14, fontweight='bold', color='white')
ax.legend(fontsize=11)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.grid(True, alpha=0.2, color=COLORS['grid'])
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'regression_scatter.png'), dpi=150, facecolor=COLORS['bg'])
plt.close()

# Plot 2: Feature Importance
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor(COLORS['bg'])
ax.set_facecolor(COLORS['bg'])
importances = rf.feature_importances_
idx = np.argsort(importances)
ax.barh(
    [features_reg[i] for i in idx], importances[idx],
    color=COLORS['primary'], edgecolor=COLORS['accent'], linewidth=0.5
)
ax.set_xlabel('Importance', fontsize=12, color='white')
ax.set_title('Feature Importance — Lap Time Prediction', fontsize=14, fontweight='bold', color='white')
ax.tick_params(colors='white')
ax.grid(axis='x', alpha=0.2, color=COLORS['grid'])
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'regression_importance.png'), dpi=150, facecolor=COLORS['bg'])
plt.close()

# Plot 3: Residuals distribution
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor(COLORS['bg'])
ax.set_facecolor(COLORS['bg'])
residuals = (y_test_r.values - y_pred_r) / 1000
ax.hist(residuals, bins=100, color=COLORS['accent'], alpha=0.8, edgecolor='none', range=(-20, 20))
ax.axvline(0, color=COLORS['primary'], linewidth=2, linestyle='--')
ax.set_xlabel('Residual (sec)', fontsize=12, color='white')
ax.set_ylabel('Frequency', fontsize=12, color='white')
ax.set_title('Residual Distribution — Lap Time Prediction', fontsize=14, fontweight='bold', color='white')
ax.tick_params(colors='white')
ax.grid(axis='y', alpha=0.2, color=COLORS['grid'])
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'regression_residuals.png'), dpi=150, facecolor=COLORS['bg'])
plt.close()

print("  ✓ Regression model & plots saved!")

# ================================================================
#  2. CLASSIFICATION — Predict Podium Finish (Top 3)
# ================================================================
print("\n" + "=" * 70)
print("  TASK 2: CLASSIFICATION — Predicting Podium Finishes")
print("=" * 70)

# Use the comprehensive results + races data
df_cls = results.merge(races[['raceId', 'year', 'circuitId', 'round']], on='raceId', how='left')
df_cls = df_cls.merge(circuits[['circuitId', 'circuitRef']], on='circuitId', how='left')

# Create target: 1 if finished in top-3 (podium), 0 otherwise
df_cls['position_num'] = pd.to_numeric(df_cls['positionOrder'], errors='coerce')
df_cls = df_cls.dropna(subset=['position_num'])
df_cls['podium'] = (df_cls['position_num'] <= 3).astype(int)

# Encode circuit
le_circuit_cls = LabelEncoder()
df_cls['circuit_enc'] = le_circuit_cls.fit_transform(df_cls['circuitRef'].fillna('unknown'))

# Features
features_cls = ['grid', 'constructorId', 'year', 'round', 'circuit_enc', 'laps']

# Clean data
df_cls['grid'] = pd.to_numeric(df_cls['grid'], errors='coerce').fillna(20)
df_cls['laps'] = pd.to_numeric(df_cls['laps'], errors='coerce').fillna(0)

X_cls = df_cls[features_cls].fillna(0)
y_cls = df_cls['podium']

print(f"  Dataset size: {X_cls.shape[0]:,} rows, {X_cls.shape[1]} features")
print(f"  Podium ratio: {y_cls.mean():.2%}")

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls
)

scaler_cls = StandardScaler()
X_train_cs = scaler_cls.fit_transform(X_train_c)
X_test_cs  = scaler_cls.transform(X_test_c)

# Train Gradient Boosting Classifier
print("  Training Gradient Boosting Classifier ...")
gbc = GradientBoostingClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.1,
    subsample=0.8, random_state=42
)
gbc.fit(X_train_cs, y_train_c)

y_pred_c = gbc.predict(X_test_cs)
y_prob_c = gbc.predict_proba(X_test_cs)[:, 1]

acc  = accuracy_score(y_test_c, y_pred_c)
prec = precision_score(y_test_c, y_pred_c)
rec  = recall_score(y_test_c, y_pred_c)
f1   = f1_score(y_test_c, y_pred_c)

print(f"  Accuracy  : {acc:.4f}")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print(f"  F1 Score  : {f1:.4f}")

all_metrics['classification'] = {
    'accuracy': round(acc, 4),
    'precision': round(prec, 4),
    'recall': round(rec, 4),
    'f1': round(f1, 4),
    'train_size': int(X_train_c.shape[0]),
    'test_size': int(X_test_c.shape[0]),
    'podium_ratio': round(float(y_cls.mean()), 4),
    'features': features_cls,
}

# Save model
joblib.dump(gbc, os.path.join(MODEL_DIR, 'classification_gbc.pkl'))
joblib.dump(scaler_cls, os.path.join(MODEL_DIR, 'scaler_classification.pkl'))
joblib.dump(le_circuit_cls, os.path.join(MODEL_DIR, 'le_circuit_cls.pkl'))

# ── Classification Plots ──────────────────────────────────────

# Plot 1: Confusion Matrix
fig, ax = plt.subplots(figsize=(7, 6))
fig.patch.set_facecolor(COLORS['bg'])
ax.set_facecolor(COLORS['bg'])
cm = confusion_matrix(y_test_c, y_pred_c)
sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', ax=ax,
            xticklabels=['No Podium', 'Podium'],
            yticklabels=['No Podium', 'Podium'],
            linewidths=2, linecolor=COLORS['bg'])
ax.set_xlabel('Predicted', fontsize=12, color='white')
ax.set_ylabel('Actual', fontsize=12, color='white')
ax.set_title('Confusion Matrix — Podium Prediction', fontsize=14, fontweight='bold', color='white')
ax.tick_params(colors='white')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'classification_confusion.png'), dpi=150, facecolor=COLORS['bg'])
plt.close()

# Plot 2: Feature Importance
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor(COLORS['bg'])
ax.set_facecolor(COLORS['bg'])
imp_c = gbc.feature_importances_
idx_c = np.argsort(imp_c)
ax.barh(
    [features_cls[i] for i in idx_c], imp_c[idx_c],
    color=COLORS['gold'], edgecolor=COLORS['primary'], linewidth=0.5
)
ax.set_xlabel('Importance', fontsize=12, color='white')
ax.set_title('Feature Importance — Podium Prediction', fontsize=14, fontweight='bold', color='white')
ax.tick_params(colors='white')
ax.grid(axis='x', alpha=0.2, color=COLORS['grid'])
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'classification_importance.png'), dpi=150, facecolor=COLORS['bg'])
plt.close()

# Plot 3: Metrics bar chart
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor(COLORS['bg'])
ax.set_facecolor(COLORS['bg'])
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
metric_vals  = [acc, prec, rec, f1]
bars = ax.bar(metric_names, metric_vals, color=[COLORS['primary'], COLORS['accent'], COLORS['gold'], COLORS['silver']], edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, metric_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}',
            ha='center', va='bottom', fontsize=13, fontweight='bold', color='white')
ax.set_ylim(0, 1.1)
ax.set_title('Classification Metrics — Podium Prediction', fontsize=14, fontweight='bold', color='white')
ax.tick_params(colors='white')
ax.grid(axis='y', alpha=0.2, color=COLORS['grid'])
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'classification_metrics.png'), dpi=150, facecolor=COLORS['bg'])
plt.close()

print("  ✓ Classification model & plots saved!")

# ================================================================
#  3. CLUSTERING — F1 Driver Profiles
# ================================================================
print("\n" + "=" * 70)
print("  TASK 3: CLUSTERING — F1 Driver Profiling")
print("=" * 70)

df_clust = pd.read_csv(os.path.join(DATA, 'f1_clustering', 'F1DriversDataset.csv'))
print(f"  Raw dataset: {df_clust.shape[0]} drivers, {df_clust.shape[1]} columns")

# Select numeric features for clustering
num_cols = df_clust.select_dtypes(include=[np.number]).columns.tolist()
# Remove boolean-like columns that aren't useful
drop_cols = [c for c in num_cols if df_clust[c].nunique() <= 2]
cluster_features = [c for c in num_cols if c not in drop_cols]

print(f"  Clustering features: {cluster_features}")

X_clust = df_clust[cluster_features].fillna(0)

scaler_clust = StandardScaler()
X_clust_scaled = scaler_clust.fit_transform(X_clust)

# Find optimal K using silhouette score
k_range = range(2, 10)
sil_scores = []
inertias = []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_clust_scaled)
    sil_scores.append(silhouette_score(X_clust_scaled, labels))
    inertias.append(km.inertia_)

best_k = list(k_range)[np.argmax(sil_scores)]
print(f"  Best K = {best_k} (silhouette = {max(sil_scores):.4f})")

# Final model
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_labels = km_final.fit_predict(X_clust_scaled)
df_clust['cluster'] = cluster_labels

sil = silhouette_score(X_clust_scaled, cluster_labels)

# PCA for visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_clust_scaled)

all_metrics['clustering'] = {
    'n_clusters': int(best_k),
    'silhouette_score': round(sil, 4),
    'n_drivers': int(X_clust.shape[0]),
    'features': cluster_features,
    'pca_variance': [round(v, 4) for v in pca.explained_variance_ratio_.tolist()],
    'cluster_sizes': {str(i): int((cluster_labels == i).sum()) for i in range(best_k)},
}

# Cluster profile — compute mean for each cluster
cluster_profile = df_clust.groupby('cluster')[cluster_features].mean().round(2).to_dict(orient='index')
all_metrics['clustering']['cluster_profiles'] = cluster_profile

# Save model
joblib.dump(km_final, os.path.join(MODEL_DIR, 'clustering_kmeans.pkl'))
joblib.dump(scaler_clust, os.path.join(MODEL_DIR, 'scaler_clustering.pkl'))
joblib.dump(pca, os.path.join(MODEL_DIR, 'pca_clustering.pkl'))

# Save driver names with their clusters
driver_clusters = df_clust[['Driver', 'cluster']].to_dict(orient='records')
with open(os.path.join(MODEL_DIR, 'driver_clusters.json'), 'w') as f:
    json.dump(driver_clusters, f, indent=2)

# ── Clustering Plots ──────────────────────────────────────────

cluster_colors = ['#E10600', '#00D2BE', '#FFD700', '#FF8700', '#0090FF',
                  '#9B59B6', '#1ABC9C', '#E74C3C', '#2ECC71']

# Plot 1: PCA Scatter
fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor(COLORS['bg'])
ax.set_facecolor(COLORS['bg'])
for i in range(best_k):
    mask = cluster_labels == i
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], s=40, alpha=0.7,
               color=cluster_colors[i % len(cluster_colors)],
               label=f'Cluster {i} ({mask.sum()} drivers)',
               edgecolors='white', linewidth=0.3)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12, color='white')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12, color='white')
ax.set_title('F1 Driver Clusters (PCA Visualization)', fontsize=14, fontweight='bold', color='white')
ax.legend(fontsize=10, loc='upper right')
ax.tick_params(colors='white')
ax.grid(True, alpha=0.15, color=COLORS['grid'])
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'clustering_pca.png'), dpi=150, facecolor=COLORS['bg'])
plt.close()

# Plot 2: Elbow + Silhouette
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(COLORS['bg'])
for ax in [ax1, ax2]:
    ax.set_facecolor(COLORS['bg'])

ax1.plot(list(k_range), inertias, 'o-', color=COLORS['primary'], linewidth=2, markersize=8)
ax1.axvline(best_k, linestyle='--', color=COLORS['accent'], linewidth=1.5, label=f'Best K={best_k}')
ax1.set_xlabel('Number of Clusters (K)', fontsize=12, color='white')
ax1.set_ylabel('Inertia', fontsize=12, color='white')
ax1.set_title('Elbow Method', fontsize=14, fontweight='bold', color='white')
ax1.legend(fontsize=11)
ax1.tick_params(colors='white')
ax1.grid(True, alpha=0.2, color=COLORS['grid'])

ax2.plot(list(k_range), sil_scores, 's-', color=COLORS['accent'], linewidth=2, markersize=8)
ax2.axvline(best_k, linestyle='--', color=COLORS['primary'], linewidth=1.5, label=f'Best K={best_k}')
ax2.set_xlabel('Number of Clusters (K)', fontsize=12, color='white')
ax2.set_ylabel('Silhouette Score', fontsize=12, color='white')
ax2.set_title('Silhouette Analysis', fontsize=14, fontweight='bold', color='white')
ax2.legend(fontsize=11)
ax2.tick_params(colors='white')
ax2.grid(True, alpha=0.2, color=COLORS['grid'])

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'clustering_elbow.png'), dpi=150, facecolor=COLORS['bg'])
plt.close()

# Plot 3: Cluster radar / bar comparison
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor(COLORS['bg'])
ax.set_facecolor(COLORS['bg'])

# Normalize cluster means for comparison
cluster_means = df_clust.groupby('cluster')[cluster_features].mean()
cluster_means_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min() + 1e-9)

x_pos = np.arange(len(cluster_features))
width = 0.8 / best_k
for i in range(best_k):
    ax.bar(x_pos + i * width, cluster_means_norm.loc[i].values, width,
           color=cluster_colors[i % len(cluster_colors)], alpha=0.85,
           label=f'Cluster {i}', edgecolor='white', linewidth=0.3)

ax.set_xticks(x_pos + width * (best_k - 1) / 2)
ax.set_xticklabels(cluster_features, rotation=45, ha='right', fontsize=9, color='white')
ax.set_ylabel('Normalized Value', fontsize=12, color='white')
ax.set_title('Cluster Feature Comparison', fontsize=14, fontweight='bold', color='white')
ax.legend(fontsize=10)
ax.tick_params(colors='white')
ax.grid(axis='y', alpha=0.2, color=COLORS['grid'])
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'clustering_comparison.png'), dpi=150, facecolor=COLORS['bg'])
plt.close()

print("  ✓ Clustering model & plots saved!")

# ════════════════════════════════════════════════════════════════
# Save all metrics
# ════════════════════════════════════════════════════════════════
metrics_path = os.path.join(BASE, 'static', 'metrics.json')
with open(metrics_path, 'w') as f:
    json.dump(all_metrics, f, indent=2)

print("\n" + "=" * 70)
print("  ALL MODELS TRAINED SUCCESSFULLY!")
print("=" * 70)
print(f"  Models saved to: {MODEL_DIR}")
print(f"  Plots saved to:  {PLOT_DIR}")
print(f"  Metrics saved to: {metrics_path}")
print("=" * 70)
