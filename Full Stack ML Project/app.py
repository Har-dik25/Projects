"""
Dynamic ML Platform — Flask Backend
======================================
• Lists all CSV files from datasets/ folder
• Performs exploratory data analysis on any selected dataset
• Auto-cleans data (missing values, encoding, scaling)
• Trains Regression / Classification / Clustering models
• Returns all parameters, metrics, plots, and plain-language explanations
"""

import os, json, io, base64, traceback, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    RandomForestClassifier, GradientBoostingClassifier
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, silhouette_score
)
from sklearn.decomposition import PCA
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE, 'datasets')

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# ── plot styling ───────────────────────────────────────────────
plt.style.use('dark_background')
COLORS = {
    'primary': '#E10600', 'accent': '#00D2BE', 'gold': '#FFD700',
    'orange': '#FF8700', 'blue': '#0090FF', 'purple': '#9B59B6',
    'bg': '#1A1A2E', 'grid': '#333355',
}
PALETTE = ['#E10600', '#00D2BE', '#FFD700', '#FF8700', '#0090FF',
           '#9B59B6', '#1ABC9C', '#E74C3C', '#2ECC71', '#F39C12']


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight', facecolor=COLORS['bg'])
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{b64}"


# ═══════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')


# ── 1. List all datasets ──────────────────────────────────────
@app.route('/api/datasets')
def list_datasets():
    """Recursively find all CSV files in datasets/ folder."""
    datasets = []
    for root, dirs, files in os.walk(DATASETS_DIR):
        for f in files:
            if f.lower().endswith('.csv'):
                rel = os.path.relpath(os.path.join(root, f), DATASETS_DIR)
                size = os.path.getsize(os.path.join(root, f))
                datasets.append({
                    'name': f,
                    'path': rel.replace('\\', '/'),
                    'size_kb': round(size / 1024, 1),
                    'folder': os.path.relpath(root, DATASETS_DIR).replace('\\', '/'),
                })
    datasets.sort(key=lambda x: x['path'])
    return jsonify(datasets)


# ── 2. Analyze a dataset (EDA) ────────────────────────────────
@app.route('/api/analyze', methods=['POST'])
def analyze_dataset():
    """Load a CSV and return comprehensive EDA."""
    data = request.json
    path = os.path.join(DATASETS_DIR, data['path'])

    if not os.path.exists(path):
        return jsonify({'error': f'File not found: {path}'}), 404

    try:
        df = pd.read_csv(path)
    except Exception as e:
        return jsonify({'error': f'Failed to read CSV: {str(e)}'}), 400

    # Basic info
    info = {
        'filename': os.path.basename(path),
        'rows': int(df.shape[0]),
        'columns': int(df.shape[1]),
        'memory_kb': round(df.memory_usage(deep=True).sum() / 1024, 1),
    }

    # Column details
    columns = []
    for col in df.columns:
        col_info = {
            'name': col,
            'dtype': str(df[col].dtype),
            'missing': int(df[col].isnull().sum()),
            'missing_pct': round(df[col].isnull().mean() * 100, 1),
            'unique': int(df[col].nunique()),
        }
        if df[col].dtype in ['int64', 'float64']:
            col_info['is_numeric'] = True
            col_info['mean'] = round(float(df[col].mean()), 2) if not df[col].isnull().all() else None
            col_info['std'] = round(float(df[col].std()), 2) if not df[col].isnull().all() else None
            col_info['min'] = round(float(df[col].min()), 2) if not df[col].isnull().all() else None
            col_info['max'] = round(float(df[col].max()), 2) if not df[col].isnull().all() else None
        else:
            col_info['is_numeric'] = False
            top = df[col].value_counts().head(3).to_dict()
            col_info['top_values'] = {str(k): int(v) for k, v in top.items()}
        columns.append(col_info)

    # Sample rows
    sample = df.head(5).fillna('NaN').astype(str).values.tolist()
    sample_cols = df.columns.tolist()

    # Statistical summary for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Generate EDA plots
    plots = []

    # Plot 1: Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor(COLORS['bg'])
        ax.set_facecolor(COLORS['bg'])
        missing_data = missing[missing > 0].sort_values(ascending=True)
        ax.barh(missing_data.index, missing_data.values, color=COLORS['primary'])
        ax.set_xlabel('Missing Count', color='white')
        ax.set_title('Missing Values by Column', fontweight='bold', color='white', fontsize=13)
        ax.tick_params(colors='white')
        ax.grid(axis='x', alpha=0.2, color=COLORS['grid'])
        plots.append({'title': 'Missing Values', 'image': fig_to_base64(fig)})

    # Plot 2: Data type distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])
    dtype_counts = df.dtypes.astype(str).value_counts()
    ax.pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.0f%%',
           colors=PALETTE[:len(dtype_counts)], textprops={'color': 'white', 'fontsize': 11})
    ax.set_title('Column Data Types', fontweight='bold', color='white', fontsize=13)
    plots.append({'title': 'Data Types', 'image': fig_to_base64(fig)})

    # Plot 3: Correlation heatmap (if enough numeric columns)
    if len(numeric_cols) >= 2:
        corr_cols = numeric_cols[:12]  # limit to 12 for readability
        fig, ax = plt.subplots(figsize=(max(8, len(corr_cols)), max(6, len(corr_cols) * 0.7)))
        fig.patch.set_facecolor(COLORS['bg'])
        ax.set_facecolor(COLORS['bg'])
        corr = df[corr_cols].corr()
        sns.heatmap(corr, annot=len(corr_cols) <= 8, fmt='.2f', cmap='RdYlGn',
                    ax=ax, linewidths=0.5, linecolor=COLORS['bg'],
                    cbar_kws={'shrink': 0.8})
        ax.set_title('Correlation Heatmap', fontweight='bold', color='white', fontsize=13)
        ax.tick_params(colors='white', labelsize=9)
        plots.append({'title': 'Correlation Heatmap', 'image': fig_to_base64(fig)})

    # Plot 4: Distribution of first few numeric columns
    if len(numeric_cols) >= 1:
        show_cols = numeric_cols[:4]
        fig, axes = plt.subplots(1, len(show_cols), figsize=(4 * len(show_cols), 4))
        fig.patch.set_facecolor(COLORS['bg'])
        if len(show_cols) == 1:
            axes = [axes]
        for i, col in enumerate(show_cols):
            axes[i].set_facecolor(COLORS['bg'])
            data = df[col].dropna()
            axes[i].hist(data, bins=30, color=PALETTE[i % len(PALETTE)], alpha=0.85, edgecolor='none')
            axes[i].set_title(col, color='white', fontsize=11, fontweight='bold')
            axes[i].tick_params(colors='white', labelsize=8)
            axes[i].grid(axis='y', alpha=0.2, color=COLORS['grid'])
        fig.suptitle('Numeric Distributions', color='white', fontweight='bold', fontsize=13, y=1.02)
        plt.tight_layout()
        plots.append({'title': 'Distributions', 'image': fig_to_base64(fig)})

    return jsonify({
        'info': info,
        'columns': columns,
        'sample': sample,
        'sample_cols': sample_cols,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'plots': plots,
    })


# ── 3. Train a model ──────────────────────────────────────────
@app.route('/api/train', methods=['POST'])
def train_model():
    """
    Auto-clean data and train the selected model.
    Returns metrics, parameters, plots, and plain-language explanations.
    """
    data = request.json
    path = os.path.join(DATASETS_DIR, data['path'])
    task = data['task']  # regression, classification, clustering
    target_col = data.get('target', None)
    feature_cols = data.get('features', [])

    try:
        df = pd.read_csv(path)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    result = {
        'task': task,
        'steps': [],
        'metrics': {},
        'parameters': {},
        'plots': [],
        'explanations': [],
    }

    try:
        return _do_train(df, task, target_col, feature_cols, result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Training failed: {str(e)}'}), 500


def _do_train(df, task, target_col, feature_cols, result):

    # ── STEP 1: Data Cleaning ──────────────────────────────────
    original_shape = df.shape
    cleaning_log = []

    # Drop columns that are >60% missing
    thresh = 0.6
    for col in df.columns:
        if df[col].isnull().mean() > thresh:
            df.drop(col, axis=1, inplace=True)
            cleaning_log.append(f"Dropped column '{col}' (>{thresh*100:.0f}% missing)")

    # Fill numeric missing values with median
    numeric_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols_all:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            count = int(df[col].isnull().sum())
            df[col].fillna(median_val, inplace=True)
            cleaning_log.append(f"Filled {count} missing values in '{col}' with median ({median_val:.2f})")

    # Fill categorical missing values with mode
    cat_cols_all = df.select_dtypes(exclude=[np.number]).columns.tolist()
    for col in cat_cols_all:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
            count = int(df[col].isnull().sum())
            df[col].fillna(mode_val, inplace=True)
            cleaning_log.append(f"Filled {count} missing values in '{col}' with mode ('{mode_val}')")

    # Drop duplicate rows
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        df.drop_duplicates(inplace=True)
        cleaning_log.append(f"Removed {dup_count} duplicate rows")

    if not cleaning_log:
        cleaning_log.append("No cleaning needed — dataset was already clean!")

    result['steps'].append({
        'title': '🧹 Step 1: Data Cleaning',
        'details': cleaning_log,
        'explanation': f"I started by cleaning the dataset. The original data had {original_shape[0]} rows and {original_shape[1]} columns. "
                       f"After cleaning, we have {df.shape[0]} rows and {df.shape[1]} columns. "
                       f"I removed columns with too many missing values, filled numeric gaps with the median (middle value), "
                       f"and filled text gaps with the most common value. I also removed any duplicate rows."
    })

    # ── STEP 2: Feature Engineering / Encoding ─────────────────
    encoding_log = []
    label_encoders = {}

    if task in ['regression', 'classification']:
        if not target_col or target_col not in df.columns:
            return jsonify({'error': f"Target column '{target_col}' not found in dataset"}), 400

        # If no features selected, use all columns except target
        if not feature_cols:
            feature_cols = [c for c in df.columns if c != target_col]

        # Encode categorical features
        for col in feature_cols[:]:
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                if df[col].nunique() <= 50:
                    le = LabelEncoder()
                    df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
                    label_encoders[col] = le
                    feature_cols.remove(col)
                    feature_cols.append(col + '_encoded')
                    encoding_log.append(f"Encoded '{col}' → '{col}_encoded' ({df[col].nunique()} unique values → numbers)")
                else:
                    feature_cols.remove(col)
                    encoding_log.append(f"Dropped '{col}' (too many unique values: {df[col].nunique()})")

        # For classification: encode target if it's categorical
        if task == 'classification' and (df[target_col].dtype == 'object' or df[target_col].dtype.name == 'bool'):
            le_target = LabelEncoder()
            df[target_col] = le_target.fit_transform(df[target_col].astype(str))
            encoding_log.append(f"Encoded target '{target_col}' into numbers: {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")

        # Keep only numeric features
        valid_features = [c for c in feature_cols if c in df.columns and df[c].dtype in ['int64', 'float64', 'int32', 'float32']]
        if not valid_features:
            return jsonify({'error': 'No valid numeric features available after encoding'}), 400

        feature_cols = valid_features

    elif task == 'clustering':
        # Use all numeric columns for clustering
        if feature_cols:
            feature_cols = [c for c in feature_cols if c in df.columns]
        if not feature_cols:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not feature_cols:
            return jsonify({'error': 'No numeric columns for clustering'}), 400

    if not encoding_log:
        encoding_log.append("All selected features are already numeric — no encoding needed!")

    result['steps'].append({
        'title': '🔧 Step 2: Feature Engineering',
        'details': encoding_log,
        'explanation': f"I prepared the features for the model. Machine learning models need numbers, not text. "
                       f"So I converted text columns into numbers using Label Encoding (assigning each unique text a number). "
                       f"Final features used: {', '.join(feature_cols)}"
    })

    # ── STEP 3: Train Model ────────────────────────────────────
    plots = []

    if task == 'regression':
        X = df[feature_cols].values
        y = df[target_col].values.astype(float)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = RandomForestRegressor(n_estimators=150, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1)
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        result['metrics'] = {
            'R² Score': {'value': round(r2, 4), 'explanation': f"The model explains {r2*100:.1f}% of the variation in '{target_col}'. A perfect score is 1.0. "
                         f"{'This is a strong result!' if r2 > 0.7 else 'This is moderate.' if r2 > 0.4 else 'This is weak — the features may not be very predictive.'}"},
            'RMSE': {'value': round(rmse, 4), 'explanation': f"On average, the predictions are off by about {rmse:.2f} units. "
                     f"Compared to the range of '{target_col}' ({float(y.min()):.1f} to {float(y.max()):.1f}), "
                     f"{'this error is small.' if rmse < (y.max() - y.min()) * 0.1 else 'this error is moderate.' if rmse < (y.max() - y.min()) * 0.25 else 'this error is significant.'}"},
            'MAE': {'value': round(mae, 4), 'explanation': f"The average absolute error is {mae:.2f}. This means on average each prediction is {mae:.2f} units away from the real value."},
            'Train Size': {'value': int(X_train.shape[0]), 'explanation': f"We used {X_train.shape[0]:,} rows (80%) to teach the model."},
            'Test Size': {'value': int(X_test.shape[0]), 'explanation': f"We tested the model on {X_test.shape[0]:,} rows (20%) it had never seen before."},
        }

        result['parameters'] = {
            'Algorithm': 'Random Forest Regressor',
            'Number of Trees': 150,
            'Max Depth': 15,
            'Min Samples Split': 5,
            'Test Split': '20%',
            'Scaling': 'StandardScaler (zero mean, unit variance)',
            'Features Used': feature_cols,
            'Target Column': target_col,
        }

        # Plot 1: Actual vs Predicted
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor(COLORS['bg']); ax.set_facecolor(COLORS['bg'])
        sample_n = min(3000, len(y_test))
        idx = np.random.choice(len(y_test), sample_n, replace=False)
        ax.scatter(y_test[idx], y_pred[idx], alpha=0.35, s=10, color=COLORS['accent'], edgecolors='none')
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, '--', color=COLORS['primary'], linewidth=2, label='Perfect Prediction')
        ax.set_xlabel('Actual Values', color='white', fontsize=12)
        ax.set_ylabel('Predicted Values', color='white', fontsize=12)
        ax.set_title(f'Actual vs Predicted — {target_col}', fontweight='bold', color='white', fontsize=14)
        ax.legend(fontsize=11); ax.tick_params(colors='white')
        ax.grid(True, alpha=0.15, color=COLORS['grid'])
        plots.append({'title': 'Actual vs Predicted', 'image': fig_to_base64(fig),
                       'explanation': 'Each dot is one data point. If the model were perfect, all dots would sit on the red dashed line. Points close to the line = good predictions.'})

        # Plot 2: Feature Importance
        fig, ax = plt.subplots(figsize=(8, max(4, len(feature_cols) * 0.5)))
        fig.patch.set_facecolor(COLORS['bg']); ax.set_facecolor(COLORS['bg'])
        imp = model.feature_importances_
        sorted_idx = np.argsort(imp)
        ax.barh([feature_cols[i] for i in sorted_idx], imp[sorted_idx],
                color=COLORS['primary'], edgecolor=COLORS['accent'], linewidth=0.5)
        ax.set_xlabel('Importance', color='white', fontsize=12)
        ax.set_title('Feature Importance', fontweight='bold', color='white', fontsize=14)
        ax.tick_params(colors='white'); ax.grid(axis='x', alpha=0.2, color=COLORS['grid'])
        plots.append({'title': 'Feature Importance', 'image': fig_to_base64(fig),
                       'explanation': 'This shows which features the model relies on the most. Longer bars = more important for making predictions.'})

        # Plot 3: Residuals
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor(COLORS['bg']); ax.set_facecolor(COLORS['bg'])
        residuals = y_test - y_pred
        ax.hist(residuals, bins=60, color=COLORS['accent'], alpha=0.85, edgecolor='none')
        ax.axvline(0, color=COLORS['primary'], linewidth=2, linestyle='--')
        ax.set_xlabel('Error (Actual - Predicted)', color='white', fontsize=12)
        ax.set_ylabel('Count', color='white', fontsize=12)
        ax.set_title('Prediction Errors Distribution', fontweight='bold', color='white', fontsize=14)
        ax.tick_params(colors='white'); ax.grid(axis='y', alpha=0.2, color=COLORS['grid'])
        plots.append({'title': 'Residual Distribution', 'image': fig_to_base64(fig),
                       'explanation': 'This shows how the errors are distributed. A bell-shape centered at 0 means the model is unbiased — it does not consistently over-predict or under-predict.'})

        result['steps'].append({
            'title': '🤖 Step 3: Model Training — Regression',
            'details': [
                f"Algorithm: Random Forest Regressor with 150 decision trees",
                f"Split data: 80% training ({X_train.shape[0]:,} rows), 20% testing ({X_test.shape[0]:,} rows)",
                f"Applied StandardScaler to normalize all features",
                f"Trained model on {len(feature_cols)} features to predict '{target_col}'",
            ],
            'explanation': f"I used a Random Forest Regressor — it's like asking 150 different 'decision trees' to each make a prediction, "
                           f"then averaging their answers. This makes it very accurate and resistant to overfitting. "
                           f"I first scaled all the features so they're on the same scale (zero mean, unit spread). "
                           f"The model achieved an R² of {r2:.4f}, meaning it explains {r2*100:.1f}% of what drives '{target_col}'."
        })

    elif task == 'classification':
        X = df[feature_cols].values
        y = df[target_col].values

        # Make sure y is valid
        unique_classes = np.unique(y[~pd.isnull(y)])
        n_classes = len(unique_classes)

        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, subsample=0.8, random_state=42)
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

        acc = accuracy_score(y_test, y_pred)
        avg = 'binary' if n_classes == 2 else 'weighted'
        prec = precision_score(y_test, y_pred, average=avg, zero_division=0)
        rec = recall_score(y_test, y_pred, average=avg, zero_division=0)
        f1 = f1_score(y_test, y_pred, average=avg, zero_division=0)

        result['metrics'] = {
            'Accuracy': {'value': round(acc, 4), 'explanation': f"The model correctly classified {acc*100:.1f}% of all test samples. "
                         f"{'Excellent!' if acc > 0.9 else 'Good performance.' if acc > 0.75 else 'Room for improvement.'}"},
            'Precision': {'value': round(prec, 4), 'explanation': f"When the model says 'yes', it's right {prec*100:.1f}% of the time. "
                          f"High precision = few false alarms."},
            'Recall': {'value': round(rec, 4), 'explanation': f"The model catches {rec*100:.1f}% of actual positive cases. "
                       f"High recall = rarely misses a real positive."},
            'F1 Score': {'value': round(f1, 4), 'explanation': f"F1 = {f1:.4f}. This balances Precision and Recall. "
                         f"{'Strong balance!' if f1 > 0.8 else 'Decent balance.' if f1 > 0.6 else 'Precision and recall are imbalanced.'}"},
            'Classes': {'value': int(n_classes), 'explanation': f"The model distinguishes between {n_classes} different classes/categories in '{target_col}'."},
            'Train Size': {'value': int(X_train.shape[0]), 'explanation': f"{X_train.shape[0]:,} samples used for training (80%)."},
            'Test Size': {'value': int(X_test.shape[0]), 'explanation': f"{X_test.shape[0]:,} samples used for evaluation (20%, never seen during training)."},
        }

        result['parameters'] = {
            'Algorithm': 'Gradient Boosting Classifier',
            'Number of Trees': 150,
            'Max Depth': 5,
            'Learning Rate': 0.1,
            'Subsample': '80%',
            'Test Split': '20%',
            'Scaling': 'StandardScaler',
            'Features Used': feature_cols,
            'Target Column': target_col,
            'Number of Classes': int(n_classes),
        }

        # Plot 1: Confusion Matrix
        fig, ax = plt.subplots(figsize=(max(6, n_classes), max(5, n_classes * 0.8)))
        fig.patch.set_facecolor(COLORS['bg']); ax.set_facecolor(COLORS['bg'])
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', ax=ax, linewidths=2, linecolor=COLORS['bg'])
        ax.set_xlabel('Predicted', color='white', fontsize=12)
        ax.set_ylabel('Actual', color='white', fontsize=12)
        ax.set_title('Confusion Matrix', fontweight='bold', color='white', fontsize=14)
        ax.tick_params(colors='white')
        plots.append({'title': 'Confusion Matrix', 'image': fig_to_base64(fig),
                       'explanation': 'The diagonal (top-left to bottom-right) shows correct predictions. Off-diagonal cells show mistakes. Darker green = more correct, red = more errors.'})

        # Plot 2: Metrics bar
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor(COLORS['bg']); ax.set_facecolor(COLORS['bg'])
        names = ['Accuracy', 'Precision', 'Recall', 'F1']
        vals = [acc, prec, rec, f1]
        bars = ax.bar(names, vals, color=[COLORS['primary'], COLORS['accent'], COLORS['gold'], COLORS['blue']],
                      edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                    f'{val:.3f}', ha='center', fontsize=13, fontweight='bold', color='white')
        ax.set_ylim(0, 1.15); ax.set_title('Classification Metrics', fontweight='bold', color='white', fontsize=14)
        ax.tick_params(colors='white'); ax.grid(axis='y', alpha=0.2, color=COLORS['grid'])
        plots.append({'title': 'Classification Metrics', 'image': fig_to_base64(fig),
                       'explanation': 'All four metrics range from 0 to 1 (higher = better). A good model has all bars high and roughly equal.'})

        # Plot 3: Feature Importance
        fig, ax = plt.subplots(figsize=(8, max(4, len(feature_cols) * 0.5)))
        fig.patch.set_facecolor(COLORS['bg']); ax.set_facecolor(COLORS['bg'])
        imp = model.feature_importances_
        sorted_idx = np.argsort(imp)
        ax.barh([feature_cols[i] for i in sorted_idx], imp[sorted_idx],
                color=COLORS['gold'], edgecolor=COLORS['primary'], linewidth=0.5)
        ax.set_xlabel('Importance', color='white', fontsize=12)
        ax.set_title('Feature Importance', fontweight='bold', color='white', fontsize=14)
        ax.tick_params(colors='white'); ax.grid(axis='x', alpha=0.2, color=COLORS['grid'])
        plots.append({'title': 'Feature Importance', 'image': fig_to_base64(fig),
                       'explanation': 'Shows which features the model pays most attention to. The top feature has the most influence on the prediction.'})

        result['steps'].append({
            'title': '🤖 Step 3: Model Training — Classification',
            'details': [
                f"Algorithm: Gradient Boosting Classifier with 150 boosted trees",
                f"Split data: 80% training, 20% testing (stratified to keep class balance)",
                f"Applied StandardScaler to normalize features",
                f"Predicting '{target_col}' with {n_classes} classes using {len(feature_cols)} features",
            ],
            'explanation': f"I used Gradient Boosting — it builds trees one after another, where each new tree tries to fix the mistakes "
                           f"of the previous ones. It's one of the best algorithms for classification. "
                           f"I used stratified splitting to make sure each class is fairly represented in both training and test sets. "
                           f"The model achieved {acc*100:.1f}% accuracy with an F1 score of {f1:.4f}."
        })

    elif task == 'clustering':
        X = df[feature_cols].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Find optimal K
        k_range = range(2, min(10, len(df)))
        sil_scores = []
        inertias = []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            sil_scores.append(silhouette_score(X_scaled, labels))
            inertias.append(km.inertia_)

        best_k = list(k_range)[np.argmax(sil_scores)]
        best_sil = max(sil_scores)

        km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        cluster_labels = km_final.fit_predict(X_scaled)
        df['Cluster'] = cluster_labels

        cluster_sizes = {f'Cluster {i}': int((cluster_labels == i).sum()) for i in range(best_k)}

        result['metrics'] = {
            'Optimal K': {'value': best_k, 'explanation': f"The algorithm found that {best_k} groups best separates this data. "
                          f"I tested K from 2 to {max(k_range)} and picked the one with the best Silhouette Score."},
            'Silhouette Score': {'value': round(best_sil, 4), 'explanation': f"Score = {best_sil:.4f} (range: -1 to 1). "
                                 f"{'Excellent separation!' if best_sil > 0.7 else 'Good separation.' if best_sil > 0.5 else 'Moderate separation.' if best_sil > 0.25 else 'Weak separation — clusters overlap.'} "
                                 f"Higher score means the groups are more distinct from each other."},
            'Total Samples': {'value': int(len(df)), 'explanation': f"All {len(df):,} data points were assigned to a cluster."},
            'Features Used': {'value': len(feature_cols), 'explanation': f"Used {len(feature_cols)} numeric features for grouping."},
        }

        for cname, csize in cluster_sizes.items():
            result['metrics'][cname] = {'value': csize, 'explanation': f"This group contains {csize} data points ({csize/len(df)*100:.1f}% of the data)."}

        result['parameters'] = {
            'Algorithm': 'KMeans Clustering',
            'Optimal K': best_k,
            'Initialization': 'k-means++ (10 runs)',
            'Scaling': 'StandardScaler',
            'Features Used': feature_cols,
            'Selection Method': 'Silhouette Score optimization',
        }

        # PCA for visualization
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)

        # Plot 1: PCA Scatter
        fig, ax = plt.subplots(figsize=(10, 7))
        fig.patch.set_facecolor(COLORS['bg']); ax.set_facecolor(COLORS['bg'])
        for i in range(best_k):
            mask = cluster_labels == i
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], s=30, alpha=0.7,
                       color=PALETTE[i % len(PALETTE)], label=f'Cluster {i} ({mask.sum()})',
                       edgecolors='white', linewidth=0.3)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', color='white', fontsize=12)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', color='white', fontsize=12)
        ax.set_title('Cluster Visualization (PCA)', fontweight='bold', color='white', fontsize=14)
        ax.legend(fontsize=10); ax.tick_params(colors='white'); ax.grid(True, alpha=0.15, color=COLORS['grid'])
        plots.append({'title': 'Cluster Visualization', 'image': fig_to_base64(fig),
                       'explanation': 'This 2D plot compresses all features into two axes using PCA. Each color is a different cluster. Well-separated colors = distinct groups.'})

        # Plot 2: Elbow + Silhouette
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor(COLORS['bg'])
        for ax in [ax1, ax2]: ax.set_facecolor(COLORS['bg'])
        ax1.plot(list(k_range), inertias, 'o-', color=COLORS['primary'], linewidth=2, markersize=8)
        ax1.axvline(best_k, linestyle='--', color=COLORS['accent'], linewidth=1.5, label=f'Best K={best_k}')
        ax1.set_xlabel('K', color='white'); ax1.set_ylabel('Inertia', color='white')
        ax1.set_title('Elbow Method', fontweight='bold', color='white', fontsize=13)
        ax1.legend(fontsize=11); ax1.tick_params(colors='white'); ax1.grid(True, alpha=0.2, color=COLORS['grid'])

        ax2.plot(list(k_range), sil_scores, 's-', color=COLORS['accent'], linewidth=2, markersize=8)
        ax2.axvline(best_k, linestyle='--', color=COLORS['primary'], linewidth=1.5, label=f'Best K={best_k}')
        ax2.set_xlabel('K', color='white'); ax2.set_ylabel('Silhouette Score', color='white')
        ax2.set_title('Silhouette Analysis', fontweight='bold', color='white', fontsize=13)
        ax2.legend(fontsize=11); ax2.tick_params(colors='white'); ax2.grid(True, alpha=0.2, color=COLORS['grid'])
        plots.append({'title': 'Elbow & Silhouette', 'image': fig_to_base64(fig),
                       'explanation': 'Left: The "elbow" shows where adding more clusters stops being helpful. Right: Silhouette Score — higher is better. The dotted line marks the best K.'})

        # Plot 3: Cluster comparison
        cluster_means = df.groupby('Cluster')[feature_cols].mean()
        cluster_means_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min() + 1e-9)
        show_feats = feature_cols[:10]

        fig, ax = plt.subplots(figsize=(max(10, len(show_feats)), 6))
        fig.patch.set_facecolor(COLORS['bg']); ax.set_facecolor(COLORS['bg'])
        x_pos = np.arange(len(show_feats))
        width = 0.8 / best_k
        for i in range(best_k):
            vals = [cluster_means_norm.loc[i, f] if f in cluster_means_norm.columns else 0 for f in show_feats]
            ax.bar(x_pos + i * width, vals, width, color=PALETTE[i % len(PALETTE)], alpha=0.85,
                   label=f'Cluster {i}', edgecolor='white', linewidth=0.3)
        ax.set_xticks(x_pos + width * (best_k - 1) / 2)
        ax.set_xticklabels(show_feats, rotation=45, ha='right', fontsize=9, color='white')
        ax.set_ylabel('Normalized Value', color='white'); ax.set_title('Cluster Profiles', fontweight='bold', color='white', fontsize=14)
        ax.legend(fontsize=10); ax.tick_params(colors='white'); ax.grid(axis='y', alpha=0.2, color=COLORS['grid'])
        plots.append({'title': 'Cluster Profiles', 'image': fig_to_base64(fig),
                       'explanation': 'Each bar group shows how each cluster scores on different features. This reveals what makes each group unique.'})

        result['steps'].append({
            'title': '🤖 Step 3: Model Training — Clustering',
            'details': [
                f"Algorithm: KMeans with automatic K selection (tested K=2 to K={max(k_range)})",
                f"Found optimal K={best_k} using Silhouette Score ({best_sil:.4f})",
                f"Applied StandardScaler before clustering",
                f"Used PCA for 2D visualization",
            ],
            'explanation': f"I used KMeans Clustering to find natural groups in the data. Instead of predicting a specific value, "
                           f"clustering finds patterns — it groups similar data points together. I tested different numbers of groups (2 to {max(k_range)}) "
                           f"and found that {best_k} groups give the best separation. "
                           f"The Silhouette Score of {best_sil:.4f} tells us how well-defined the groups are."
        })

    result['plots'] = plots
    return jsonify(result)


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  🧠 Dynamic ML Platform — http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)
