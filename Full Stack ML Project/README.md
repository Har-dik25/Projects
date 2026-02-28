# 🏎️ Full Stack ML Project — Dynamic ML Platform

A full-stack **Machine Learning platform** built with Flask that lets users select datasets, perform EDA, and train ML models — powered by **Formula 1 racing data**.

---

## ✨ Features

- **Dataset Selection** — Browse and select any CSV file from the datasets folder
- **Exploratory Data Analysis (EDA)** — Automated statistical analysis with visualizations
- **3 ML Tasks:**
  - 🔢 **Regression** — Predict F1 lap times using Random Forest
  - 🏆 **Classification** — Predict podium finishes using Gradient Boosting
  - 🧩 **Clustering** — Segment F1 drivers using K-Means + PCA
- **Auto Data Cleaning** — Missing value handling, encoding, scaling
- **Interactive Visualizations** — Confusion matrices, feature importance, actual vs predicted plots
- **Plain Language Explanations** — Model results explained in simple terms

---

## 🏗️ Project Structure

```
Full Stack ML Project/
├── app.py                  # Flask web server (API + routes)
├── train_models.py         # Model training pipeline
├── prepare_datasets.py     # Dataset preparation from raw F1 data
├── download_datasets.py    # Script to download F1 datasets
├── datasets/               # CSV datasets
├── models/                 # Trained model files (.pkl)
├── static/                 # CSS, JS, plots
└── templates/              # HTML templates
```

---

## ▶️ How to Run

### 1. Install Dependencies
```bash
pip install flask flask-cors scikit-learn pandas numpy matplotlib joblib
```

### 2. Prepare Datasets
```bash
python download_datasets.py
python prepare_datasets.py
```

### 3. Train Models
```bash
python train_models.py
```

### 4. Start the Server
```bash
python app.py
```
Open **http://localhost:5000** in your browser.

---

## 🧠 Tech Stack
- **Backend:** Flask, Scikit-learn, Pandas, NumPy
- **Frontend:** HTML, CSS, JavaScript
- **ML Models:** Random Forest, Gradient Boosting, K-Means
- **Visualization:** Matplotlib, Base64-encoded plots
