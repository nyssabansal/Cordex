# 🫀 Heart Disease Prediction Using Machine Learning

A supervised machine learning project that predicts the presence of heart disease using clinical patient data. Built as part of a major ML project, it compares four classification models and includes an interactive Streamlit web app with SHAP explainability and a What-If Simulator.

---

## 📌 Problem Statement

Heart disease is one of the leading causes of death worldwide. Given patient medical parameters (age, cholesterol, ECG results, etc.), can we accurately predict whether a patient has heart disease?

---

## 🎯 Objectives

- Build a supervised classification model (target: 0 or 1)
- Compare **Logistic Regression**, **Decision Tree**, **Random Forest**, and **Neural Network**
- Evaluate using **Accuracy, Precision, Recall, F1-Score, ROC-AUC**
- Interpret feature importance and understand key risk factors
- Deploy an interactive prediction interface using **Streamlit**

---

## 📁 Project Structure

```
heart-disease-prediction/
│
├── heart.csv                        # Dataset (303 patients, 13 features)
├── heart_disease_prediction.py      # Full ML pipeline (EDA → tuning → evaluation → SHAP)
├── app.py                           # Streamlit web app
│
├── outputs/                         # Generated plots
│   ├── eda.png                      # Exploratory Data Analysis charts
│   ├── evaluation.png               # Model comparison (metrics, ROC, confusion matrices)
│   ├── feature_importance.png       # Random Forest feature importance
│   ├── shap_summary.png             # SHAP global feature importance
│   └── shap_waterfall.png           # SHAP waterfall for highest-risk patient
│
├── requirements.txt                 # Python dependencies
└── README.md
```

---

## 🧠 Models & Results

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.8033   | 0.7838    | 0.8788 | 0.8286   | 0.8745  |
| Decision Tree       | 0.7377   | 0.7297    | 0.8182 | 0.7714   | 0.8030  |
| Random Forest       | 0.7869   | 0.7632    | 0.8788 | 0.8169   | **0.9042** |
| Neural Network      | 0.7377   | 0.7429    | 0.7879 | 0.7647   | 0.8496  |

> All models tuned using **GridSearchCV** with 5-fold cross-validation.

---

## 🔬 Unique Features

### 1. SHAP Explainability
- **Global SHAP summary** — which features matter most across all patients
- **SHAP waterfall chart** — explains *why* a specific patient is classified as high-risk
- Red bars = features increasing risk | Blue bars = features decreasing risk

### 2. What-If Simulator (in the Streamlit app)
- Set a baseline patient profile
- Adjust a single clinical parameter (e.g. lower cholesterol by 40)
- Instantly see how the risk score changes
- SHAP Δ chart shows which feature influences shifted

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/heart-disease-prediction.git
cd heart-disease-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the ML pipeline
```bash
python heart_disease_prediction.py
```

### 4. Launch the Streamlit app
```bash
streamlit run app.py
```

---

## 📊 Dataset

- **Source:** [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease)
- **Samples:** 303 patients
- **Features:** 13 clinical attributes + 1 binary target
- **Missing values:** None

Key features: `age`, `sex`, `cp` (chest pain type), `thalach` (max heart rate), `oldpeak` (ST depression), `ca` (vessels), `thal` (thalassemia)

---

## 🛠️ Tech Stack

- **Python 3.x**
- `scikit-learn` — model training, GridSearchCV, evaluation
- `shap` — model explainability
- `streamlit` — web app deployment
- `pandas`, `numpy` — data processing
- `matplotlib`, `seaborn` — visualisation

---

## 📄 License

MIT License — free to use and modify.
