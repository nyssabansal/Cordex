# 🫀 Cordex — Heart Disease Prediction Using Machine Learning

**Cordex** is a supervised machine learning project that predicts the presence of heart disease using clinical patient data. It compares four classification models, uses GridSearchCV for hyperparameter tuning, and includes an interactive Streamlit web app with **SHAP explainability** and a **What-If Simulator**.

---

## 📌 Problem Statement

Heart disease is one of the leading causes of death worldwide. Given patient medical parameters (age, cholesterol, ECG results, etc.), can we accurately predict whether a patient has heart disease?

---

## 🎯 Objectives

- Build a supervised classification model (target: 0 = no disease, 1 = heart disease)
- Compare **Logistic Regression**, **Decision Tree**, **Random Forest**, and **Neural Network**
- Tune all models using **GridSearchCV** with 5-fold cross-validation
- Evaluate using **Accuracy, Precision, Recall, F1-Score, ROC-AUC**
- Interpret predictions using **SHAP explainability**
- Deploy an interactive prediction interface using **Streamlit**

---

## 📁 Project Structure

```
Cordex/
│
├── heart.csv                        # Dataset (303 patients, 13 features)
├── heart_disease_prediction.py      # Full ML pipeline (EDA → tuning → evaluation → SHAP)
├── app.py                           # Streamlit web app (4 pages)
├── requirements.txt                 # Python dependencies
├── .gitignore
└── README.md
```

---

## 🧠 Models & Results

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC       |
|---------------------|----------|-----------|--------|----------|---------------|
| Logistic Regression | 0.8033   | 0.7838    | 0.8788 | **0.8286**   | 0.8745    |
| Decision Tree       | 0.7377   | 0.7297    | 0.8182 | 0.7714   | 0.8030        |
| Random Forest       | 0.7869   | 0.7632    | 0.8788 | 0.8169   | **0.9042** ✅ |
| Neural Network      | 0.7377   | 0.7429    | 0.7879 | 0.7647   | 0.8496        |

> All models tuned using **GridSearchCV** with 5-fold cross-validation.  
> ✅ **Random Forest** wins on ROC-AUC — best for clinical risk ranking.  
> ✅ **Logistic Regression** wins on F1-Score — best for binary clinical decisions.

---

## ✨ Unique Features

### 1. 🔬 SHAP Explainability
- **Global SHAP summary** — shows which features matter most across all patients
- **SHAP waterfall chart** — explains *exactly why* a specific patient is flagged as high-risk
- 🔴 Red bars = features pushing risk **up** | 🔵 Blue bars = features pushing risk **down**
- Goes beyond standard feature importance by showing **direction and magnitude** per patient

### 2. 🧪 What-If Simulator
- Set a **baseline patient** profile with all 13 clinical parameters
- Tweak **one parameter** at a time (e.g. reduce cholesterol, increase max heart rate)
- Instantly see the **risk score change** with a side-by-side comparison
- **SHAP Δ chart** shows exactly which feature influences shifted after the change
- Useful for simulating the effect of lifestyle changes or treatment interventions

---

## 🖥️ Streamlit App — 4 Pages

| Page | Description |
|------|-------------|
| 📊 EDA | Target distribution, age histogram, correlation heatmap, feature deep-dive |
| 📈 Model Performance | Metrics table, ROC curves, confusion matrices, SHAP global summary |
| 🔍 Predict + SHAP | Enter patient values → get risk score + SHAP waterfall explanation |
| 🧪 What-If Simulator | Tweak parameters and see risk + SHAP shift in real time |

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/Cordex.git
cd Cordex
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
Opens at `http://localhost:8501`

---

## 📊 Dataset

- **Source:** [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease)
- **Samples:** 303 patients
- **Features:** 13 clinical attributes + 1 binary target
- **Missing values:** None

| Feature | Description |
|---------|-------------|
| `age` | Age in years |
| `sex` | 1 = male, 0 = female |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl |
| `restecg` | Resting ECG results |
| `thalach` | Max heart rate achieved |
| `exang` | Exercise-induced angina |
| `oldpeak` | ST depression |
| `slope` | Slope of ST segment |
| `ca` | Number of major vessels |
| `thal` | Thalassemia type |

---

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| `scikit-learn` | Model training, GridSearchCV, evaluation metrics |
| `shap` | Model explainability (SHAP values) |
| `streamlit` | Interactive web app |
| `pandas`, `numpy` | Data processing |
| `matplotlib`, `seaborn` | Visualisation |

---

## 📄 License

MIT License — free to use and modify.
