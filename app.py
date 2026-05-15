# =============================================================================
# Heart Disease Prediction — Streamlit Web App
# =============================================================================
# Run with : streamlit run app.py
# Requires : heart.csv in the same directory
# Install  : pip install streamlit scikit-learn shap pandas matplotlib seaborn
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import shap
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .risk-high  { background:#fdecea; border-left:5px solid #E74C3C;
                  padding:16px; border-radius:6px; margin:10px 0; }
    .risk-low   { background:#eafaf1; border-left:5px solid #2ECC71;
                  padding:16px; border-radius:6px; margin:10px 0; }
    .shap-box   { background:#f0f4ff; border-left:5px solid #3498DB;
                  padding:14px; border-radius:6px; margin:10px 0; }
    .metric-card{ text-align:center; padding:12px; border-radius:8px;
                  background:#f8f9fa; border:1px solid #dee2e6; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA & MODEL (cached so they only run once)
# =============================================================================

@st.cache_data
def load_data():
    df = pd.read_csv('heart.csv')
    df.columns = df.columns.str.strip()
    return df

@st.cache_resource
def build_everything(df):
    cat_cols = ['cp','thal','slope']
    df_enc = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    X = df_enc.drop('target', axis=1)
    y = df_enc['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Best params from GridSearchCV
    models = {
        'Logistic Regression': LogisticRegression(
            C=0.01, solver='liblinear', max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=3, criterion='gini', min_samples_split=2, random_state=42),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=5, min_samples_split=5, random_state=42),
        'Neural Network': MLPClassifier(
            hidden_layer_sizes=(32,), alpha=0.01, learning_rate='constant',
            max_iter=500, random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        y_pred  = model.predict(X_test_s)
        y_proba = model.predict_proba(X_test_s)[:,1]
        results[name] = {
            'model':     model,
            'y_pred':    y_pred,
            'y_proba':   y_proba,
            'accuracy':  accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall':    recall_score(y_test, y_pred),
            'f1':        f1_score(y_test, y_pred),
            'roc_auc':   roc_auc_score(y_test, y_proba),
            'cm':        confusion_matrix(y_test, y_pred),
        }

    # SHAP explainer on Random Forest (best AUC model)
    rf_model  = results['Random Forest']['model']
    explainer = shap.TreeExplainer(rf_model)
    shap_vals = explainer.shap_values(X_test_s)   # (n, features, 2)

    return scaler, X, X_test_s, y_test, results, explainer, shap_vals

# ── Helper: encode one patient row ───────────────────────────────────────────
def encode_patient(row_dict, X_cols, scaler):
    raw = pd.DataFrame([row_dict])
    cat_cols = ['cp','thal','slope']
    raw_enc  = pd.get_dummies(raw, columns=cat_cols, drop_first=False)
    for col in X_cols.columns:
        if col not in raw_enc.columns:
            raw_enc[col] = 0
    raw_enc = raw_enc[X_cols.columns]
    return scaler.transform(raw_enc), raw_enc

# ── Helper: risk gauge bar ─────────────────────────────────────────────────
def draw_gauge(prob):
    fig, ax = plt.subplots(figsize=(7, 1.4))
    ax.barh([''], [prob],        color='#E74C3C', height=0.5)
    ax.barh([''], [1 - prob], left=[prob], color='#2ECC71', height=0.5)
    ax.axvline(0.5, color='#2C3E50', linestyle='--', linewidth=1.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Risk Probability')
    ax.set_title(f'Risk Score : {prob:.1%}', fontweight='bold', fontsize=12)
    ax.text(0.02, 0, 'Low Risk', va='center', color='white',
            fontsize=9, fontweight='bold')
    ax.text(0.98, 0, 'High Risk', va='center', color='white',
            fontsize=9, fontweight='bold', ha='right')
    fig.tight_layout()
    return fig

# ── Helper: SHAP waterfall for one patient ───────────────────────────────────
def draw_shap_waterfall(patient_scaled, explainer, feature_names, title=""):
    sv = explainer.shap_values(patient_scaled)       # (1, features, 2)
    patient_shap = sv[0, :, 1]                       # class=1 SHAP values

    top_n = 12
    sorted_idx = np.argsort(np.abs(patient_shap))[-top_n:]
    vals   = patient_shap[sorted_idx]
    feats  = [feature_names[i] for i in sorted_idx]
    colors = ['#E74C3C' if v > 0 else '#3498DB' for v in vals]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(feats, vals, color=colors, edgecolor='white', height=0.6)
    ax.axvline(0, color='#2C3E50', linewidth=1)
    for bar, val in zip(bars, vals):
        ax.text(val + (0.004 if val >= 0 else -0.004),
                bar.get_y() + bar.get_height() / 2,
                f'{val:+.3f}', va='center',
                ha='left' if val >= 0 else 'right', fontsize=9)
    red_patch  = mpatches.Patch(color='#E74C3C', label='↑ Increases disease risk')
    blue_patch = mpatches.Patch(color='#3498DB', label='↓ Decreases disease risk')
    ax.legend(handles=[red_patch, blue_patch], fontsize=9, loc='lower right')
    ax.set_title(title or 'SHAP Explanation — Why this prediction?',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('SHAP Value (impact on model output)')
    fig.tight_layout()
    return fig, patient_shap

# =============================================================================
# LOAD
# =============================================================================

df = load_data()
with st.spinner("Training models…"):
    scaler, X_cols, X_test_s, y_test, results, explainer, shap_vals = build_everything(df)

model_names  = list(results.keys())
short_names  = ['LR','DT','RF','NN']
colors_model = ['#3498DB','#F39C12','#2ECC71','#9B59B6']

# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.title("🫀 Heart Disease Predictor")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "📊 EDA",
    "📈 Model Performance",
    "🔍 Predict + SHAP",
    "🧪 What-If Simulator",
])
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Models used** (GridSearchCV tuned)\n"
    "- Logistic Regression\n- Decision Tree\n- Random Forest\n- Neural Network"
)

# =============================================================================
# PAGE 1 — EDA
# =============================================================================

if page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")
    st.markdown(f"**{df.shape[0]} patients · {df.shape[1]-1} features · 0 missing values**")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Patients", df.shape[0])
    c2.metric("Heart Disease",  int(df['target'].sum()))
    c3.metric("No Disease",     int((df['target']==0).sum()))
    c4.metric("Features",       df.shape[1]-1)

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["📌 Distributions", "🌡️ Correlations", "🔬 Feature Deep-Dive"])

    with tab1:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        tc = df['target'].value_counts()
        axes[0].pie(tc, labels=['Heart Disease','No Disease'], autopct='%1.1f%%',
                    colors=['#E74C3C','#2ECC71'],
                    wedgeprops={'edgecolor':'white','linewidth':2})
        axes[0].set_title('Target Distribution')
        for t, col, lbl in zip([0,1],['#E74C3C','#2ECC71'],['No Disease','Heart Disease']):
            axes[1].hist(df[df['target']==t]['age'], bins=15, alpha=0.7,
                         color=col, label=lbl, edgecolor='white')
        axes[1].set_title('Age Distribution by Target'); axes[1].legend()
        st.pyplot(fig); plt.close()

        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
        cp_counts = df.groupby(['cp','target']).size().unstack(fill_value=0)
        cp_counts.plot(kind='bar', ax=axes2[0], color=['#E74C3C','#2ECC71'],
                       edgecolor='white', rot=0)
        axes2[0].set_title('Chest Pain Type vs Target')
        axes2[0].legend(['No Disease','Heart Disease'])
        df_plot = df.copy()
        df_plot['Target'] = df_plot['target'].map({0:'No Disease',1:'Heart Disease'})
        sns.boxplot(data=df_plot, x='Target', y='thalach', ax=axes2[1],
                    palette={'No Disease':'#E74C3C','Heart Disease':'#2ECC71'})
        axes2[1].set_title('Max Heart Rate by Target'); axes2[1].set_xlabel('')
        st.pyplot(fig2); plt.close()

    with tab2:
        fig, ax = plt.subplots(figsize=(10, 7))
        corr = df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, ax=ax, cmap='RdYlGn', annot=True,
                    fmt='.2f', linewidths=.5, annot_kws={'size':7}, center=0)
        ax.set_title('Feature Correlation Matrix')
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        st.pyplot(fig); plt.close()
        st.info("**Key correlations with target:** thalach (+), cp (+), exang (−), oldpeak (−), ca (−)")

    with tab3:
        feature = st.selectbox("Select a feature to explore:",
                               ['thalach','oldpeak','chol','trestbps','age','ca'])
        df_plot2 = df.copy()
        df_plot2['Target'] = df_plot2['target'].map({0:'No Disease',1:'Heart Disease'})
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.boxplot(data=df_plot2, x='Target', y=feature, ax=axes[0],
                    palette={'No Disease':'#E74C3C','Heart Disease':'#2ECC71'})
        axes[0].set_title(f'{feature} — Boxplot by Target'); axes[0].set_xlabel('')
        for t, col, lbl in zip([0,1],['#E74C3C','#2ECC71'],['No Disease','Heart Disease']):
            axes[1].hist(df_plot2[df_plot2['target']==t][feature], bins=18,
                         alpha=0.7, color=col, label=lbl, edgecolor='white')
        axes[1].set_title(f'{feature} — Histogram by Target'); axes[1].legend()
        st.pyplot(fig); plt.close()

# =============================================================================
# PAGE 2 — Model Performance
# =============================================================================

elif page == "📈 Model Performance":
    st.title("📈 Model Evaluation (GridSearchCV Tuned)")

    # Metrics table
    metrics_df = pd.DataFrame({
        name: {
            'Accuracy':  round(r['accuracy'], 4),
            'Precision': round(r['precision'],4),
            'Recall':    round(r['recall'],   4),
            'F1-Score':  round(r['f1'],       4),
            'ROC-AUC':   round(r['roc_auc'],  4),
        }
        for name, r in results.items()
    }).T
    st.dataframe(metrics_df.style.highlight_max(axis=0, color='#d4edda')
                                  .highlight_min(axis=0, color='#f8d7da'),
                 use_container_width=True)

    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs([
        "📉 ROC Curves", "🟦 Confusion Matrices",
        "🌲 Feature Importance", "🔮 SHAP Global"
    ])

    with tab1:
        fig, ax = plt.subplots(figsize=(7, 5))
        for (name,r), col, sn in zip(results.items(), colors_model, short_names):
            fpr, tpr, _ = roc_curve(y_test, r['y_proba'])
            ax.plot(fpr, tpr, color=col, lw=2,
                    label=f"{sn} (AUC={r['roc_auc']:.3f})")
        ax.plot([0,1],[0,1],'--',color='grey',lw=1)
        ax.set_title('ROC Curves — All Models')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(fontsize=10)
        st.pyplot(fig); plt.close()

    with tab2:
        cols = st.columns(2)
        for i, (name, r) in enumerate(results.items()):
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(r['cm'], annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['No Disease','Disease'],
                        yticklabels=['No Disease','Disease'],
                        linewidths=1, linecolor='white', cbar=False)
            ax.set_title(name, fontsize=10, fontweight='bold')
            ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')
            cols[i % 2].pyplot(fig); plt.close()

    with tab3:
        rf = results['Random Forest']['model']
        feat_df = (pd.DataFrame({'Feature':X_cols.columns,
                                  'Importance':rf.feature_importances_})
                     .sort_values('Importance', ascending=True))
        fig, ax = plt.subplots(figsize=(9, 6))
        bars = ax.barh(feat_df['Feature'], feat_df['Importance'],
                       color=plt.cm.RdYlGn(
                           feat_df['Importance']/feat_df['Importance'].max()),
                       edgecolor='white')
        for bar, val in zip(bars, feat_df['Importance']):
            ax.text(val+0.001, bar.get_y()+bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=8)
        ax.set_title('Feature Importance – Random Forest', fontweight='bold')
        ax.set_xlabel('Importance Score')
        st.pyplot(fig); plt.close()

    with tab4:
        st.markdown("**Global SHAP Summary** — average impact of each feature across all test patients (Random Forest)")
        shap_class1  = shap_vals[:,:,1]
        mean_shap    = np.abs(shap_class1).mean(axis=0)
        shap_feat_df = (pd.DataFrame({'Feature':X_cols.columns,'Mean |SHAP|':mean_shap})
                          .sort_values('Mean |SHAP|', ascending=True))
        fig, ax = plt.subplots(figsize=(9, 6))
        bars = ax.barh(shap_feat_df['Feature'], shap_feat_df['Mean |SHAP|'],
                       color=plt.cm.RdYlGn(
                           shap_feat_df['Mean |SHAP|']/shap_feat_df['Mean |SHAP|'].max()),
                       edgecolor='white')
        for bar, val in zip(bars, shap_feat_df['Mean |SHAP|']):
            ax.text(val+0.001, bar.get_y()+bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=8)
        ax.set_title('SHAP Global Feature Importance (Mean |SHAP|)', fontweight='bold')
        ax.set_xlabel('Mean |SHAP Value| — Average impact on model output')
        st.pyplot(fig); plt.close()
        st.markdown('<div class="shap-box">💡 <b>SHAP vs. standard feature importance:</b> '
                    'SHAP values show the <i>direction and magnitude</i> of each feature\'s '
                    'contribution to each individual prediction — not just an average score. '
                    'This makes the model clinically interpretable.</div>',
                    unsafe_allow_html=True)

# =============================================================================
# PAGE 3 — Predict + SHAP
# =============================================================================

elif page == "🔍 Predict + SHAP":
    st.title("🔍 Patient Prediction with SHAP Explanation")
    st.markdown("Enter clinical parameters → get a risk score + an explanation of *why*.")

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age      = st.slider("Age", 20, 80, 55)
            sex      = st.selectbox("Sex", [1,0],
                                    format_func=lambda x:"Male" if x==1 else "Female")
            cp       = st.selectbox("Chest Pain Type",  [0,1,2,3],
                                    format_func=lambda x:{0:'Typical Angina',
                                    1:'Atypical Angina',2:'Non-anginal',3:'Asymptomatic'}[x])
            trestbps = st.slider("Resting BP (mm Hg)", 90, 200, 130)
            chol     = st.slider("Cholesterol (mg/dl)", 100, 600, 240)
        with c2:
            fbs     = st.selectbox("Fasting Blood Sugar > 120", [0,1],
                                   format_func=lambda x:"Yes" if x==1 else "No")
            restecg = st.selectbox("Resting ECG", [0,1,2],
                                   format_func=lambda x:{0:'Normal',
                                   1:'ST-T Abnormality',2:'LV Hypertrophy'}[x])
            thalach = st.slider("Max Heart Rate", 60, 210, 150)
            exang   = st.selectbox("Exercise Angina", [0,1],
                                   format_func=lambda x:"Yes" if x==1 else "No")
            oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.5, 1.0, 0.1)
        with c3:
            slope   = st.selectbox("ST Slope", [0,1,2],
                                   format_func=lambda x:{0:'Upsloping',
                                   1:'Flat',2:'Downsloping'}[x])
            ca      = st.selectbox("Major Vessels (ca)", [0,1,2,3,4])
            thal    = st.selectbox("Thalassemia", [0,1,2,3],
                                   format_func=lambda x:{0:'Normal',1:'Fixed Defect',
                                   2:'Reversible',3:'Unknown'}[x])
            model_choice = st.selectbox("Model", model_names)
        submitted = st.form_submit_button("🔮 Predict & Explain", use_container_width=True)

    if submitted:
        patient = dict(age=age, sex=sex, trestbps=trestbps, chol=chol,
                       fbs=fbs, restecg=restecg, thalach=thalach, exang=exang,
                       oldpeak=oldpeak, ca=ca, cp=cp, thal=thal, slope=slope)
        patient_scaled, _ = encode_patient(patient, X_cols, scaler)

        chosen = results[model_choice]['model']
        pred   = chosen.predict(patient_scaled)[0]
        prob   = chosen.predict_proba(patient_scaled)[0][1]

        st.markdown("---")
        # Result banner
        if pred == 1:
            st.markdown(f'<div class="risk-high">⚠️ <b>Heart Disease Detected</b> &nbsp;|&nbsp; '
                        f'Risk Score: <b>{prob:.1%}</b></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="risk-low">✅ <b>No Heart Disease Detected</b> &nbsp;|&nbsp; '
                        f'Risk Score: <b>{prob:.1%}</b></div>', unsafe_allow_html=True)

        col_g, col_s = st.columns([1, 1])
        with col_g:
            st.pyplot(draw_gauge(prob)); plt.close()
            st.caption(f"Model: **{model_choice}** · "
                       f"Test Accuracy: {results[model_choice]['accuracy']*100:.1f}% · "
                       f"ROC-AUC: {results[model_choice]['roc_auc']:.3f}")

        with col_s:
            st.markdown("### 🔬 SHAP Explanation")
            st.markdown("Why did the model give this score?")
            # Only RF has TreeExplainer; for others fall back to RF explanation note
            if model_choice == 'Random Forest':
                fig_shap, _ = draw_shap_waterfall(
                    patient_scaled, explainer, list(X_cols.columns),
                    title=f"SHAP Waterfall — {model_choice}")
                st.pyplot(fig_shap); plt.close()
            else:
                st.info("SHAP waterfall is shown using the Random Forest explainer "
                        "(most accurate for tree-based SHAP). "
                        "The selected model's prediction is shown above.")
                fig_shap, _ = draw_shap_waterfall(
                    patient_scaled, explainer, list(X_cols.columns),
                    title="SHAP Waterfall — Random Forest (reference explainer)")
                st.pyplot(fig_shap); plt.close()

        st.markdown(
            '<div class="shap-box">💡 <b>How to read this chart:</b> '
            '<span style="color:#E74C3C">■ Red bars</span> push the risk <b>higher</b>. '
            '<span style="color:#3498DB">■ Blue bars</span> push the risk <b>lower</b>. '
            'Longer bars = stronger influence on this patient\'s prediction.</div>',
            unsafe_allow_html=True)

        st.caption("⚕️ For educational purposes only. Not a substitute for medical diagnosis.")

# =============================================================================
# PAGE 4 — What-If Simulator
# =============================================================================

elif page == "🧪 What-If Simulator":
    st.title("🧪 What-If Simulator")
    st.markdown(
        "Set a **baseline patient**, then adjust individual parameters to see "
        "how the risk score and SHAP explanation change in real time."
    )

    st.markdown("### Step 1 — Set Baseline Patient")
    with st.expander("Baseline patient parameters", expanded=True):
        bc1, bc2, bc3 = st.columns(3)
        with bc1:
            b_age      = st.slider("Age",               20,  80,  58, key='b_age')
            b_sex      = st.selectbox("Sex", [1,0],
                                      format_func=lambda x:"Male" if x==1 else "Female",
                                      key='b_sex')
            b_cp       = st.selectbox("Chest Pain Type", [0,1,2,3],
                                      format_func=lambda x:{0:'Typical Angina',
                                      1:'Atypical Angina',2:'Non-anginal',3:'Asymptomatic'}[x],
                                      index=3, key='b_cp')
            b_trestbps = st.slider("Resting BP",        90, 200, 150, key='b_trestbps')
            b_chol     = st.slider("Cholesterol",       100, 600, 280, key='b_chol')
        with bc2:
            b_fbs     = st.selectbox("Fasting Blood Sugar > 120", [0,1],
                                     format_func=lambda x:"Yes" if x==1 else "No",
                                     key='b_fbs')
            b_restecg = st.selectbox("Resting ECG", [0,1,2],
                                     format_func=lambda x:{0:'Normal',
                                     1:'ST-T Abnormality',2:'LV Hypertrophy'}[x],
                                     key='b_restecg')
            b_thalach = st.slider("Max Heart Rate", 60, 210, 130, key='b_thalach')
            b_exang   = st.selectbox("Exercise Angina", [0,1],
                                     format_func=lambda x:"Yes" if x==1 else "No",
                                     index=1, key='b_exang')
            b_oldpeak = st.slider("ST Depression",   0.0, 6.5, 3.5, 0.1, key='b_oldpeak')
        with bc3:
            b_slope = st.selectbox("ST Slope", [0,1,2],
                                   format_func=lambda x:{0:'Upsloping',
                                   1:'Flat',2:'Downsloping'}[x],
                                   index=1, key='b_slope')
            b_ca    = st.selectbox("Major Vessels (ca)", [0,1,2,3,4],
                                   index=2, key='b_ca')
            b_thal  = st.selectbox("Thalassemia", [0,1,2,3],
                                   format_func=lambda x:{0:'Normal',1:'Fixed Defect',
                                   2:'Reversible',3:'Unknown'}[x],
                                   index=2, key='b_thal')

    baseline = dict(age=b_age, sex=b_sex, trestbps=b_trestbps, chol=b_chol,
                    fbs=b_fbs, restecg=b_restecg, thalach=b_thalach, exang=b_exang,
                    oldpeak=b_oldpeak, ca=b_ca, cp=b_cp, thal=b_thal, slope=b_slope)

    baseline_scaled, _ = encode_patient(baseline, X_cols, scaler)
    rf_model = results['Random Forest']['model']
    baseline_prob = rf_model.predict_proba(baseline_scaled)[0][1]

    st.markdown("---")
    st.markdown("### Step 2 — Tweak One Parameter")
    st.markdown("Change a single clinical value below and instantly see how risk shifts.")

    wi_col1, wi_col2 = st.columns([1, 2])
    with wi_col1:
        tweak_param = st.selectbox("Parameter to adjust:", [
            'thalach','oldpeak','chol','trestbps','age','ca',
            'cp','thal','slope','exang','sex','fbs','restecg'
        ])

        # Render appropriate widget based on type
        continuous_params = {
            'thalach':  (60,  210, int(b_thalach)),
            'oldpeak':  (0.0, 6.5, float(b_oldpeak), 0.1),
            'chol':     (100, 600, int(b_chol)),
            'trestbps': (90,  200, int(b_trestbps)),
            'age':      (20,  80,  int(b_age)),
        }
        discrete_params = {
            'ca':      ([0,1,2,3,4],     int(b_ca)),
            'cp':      ([0,1,2,3],       int(b_cp)),
            'thal':    ([0,1,2,3],       int(b_thal)),
            'slope':   ([0,1,2],         int(b_slope)),
            'exang':   ([0,1],           int(b_exang)),
            'sex':     ([0,1],           int(b_sex)),
            'fbs':     ([0,1],           int(b_fbs)),
            'restecg': ([0,1,2],         int(b_restecg)),
        }

        if tweak_param in continuous_params:
            info = continuous_params[tweak_param]
            if len(info) == 4:
                new_val = st.slider(f"New {tweak_param}", info[0], info[1], info[2], info[3])
            else:
                new_val = st.slider(f"New {tweak_param}", info[0], info[1], info[2])
        else:
            info = discrete_params[tweak_param]
            new_val = st.selectbox(f"New {tweak_param}", info[0],
                                   index=info[0].index(info[1]))

    # Build modified patient
    modified = {**baseline, tweak_param: new_val}
    modified_scaled, _ = encode_patient(modified, X_cols, scaler)
    modified_prob = rf_model.predict_proba(modified_scaled)[0][1]
    delta = modified_prob - baseline_prob

    with wi_col2:
        st.markdown("#### Risk Comparison")
        m1, m2, m3 = st.columns(3)
        m1.metric("Baseline Risk",  f"{baseline_prob:.1%}")
        m2.metric("Modified Risk",  f"{modified_prob:.1%}",
                  delta=f"{delta:+.1%}",
                  delta_color="inverse")
        m3.metric("Change",         f"{delta:+.1%}",
                  delta_color="inverse")

        # Side-by-side gauge
        fig_cmp, axes_cmp = plt.subplots(1, 2, figsize=(10, 1.5))
        for ax_g, prob_g, title_g in zip(
                axes_cmp,
                [baseline_prob, modified_prob],
                ['Baseline', f'Modified ({tweak_param}={new_val})']):
            ax_g.barh([''], [prob_g],       color='#E74C3C', height=0.5)
            ax_g.barh([''], [1-prob_g], left=[prob_g], color='#2ECC71', height=0.5)
            ax_g.axvline(0.5, color='#2C3E50', linestyle='--', linewidth=1.2)
            ax_g.set_xlim(0, 1)
            ax_g.set_title(f'{title_g}: {prob_g:.1%}', fontweight='bold', fontsize=10)
            ax_g.set_xlabel('Risk Probability')
        fig_cmp.tight_layout()
        st.pyplot(fig_cmp); plt.close()

        if abs(delta) < 0.01:
            st.info(f"Changing **{tweak_param}** to **{new_val}** has minimal effect on risk.")
        elif delta > 0:
            st.markdown(f'<div class="risk-high">⚠️ Changing <b>{tweak_param}</b> to <b>{new_val}</b> '
                        f'<b>increases</b> risk by <b>{delta:+.1%}</b></div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="risk-low">✅ Changing <b>{tweak_param}</b> to <b>{new_val}</b> '
                        f'<b>decreases</b> risk by <b>{delta:.1%}</b></div>',
                        unsafe_allow_html=True)

    # SHAP comparison — baseline vs modified
    st.markdown("---")
    st.markdown("### Step 3 — SHAP Explanation: Before vs After")
    st.markdown("See which features changed their influence after the tweak.")

    shap_col1, shap_col2 = st.columns(2)
    with shap_col1:
        st.markdown(f"**Baseline** — Risk: {baseline_prob:.1%}")
        fig_b, sv_b = draw_shap_waterfall(
            baseline_scaled, explainer, list(X_cols.columns),
            title=f"Baseline SHAP Waterfall")
        st.pyplot(fig_b); plt.close()

    with shap_col2:
        st.markdown(f"**Modified** ({tweak_param}={new_val}) — Risk: {modified_prob:.1%}")
        fig_m, sv_m = draw_shap_waterfall(
            modified_scaled, explainer, list(X_cols.columns),
            title=f"Modified SHAP Waterfall")
        st.pyplot(fig_m); plt.close()

    # Delta SHAP bar chart
    st.markdown("#### SHAP Δ — Which features changed their influence most?")
    sv_b_vals = explainer.shap_values(baseline_scaled)[0,:,1]
    sv_m_vals = explainer.shap_values(modified_scaled)[0,:,1]
    delta_shap = sv_m_vals - sv_b_vals

    top_changed = np.argsort(np.abs(delta_shap))[-10:]
    fig_d, ax_d = plt.subplots(figsize=(9, 4))
    d_colors = ['#E74C3C' if v>0 else '#3498DB' for v in delta_shap[top_changed]]
    ax_d.barh([X_cols.columns[i] for i in top_changed],
              delta_shap[top_changed], color=d_colors, edgecolor='white', height=0.55)
    ax_d.axvline(0, color='#2C3E50', linewidth=1)
    ax_d.set_title('SHAP Δ (Modified − Baseline) — Top Changed Features', fontweight='bold')
    ax_d.set_xlabel('Change in SHAP value')
    plt.tight_layout()
    st.pyplot(fig_d); plt.close()

    st.markdown(
        '<div class="shap-box">💡 <b>How to use this simulator:</b> Try reducing '
        '<b>oldpeak</b> (ST depression) or increasing <b>thalach</b> (max heart rate) '
        'to see how lifestyle or treatment changes could lower a patient\'s risk. '
        'The SHAP Δ chart shows exactly which features shifted their influence.</div>',
        unsafe_allow_html=True)

    st.caption("⚕️ For educational purposes only. Not a substitute for medical advice.")
