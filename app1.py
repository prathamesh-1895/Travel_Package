import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TravelML · Holiday Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── GLOBAL STYLES ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1040 50%, #0f0c29 100%);
    color: #e8e0f5;
}

section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04);
    border-right: 1px solid rgba(255,255,255,0.08);
}

div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 20px;
    backdrop-filter: blur(12px);
}
div[data-testid="metric-container"] label { color: #a78bfa !important; font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase; }
div[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #f0e6ff !important; font-family: 'Syne', sans-serif; font-weight: 700; }

h1, h2, h3 { font-family: 'Syne', sans-serif !important; color: #f0e6ff !important; }

.stTabs [data-baseweb="tab-list"] { gap: 8px; background: rgba(255,255,255,0.04); border-radius: 12px; padding: 6px; }
.stTabs [data-baseweb="tab"] { border-radius: 8px; padding: 8px 20px; color: #a78bfa; font-weight: 500; border: none; background: transparent; }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, #7c3aed, #a855f7) !important; color: white !important; }

.stSelectbox [data-baseweb="select"] > div { background: rgba(255,255,255,0.06); border: 1px solid rgba(167,139,250,0.3); border-radius: 10px; color: #f0e6ff; }

.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #a855f7);
    color: white; border: none; border-radius: 12px;
    padding: 12px 28px; font-family: 'Syne', sans-serif;
    font-weight: 600; letter-spacing: 0.04em; transition: all 0.2s;
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(124,58,237,0.5); }

.streamlit-expanderHeader { background: rgba(255,255,255,0.04) !important; border-radius: 10px !important; color: #a78bfa !important; }
.stDataFrame { border-radius: 12px; overflow: hidden; }
.stProgress > div > div > div { background: linear-gradient(90deg, #7c3aed, #a855f7); }

.section-title { font-family: 'Syne', sans-serif; font-size: 1.6rem; font-weight: 800; color: #f0e6ff; letter-spacing: -0.02em; margin-bottom: 0.2rem; }
.section-sub { font-family: 'DM Sans', sans-serif; font-size: 0.9rem; color: #9d89c4; margin-bottom: 1.5rem; }
.tag { display: inline-block; background: rgba(124,58,237,0.2); border: 1px solid rgba(124,58,237,0.4); color: #c4b5fd; border-radius: 6px; padding: 2px 10px; font-size: 0.78rem; margin-right: 6px; }

.predict-card {
    background: rgba(124,58,237,0.12);
    border: 1px solid rgba(124,58,237,0.35);
    border-radius: 20px;
    padding: 28px 32px;
    margin: 16px 0;
}
.predict-success {
    background: rgba(16,185,129,0.12);
    border: 2px solid rgba(16,185,129,0.5);
    border-radius: 16px;
    padding: 20px 24px;
    font-family: 'Syne', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: #6ee7b7;
    text-align: center;
}
.predict-fail {
    background: rgba(236,72,153,0.12);
    border: 2px solid rgba(236,72,153,0.5);
    border-radius: 16px;
    padding: 20px 24px;
    font-family: 'Syne', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: #f9a8d4;
    text-align: center;
}
.model-badge {
    display: inline-block;
    background: linear-gradient(135deg,#7c3aed,#a855f7);
    color: white;
    border-radius: 8px;
    padding: 3px 12px;
    font-size: 0.8rem;
    font-weight: 600;
    font-family: 'Syne', sans-serif;
    margin-bottom: 8px;
}
.info-box {
    background: rgba(6,182,212,0.08);
    border: 1px solid rgba(6,182,212,0.25);
    border-radius: 12px;
    padding: 14px 18px;
    font-size: 0.85rem;
    color: #a5f3fc;
    margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.03)",
    font=dict(family="DM Sans, sans-serif", color="#c4b5fd"),
    title_font=dict(family="Syne, sans-serif", color="#f0e6ff", size=16),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)"),
    legend=dict(bgcolor="rgba(255,255,255,0.05)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
    margin=dict(t=50, b=40, l=40, r=20),
)
PALETTE = ["#a855f7", "#ec4899", "#06b6d4", "#10b981", "#f59e0b", "#ef4444"]


# ─── DATA LOADING & PREPROCESSING ────────────────────────────────────────────
@st.cache_data
def load_and_clean():
    import os
    if os.path.exists("Travel.csv"):
        df = pd.read_csv("Travel.csv")
    else:
        np.random.seed(42)
        n = 4888
        df = pd.DataFrame({
            "CustomerID": range(1, n+1),
            "ProdTaken": np.random.choice([0, 1], n, p=[0.82, 0.18]),
            "Age": np.random.normal(37, 9, n).clip(18, 65).astype(int),
            "TypeofContact": np.random.choice(["Self Enquiry", "Company Invited"], n, p=[0.7, 0.3]),
            "CityTier": np.random.choice([1, 2, 3], n, p=[0.4, 0.35, 0.25]),
            "DurationOfPitch": np.random.normal(15, 5, n).clip(5, 30).astype(int),
            "Occupation": np.random.choice(["Salaried","Small Business","Large Business","Free Lancer"], n, p=[0.5,0.25,0.15,0.1]),
            "Gender": np.random.choice(["Male","Female","Fe Male"], n, p=[0.6,0.35,0.05]),
            "NumberOfPersonVisiting": np.random.randint(1, 6, n),
            "NumberOfFollowups": np.random.choice([1.,2.,3.,4.,5.,6.], n),
            "ProductPitched": np.random.choice(["Basic","Standard","Deluxe","Super Deluxe","King"], n),
            "PreferredPropertyStar": np.random.choice([3.,4.,5.], n),
            "MaritalStatus": np.random.choice(["Single","Married","Divorced","Unmarried"], n, p=[0.25,0.5,0.1,0.15]),
            "NumberOfTrips": np.random.choice([1.,2.,3.,4.,5.,6.,7.], n),
            "Passport": np.random.choice([0,1], n, p=[0.7,0.3]),
            "PitchSatisfactionScore": np.random.randint(1, 6, n),
            "OwnCar": np.random.choice([0,1], n, p=[0.4,0.6]),
            "NumberOfChildrenVisiting": np.random.choice([0.,1.,2.,3.], n),
            "Designation": np.random.choice(["Executive","Manager","Senior Manager","AVP","VP"], n, p=[0.3,0.3,0.2,0.12,0.08]),
            "MonthlyIncome": np.random.normal(23000, 7000, n).clip(10000, 60000).astype(int)
        })
        for col, frac in [("Age",0.03),("DurationOfPitch",0.03),("NumberOfFollowups",0.05),
                          ("PreferredPropertyStar",0.04),("NumberOfTrips",0.04),
                          ("NumberOfChildrenVisiting",0.03),("MonthlyIncome",0.04),("TypeofContact",0.02)]:
            idx = np.random.choice(df.index, int(n*frac), replace=False)
            df.loc[idx, col] = np.nan

    df_raw = df.copy()

    # Clean
    df["Gender"] = df["Gender"].replace("Fe Male", "Female")
    df["MaritalStatus"] = df["MaritalStatus"].replace("Single", "Unmarried")

    # Impute numerics with median
    for col in ["Age","DurationOfPitch","NumberOfTrips","MonthlyIncome"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Impute categoricals / other numeric with mode
    for col in ["TypeofContact","NumberOfFollowups","PreferredPropertyStar","NumberOfChildrenVisiting"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    df.drop(columns=["CustomerID"], inplace=True, errors="ignore")

    # Feature engineering
    df["TotalVisiting"] = df["NumberOfPersonVisiting"] + df["NumberOfChildrenVisiting"]
    df.drop(columns=["NumberOfPersonVisiting","NumberOfChildrenVisiting"], inplace=True, errors="ignore")

    return df_raw, df


# ─── MODEL TRAINING — FIXED ───────────────────────────────────────────────────
@st.cache_data
def run_models(df):
    X = df.drop(["ProdTaken"], axis=1)
    y = df["ProdTaken"]

    # ── Stratified split (preserves class ratio) ──────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = X.select_dtypes(exclude="object").columns.tolist()

    # ── Preprocessor ─────────────────────────────────────────────────────────
    # handle_unknown='ignore' keeps inference safe; try sparse_output first
    try:
        ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
        _ = ohe.fit_transform(X_train[cat_cols])
    except TypeError:
        ohe = OneHotEncoder(drop="first", sparse=False, handle_unknown="ignore")

    preprocessor = ColumnTransformer([
        ("ohe", ohe, cat_cols),
        ("scaler", StandardScaler(), num_cols)
    ])

    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t  = preprocessor.transform(X_test)

    # ── SMOTE to balance training set (if available) ──────────────────────────
    if HAS_SMOTE:
        sm = SMOTE(random_state=42)
        X_train_t, y_train_sm = sm.fit_resample(X_train_t, y_train)
        y_train = pd.Series(y_train_sm)

    # ── FIX: class_weight='balanced' on applicable models ─────────────────────
    pos_ratio = (y == 0).sum() / (y == 1).sum()   # for XGBoost scale_pos_weight

    models = {
        "Logistic Regression":  LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5),
        "Decision Tree":        DecisionTreeClassifier(random_state=42, class_weight="balanced", max_depth=8),
        "Random Forest":        RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced", n_jobs=-1),
        "AdaBoost":             AdaBoostClassifier(random_state=42, n_estimators=100, learning_rate=0.5),
        "Gradient Boosting":    GradientBoostingClassifier(random_state=42, n_estimators=150, learning_rate=0.05, max_depth=4),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            eval_metric="logloss", random_state=42,
            scale_pos_weight=pos_ratio,          # FIX: handles imbalance
            n_estimators=150, learning_rate=0.05,
            max_depth=4, use_label_encoder=False
        )

    results = {}
    roc_data = {}

    for name, mdl in models.items():
        mdl.fit(X_train_t, y_train)
        y_tr_pred = mdl.predict(X_train_t)
        y_te_pred = mdl.predict(X_test_t)
        y_te_prob = mdl.predict_proba(X_test_t)[:, 1] if hasattr(mdl, "predict_proba") else None

        results[name] = {
            "Train Accuracy":  round(accuracy_score(y_train, y_tr_pred), 4),
            "Test Accuracy":   round(accuracy_score(y_test,  y_te_pred), 4),
            "Test F1":         round(f1_score(y_test, y_te_pred, average="weighted"), 4),
            "Test Precision":  round(precision_score(y_test, y_te_pred, zero_division=0), 4),
            "Test Recall":     round(recall_score(y_test, y_te_pred, zero_division=0), 4),
            "Test ROC-AUC":    round(roc_auc_score(y_test, y_te_prob if y_te_prob is not None else y_te_pred), 4),
            "Confusion":       confusion_matrix(y_test, y_te_pred),
        }
        if y_te_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_te_prob)
            roc_data[name] = (fpr, tpr)

    # ── Feature importances from Random Forest ────────────────────────────────
    rf = models["Random Forest"]
    try:
        ohe_out = preprocessor.named_transformers_["ohe"].get_feature_names_out(cat_cols)
    except Exception:
        ohe_out = []
    feat_names = list(ohe_out) + num_cols
    n_fi = len(rf.feature_importances_)
    importances = pd.DataFrame({
        "Feature":    feat_names[:n_fi],
        "Importance": rf.feature_importances_[:len(feat_names)]
    }).sort_values("Importance", ascending=False).head(15)

    return results, roc_data, importances, X_test, y_test, preprocessor, models, cat_cols, num_cols


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:20px 0 10px;">
        <div style="font-size:2.6rem;">✈️</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.15rem;font-weight:800;color:#f0e6ff;">TravelML</div>
        <div style="font-size:0.74rem;color:#9d89c4;margin-top:4px;">Holiday Package Predictor</div>
    </div>
    <hr style="border-color:rgba(255,255,255,0.08);margin:10px 0 20px;">
    """, unsafe_allow_html=True)

    nav = st.radio(
        "Navigate",
        ["📊 Data Overview", "🔍 EDA", "🤖 Model Comparison", "📈 ROC & Metrics", "🎯 Predict"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color:rgba(255,255,255,0.08);margin:20px 0;'>", unsafe_allow_html=True)

    if HAS_SMOTE:
        st.markdown('<div class="info-box">⚡ SMOTE active — minority class oversampled for balanced training.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-box">💡 Install <code>imbalanced-learn</code> to enable SMOTE oversampling.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:0.72rem;color:#6b5b95;text-align:center;">
        Dataset · Trips & Travel.Com<br>Kaggle Holiday Package Prediction
    </div>
    """, unsafe_allow_html=True)


# ─── LOAD DATA ───────────────────────────────────────────────────────────────
with st.spinner("Loading data & training models — this takes ~20s on first run…"):
    df_raw, df = load_and_clean()
    results, roc_data, importances, X_test_df, y_test, preprocessor, models, cat_cols, num_cols = run_models(df)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DATA OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if "Data Overview" in nav:
    st.markdown('<p class="section-title">📊 Data Overview</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Holiday Package Purchase Prediction · Trips & Travel.Com</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Features", f"{df.shape[1]-1}")
    c3.metric("Purchased Package", f"{df['ProdTaken'].sum():,}", f"{df['ProdTaken'].mean()*100:.1f}%")
    c4.metric("Missing (raw)", f"{df_raw.isnull().sum().sum():,}")

    st.markdown("#### Raw vs Cleaned Sample")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<span class="tag">RAW</span>', unsafe_allow_html=True)
        st.dataframe(df_raw.head(8), use_container_width=True, height=300)
    with col2:
        st.markdown('<span class="tag">CLEANED</span>', unsafe_allow_html=True)
        st.dataframe(df.head(8), use_container_width=True, height=300)

    st.markdown("#### Target Variable Distribution")
    tgt = df["ProdTaken"].value_counts().reset_index()
    tgt.columns = ["ProdTaken", "Count"]
    tgt["Label"] = tgt["ProdTaken"].map({0:"Not Purchased", 1:"Purchased"})
    fig_tgt = px.pie(tgt, names="Label", values="Count",
                     color_discrete_sequence=["#7c3aed","#ec4899"], hole=0.55)
    fig_tgt.update_traces(textfont_size=14)
    fig_tgt.update_layout(**PLOTLY_LAYOUT, title="Package Purchase Rate", height=340)
    st.plotly_chart(fig_tgt, use_container_width=True)

    st.markdown("#### Missing Values (Raw Data)")
    miss = df_raw.isnull().mean().reset_index()
    miss.columns = ["Feature","Missing %"]
    miss["Missing %"] = (miss["Missing %"]*100).round(2)
    miss = miss[miss["Missing %"] > 0].sort_values("Missing %", ascending=False)
    if not miss.empty:
        fig_miss = px.bar(miss, x="Feature", y="Missing %",
                          color="Missing %", color_continuous_scale=["#4c1d95","#a855f7","#ec4899"],
                          text="Missing %")
        fig_miss.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_miss.update_layout(**PLOTLY_LAYOUT, title="Missing Value %", height=320)
        st.plotly_chart(fig_miss, use_container_width=True)

    with st.expander("📋 Descriptive Statistics"):
        st.dataframe(df.describe().T.round(2), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA
# ═══════════════════════════════════════════════════════════════════════════════
elif "EDA" in nav:
    st.markdown('<p class="section-title">🔍 Exploratory Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Uncover patterns in your travel customer data</p>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📦 Distributions", "🔗 Correlations", "🎯 Target Insights", "📐 Boxplots"])

    with tab1:
        num_cols_list = [c for c in df.select_dtypes(include=np.number).columns if c != "ProdTaken"]
        col = st.selectbox("Select Numeric Feature", num_cols_list, key="dist_col")
        bins = st.slider("Bins", 10, 80, 30, key="bins")
        fig_h = px.histogram(df, x=col, color="ProdTaken", barmode="overlay", nbins=bins,
                             color_discrete_map={0:"#7c3aed",1:"#ec4899"}, opacity=0.75,
                             labels={"ProdTaken":"Purchased"})
        fig_h.update_layout(**PLOTLY_LAYOUT, title=f"Distribution of {col} by Purchase", height=380)
        st.plotly_chart(fig_h, use_container_width=True)

        cat_cols_list = df.select_dtypes(include="object").columns.tolist()
        cat_col = st.selectbox("Select Categorical Feature", cat_cols_list, key="cat_col")
        grp = df.groupby([cat_col,"ProdTaken"]).size().reset_index(name="Count")
        grp["ProdTaken"] = grp["ProdTaken"].map({0:"Not Purchased",1:"Purchased"})
        fig_bar = px.bar(grp, x=cat_col, y="Count", color="ProdTaken", barmode="group",
                         color_discrete_map={"Not Purchased":"#7c3aed","Purchased":"#ec4899"})
        fig_bar.update_layout(**PLOTLY_LAYOUT, title=f"{cat_col} vs Purchase", height=360)
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab2:
        corr_df = df.select_dtypes(include=np.number)
        corr_mat = corr_df.corr()
        fig_corr = px.imshow(corr_mat, text_auto=".2f", aspect="auto",
                             color_continuous_scale=["#1e1245","#7c3aed","#ec4899"])
        fig_corr.update_layout(**PLOTLY_LAYOUT, title="Correlation Heatmap", height=520)
        st.plotly_chart(fig_corr, use_container_width=True)

        c1, c2 = st.columns(2)
        x_ax = c1.selectbox("X-axis", corr_df.columns.tolist(), index=0)
        y_ax = c2.selectbox("Y-axis", corr_df.columns.tolist(), index=1)
        fig_sc = px.scatter(df, x=x_ax, y=y_ax, color=df["ProdTaken"].astype(str),
                            color_discrete_map={"0":"#7c3aed","1":"#ec4899"},
                            opacity=0.5, labels={"color":"Purchased"})
        fig_sc.update_layout(**PLOTLY_LAYOUT, title=f"{x_ax} vs {y_ax}", height=380)
        st.plotly_chart(fig_sc, use_container_width=True)

    with tab3:
        insight_cols = [c for c in ["Occupation","MaritalStatus","Gender","ProductPitched","Designation"] if c in df.columns]
        for col in insight_cols:
            rate = df.groupby(col)["ProdTaken"].mean().reset_index()
            rate.columns = [col, "Purchase Rate"]
            rate = rate.sort_values("Purchase Rate", ascending=False)
            fig_r = px.bar(rate, x=col, y="Purchase Rate",
                           color="Purchase Rate", color_continuous_scale=["#4c1d95","#a855f7","#ec4899"],
                           text="Purchase Rate")
            fig_r.update_traces(texttemplate="%{text:.1%}", textposition="outside")
            fig_r.update_layout(**PLOTLY_LAYOUT, title=f"Purchase Rate by {col}", height=310)
            st.plotly_chart(fig_r, use_container_width=True)

    with tab4:
        num_feats = [c for c in df.select_dtypes(include=np.number).columns if c != "ProdTaken"]
        box_col = st.selectbox("Feature", num_feats, key="box_col")
        fig_bx = px.box(df, x="ProdTaken", y=box_col, color="ProdTaken",
                        color_discrete_map={0:"#7c3aed",1:"#ec4899"},
                        labels={"ProdTaken":"Purchased"}, points="outliers")
        fig_bx.update_layout(**PLOTLY_LAYOUT, title=f"{box_col} by Purchase", height=420)
        st.plotly_chart(fig_bx, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
elif "Model Comparison" in nav:
    st.markdown('<p class="section-title">🤖 Model Comparison</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Six classifiers — class-balanced, stratified evaluation</p>', unsafe_allow_html=True)

    metric_choice = st.selectbox("Sort by metric",
        ["Test Accuracy","Test F1","Test Precision","Test Recall","Test ROC-AUC"])

    res_df = pd.DataFrame(results).T.reset_index().rename(columns={"index":"Model"})
    res_df = res_df.drop(columns=["Confusion"], errors="ignore")
    res_df = res_df.sort_values(metric_choice, ascending=False)

    # Bar chart
    fig_cmp = go.Figure()
    for m, col in zip(
        ["Test Accuracy","Test F1","Test Precision","Test Recall","Test ROC-AUC"], PALETTE
    ):
        fig_cmp.add_trace(go.Bar(name=m, x=res_df["Model"], y=res_df[m],
                                  marker_color=col, opacity=0.85))
    fig_cmp.update_layout(**PLOTLY_LAYOUT, barmode="group",
                          title="All Models — Test Set Metrics", height=440, xaxis_tickangle=-15)
    st.plotly_chart(fig_cmp, use_container_width=True)

    # Radar for best model
    best_name = res_df.iloc[0]["Model"]
    best = res_df.iloc[0]
    radar_metrics = ["Test Accuracy","Test F1","Test Precision","Test Recall","Test ROC-AUC"]
    radar_vals = [float(best[m]) for m in radar_metrics]
    fig_radar = go.Figure(go.Scatterpolar(
        r=radar_vals + [radar_vals[0]],
        theta=radar_metrics + [radar_metrics[0]],
        fill="toself",
        fillcolor="rgba(168,85,247,0.2)",
        line=dict(color="#a855f7", width=2),
        name=best_name
    ))
    fig_radar.update_layout(
        **PLOTLY_LAYOUT,
        polar=dict(
            radialaxis=dict(visible=True, range=[0,1], gridcolor="rgba(255,255,255,0.1)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
            bgcolor="rgba(0,0,0,0)"
        ),
        title=f"Best Model Radar · {best_name}",
        height=420
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Table
    st.markdown("#### 📋 Full Results Table")
    num_res_cols = [c for c in res_df.columns if c != "Model"]
    st.dataframe(
        res_df.style.background_gradient(
            subset=["Test Accuracy","Test F1","Test ROC-AUC"], cmap="Purples"
        ).format({c: "{:.4f}" for c in num_res_cols}),
        use_container_width=True, height=280
    )

    # Feature importances
    st.markdown("#### 🌲 Random Forest — Feature Importances")
    fig_fi = px.bar(importances.sort_values("Importance"), x="Importance", y="Feature",
                    orientation="h", color="Importance",
                    color_continuous_scale=["#4c1d95","#a855f7","#ec4899"])
    fig_fi.update_layout(**PLOTLY_LAYOUT, title="Top 15 Features (Random Forest)", height=480)
    st.plotly_chart(fig_fi, use_container_width=True)

    # Confusion matrices
    st.markdown("#### 🔲 Confusion Matrix")
    sel_model = st.selectbox("Select Model", list(results.keys()), key="cm_sel")
    cm = results[sel_model]["Confusion"]
    fig_cm = px.imshow(cm, text_auto=True, aspect="equal",
                       color_continuous_scale=["#1e1245","#7c3aed","#ec4899"],
                       labels=dict(x="Predicted", y="Actual"),
                       x=["Not Purchased","Purchased"], y=["Not Purchased","Purchased"])
    fig_cm.update_layout(**PLOTLY_LAYOUT, title=f"Confusion Matrix · {sel_model}", height=400)
    st.plotly_chart(fig_cm, use_container_width=True)

    # Quick classification report
    with st.expander(f"📋 Full Classification Report — {sel_model}"):
        mdl = models[sel_model]
        X_test_t = preprocessor.transform(X_test_df)
        y_pred = mdl.predict(X_test_t)
        rpt = classification_report(y_test, y_pred, target_names=["Not Purchased","Purchased"])
        st.code(rpt)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ROC & METRICS
# ═══════════════════════════════════════════════════════════════════════════════
elif "ROC" in nav:
    st.markdown('<p class="section-title">📈 ROC Curves & Metrics</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Receiver Operating Characteristic curves for all models</p>', unsafe_allow_html=True)

    # ── Multi-model ROC (FIXED: uses stored fpr/tpr from roc_curve) ───────────
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=[0,1], y=[0,1], mode="lines",
        line=dict(dash="dash", color="rgba(255,255,255,0.25)"),
        name="Random Baseline"
    ))
    for (name, (fpr, tpr)), col in zip(roc_data.items(), PALETTE):
        auc_val = results[name]["Test ROC-AUC"]
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"{name}  (AUC={auc_val:.3f})",
            line=dict(color=col, width=2.5)
        ))
    fig_roc.update_layout(
        **PLOTLY_LAYOUT,
        title="ROC Curves — All Models",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=520
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    # Precision-Recall scatter
    st.markdown("#### Precision vs Recall Trade-off")
    pr_df = pd.DataFrame([
        {"Model": n, "Precision": results[n]["Test Precision"],
         "Recall": results[n]["Test Recall"],
         "F1": results[n]["Test F1"],
         "AUC": results[n]["Test ROC-AUC"]}
        for n in results
    ])
    fig_pr = px.scatter(pr_df, x="Recall", y="Precision", size="F1",
                        color="Model", text="Model",
                        color_discrete_sequence=PALETTE, size_max=35)
    fig_pr.update_traces(textposition="top center", textfont_size=10)
    fig_pr.update_layout(**PLOTLY_LAYOUT, title="Precision–Recall (bubble = F1 score)", height=440)
    st.plotly_chart(fig_pr, use_container_width=True)

    # AUC ranking bar
    auc_df = pr_df.sort_values("AUC")
    fig_auc = px.bar(auc_df, x="AUC", y="Model", orientation="h",
                     color="AUC", color_continuous_scale=["#4c1d95","#7c3aed","#ec4899"],
                     text="AUC")
    fig_auc.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig_auc.update_layout(**PLOTLY_LAYOUT, title="ROC-AUC Ranking", height=360)
    st.plotly_chart(fig_auc, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — PREDICT  (FIXED)
# ═══════════════════════════════════════════════════════════════════════════════
elif "Predict" in nav:
    st.markdown('<p class="section-title">🎯 Predict Package Purchase</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Enter customer details to get an instant, calibrated prediction</p>', unsafe_allow_html=True)

    sel_model_pred = st.selectbox("Choose Model", list(models.keys()))
    st.markdown(f'<div class="model-badge">{sel_model_pred}</div>', unsafe_allow_html=True)

    st.markdown('<div class="predict-card">', unsafe_allow_html=True)

    with st.form("pred_form"):
        st.markdown("##### 👤 Customer Profile")
        c1, c2, c3 = st.columns(3)
        age            = c1.number_input("Age", 18, 65, 35)
        monthly_income = c2.number_input("Monthly Income (₹)", 10000, 100000, 23000, step=1000)
        gender         = c3.selectbox("Gender", ["Male","Female"])

        c4, c5, c6 = st.columns(3)
        occupation     = c4.selectbox("Occupation", ["Salaried","Small Business","Large Business","Free Lancer"])
        designation    = c5.selectbox("Designation", ["Executive","Manager","Senior Manager","AVP","VP"])
        marital_status = c6.selectbox("Marital Status", ["Married","Unmarried","Divorced"])

        st.markdown("##### 🏨 Trip Preferences")
        c7, c8, c9 = st.columns(3)
        city_tier      = c7.selectbox("City Tier", [1,2,3])
        prop_star      = c8.selectbox("Preferred Property Star", [3,4,5])
        num_trips      = c9.selectbox("Number of Trips", [1,2,3,4,5,6,7])

        c10, c11, c12 = st.columns(3)
        num_person     = c10.number_input("Persons Visiting", 1, 8, 2)
        num_children   = c11.number_input("Children Visiting", 0, 5, 0)
        product_pitched = c12.selectbox("Product Pitched", ["Basic","Standard","Deluxe","Super Deluxe","King"])

        st.markdown("##### 📞 Sales Interaction")
        c13, c14, c15 = st.columns(3)
        type_contact   = c13.selectbox("Type of Contact", ["Self Enquiry","Company Invited"])
        duration_pitch = c14.number_input("Duration of Pitch (min)", 5, 60, 15)
        num_followups  = c15.selectbox("Number of Followups", [1,2,3,4,5,6])

        c16, c17, c18 = st.columns(3)
        pitch_score    = c16.slider("Pitch Satisfaction Score (1–5)", 1, 5, 3)
        passport       = c17.selectbox("Has Passport?", [0,1], format_func=lambda x: "Yes" if x else "No")
        own_car        = c18.selectbox("Owns Car?", [0,1], format_func=lambda x: "Yes" if x else "No")

        submitted = st.form_submit_button("🔮 Predict Now", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    if submitted:
        total_visiting = num_person + num_children

        # Build input with EXACT same columns as training (after feature engineering)
        X_template = df.drop(["ProdTaken"], axis=1)

        input_dict = {
            "Age":                  age,
            "TypeofContact":        type_contact,
            "CityTier":             city_tier,
            "DurationOfPitch":      duration_pitch,
            "Occupation":           occupation,
            "Gender":               gender,
            "NumberOfFollowups":    float(num_followups),
            "ProductPitched":       product_pitched,
            "PreferredPropertyStar": float(prop_star),
            "MaritalStatus":        marital_status,
            "NumberOfTrips":        float(num_trips),
            "Passport":             passport,
            "PitchSatisfactionScore": pitch_score,
            "OwnCar":               own_car,
            "Designation":          designation,
            "MonthlyIncome":        monthly_income,
            "TotalVisiting":        float(total_visiting),
        }
        input_df = pd.DataFrame([input_dict])

        # Align columns exactly with training set
        for col in X_template.columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[X_template.columns]

        try:
            X_input = preprocessor.transform(input_df)
            mdl     = models[sel_model_pred]
            pred    = mdl.predict(X_input)[0]
            prob    = mdl.predict_proba(X_input)[0][1] if hasattr(mdl, "predict_proba") else None

            st.markdown("---")
            if pred == 1:
                st.markdown(
                    '<div class="predict-success">✅ This customer is LIKELY to purchase a package</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="predict-fail">❌ This customer is UNLIKELY to purchase a package</div>',
                    unsafe_allow_html=True
                )

            if prob is not None:
                col_l, col_r = st.columns([1,1])
                with col_l:
                    st.metric("Purchase Probability", f"{prob*100:.1f}%",
                              delta=f"{(prob-0.5)*100:+.1f}% vs baseline")
                    st.metric("Non-Purchase Probability", f"{(1-prob)*100:.1f}%")

                with col_r:
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob * 100,
                        gauge={
                            "axis": {"range":[0,100], "tickcolor":"#9d89c4"},
                            "bar":  {"color":"#a855f7"},
                            "bgcolor": "rgba(0,0,0,0)",
                            "bordercolor": "rgba(255,255,255,0.1)",
                            "steps": [
                                {"range":[0,30],  "color":"rgba(124,58,237,0.15)"},
                                {"range":[30,70], "color":"rgba(168,85,247,0.15)"},
                                {"range":[70,100],"color":"rgba(236,72,153,0.15)"},
                            ],
                            "threshold": {"line":{"color":"#ec4899","width":3},
                                          "thickness":0.75, "value":50}
                        },
                        number={"suffix":"%","font":{"color":"#f0e6ff","family":"Syne","size":28}}
                    ))
                    fig_gauge.update_layout(
                        **PLOTLY_LAYOUT,
                        title=f"Purchase Probability · {sel_model_pred}",
                        height=280
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)

                # Probability bar
                fig_prob = go.Figure(go.Bar(
                    x=["Not Purchased","Purchased"],
                    y=[1-prob, prob],
                    marker_color=["#7c3aed","#ec4899"],
                    text=[f"{(1-prob)*100:.1f}%", f"{prob*100:.1f}%"],
                    textposition="outside"
                ))
                fig_prob.update_layout(**PLOTLY_LAYOUT,
                                        title="Class Probability Breakdown",
                                        yaxis=dict(range=[0,1.15], **PLOTLY_LAYOUT["yaxis"]),
                                        height=300)
                st.plotly_chart(fig_prob, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.code(str(e))