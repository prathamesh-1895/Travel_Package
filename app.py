import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
    confusion_matrix
)
from xgboost import XGBClassifier

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Holiday Package Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── GLOBAL STYLES ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1040 50%, #0f0c29 100%);
    color: #e8e0f5;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04);
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 20px;
    backdrop-filter: blur(12px);
}

div[data-testid="metric-container"] label {
    color: #a78bfa !important;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.78rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #f0e6ff !important;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
}

div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 0.8rem;
}

/* Headers */
h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    color: #f0e6ff !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(255,255,255,0.04);
    border-radius: 12px;
    padding: 6px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 20px;
    color: #a78bfa;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    border: none;
    background: transparent;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #7c3aed, #a855f7) !important;
    color: white !important;
}

/* Selectbox, sliders */
.stSelectbox [data-baseweb="select"] > div,
.stMultiSelect [data-baseweb="select"] > div {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(167,139,250,0.3);
    border-radius: 10px;
    color: #f0e6ff;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #a855f7);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 28px;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    letter-spacing: 0.04em;
    transition: all 0.2s;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(124,58,237,0.5);
}

/* Expander */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 10px !important;
    color: #a78bfa !important;
    font-family: 'DM Sans', sans-serif;
}

/* DataFrames */
.stDataFrame {
    border-radius: 12px;
    overflow: hidden;
}

/* Progress bar */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #7c3aed, #a855f7);
}

/* Section divider */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    color: #f0e6ff;
    letter-spacing: -0.02em;
    margin-bottom: 0.2rem;
}
.section-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.9rem;
    color: #9d89c4;
    margin-bottom: 1.5rem;
}
.tag {
    display: inline-block;
    background: rgba(124,58,237,0.2);
    border: 1px solid rgba(124,58,237,0.4);
    color: #c4b5fd;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.78rem;
    font-family: 'DM Sans', sans-serif;
    margin-right: 6px;
}
</style>
""", unsafe_allow_html=True)


# ─── PLOTLY TEMPLATE ─────────────────────────────────────────────────────────
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
    """Load Travel.csv or generate synthetic demo data."""
    import os
    if os.path.exists("Travel.csv"):
        df = pd.read_csv("Travel.csv")
    else:
        # Synthetic demo data (same schema as the Kaggle dataset)
        np.random.seed(42)
        n = 4888
        df = pd.DataFrame({
            "CustomerID": range(1, n+1),
            "ProdTaken": np.random.choice([0, 1], n, p=[0.82, 0.18]),
            "Age": np.random.normal(37, 9, n).clip(18, 65).astype(int),
            "TypeofContact": np.random.choice(["Self Enquiry", "Company Invited"], n, p=[0.7, 0.3]),
            "CityTier": np.random.choice([1, 2, 3], n, p=[0.4, 0.35, 0.25]),
            "DurationOfPitch": np.random.normal(15, 5, n).clip(5, 30).astype(int),
            "Occupation": np.random.choice(["Salaried", "Small Business", "Large Business", "Free Lancer"], n, p=[0.5, 0.25, 0.15, 0.1]),
            "Gender": np.random.choice(["Male", "Female", "Fe Male"], n, p=[0.6, 0.35, 0.05]),
            "NumberOfPersonVisiting": np.random.randint(1, 6, n),
            "NumberOfFollowups": np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], n),
            "ProductPitched": np.random.choice(["Basic", "Standard", "Deluxe", "Super Deluxe", "King"], n),
            "PreferredPropertyStar": np.random.choice([3.0, 4.0, 5.0], n),
            "MaritalStatus": np.random.choice(["Single", "Married", "Divorced", "Unmarried"], n, p=[0.25, 0.5, 0.1, 0.15]),
            "NumberOfTrips": np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], n),
            "Passport": np.random.choice([0, 1], n, p=[0.7, 0.3]),
            "PitchSatisfactionScore": np.random.randint(1, 6, n),
            "OwnCar": np.random.choice([0, 1], n, p=[0.4, 0.6]),
            "NumberOfChildrenVisiting": np.random.choice([0.0, 1.0, 2.0, 3.0], n),
            "Designation": np.random.choice(["Executive", "Manager", "Senior Manager", "AVP", "VP"], n, p=[0.3, 0.3, 0.2, 0.12, 0.08]),
            "MonthlyIncome": np.random.normal(23000, 7000, n).clip(10000, 60000).astype(int)
        })
        # Introduce NaNs
        for col, frac in [("Age", 0.03), ("DurationOfPitch", 0.03), ("NumberOfFollowups", 0.05),
                          ("PreferredPropertyStar", 0.04), ("NumberOfTrips", 0.04),
                          ("NumberOfChildrenVisiting", 0.03), ("MonthlyIncome", 0.04),
                          ("TypeofContact", 0.02)]:
            idx = np.random.choice(df.index, int(n * frac), replace=False)
            df.loc[idx, col] = np.nan

    df_raw = df.copy()

    # Clean
    df["Gender"] = df["Gender"].replace("Fe Male", "Female")
    df["MaritalStatus"] = df["MaritalStatus"].replace("Single", "Unmarried")

    # Impute
    for col in ["Age", "DurationOfPitch", "NumberOfTrips", "MonthlyIncome"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    for col in ["TypeofContact", "NumberOfFollowups", "PreferredPropertyStar", "NumberOfChildrenVisiting"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    df.drop(columns=["CustomerID"], inplace=True, errors="ignore")

    # Feature engineering
    df["TotalVisiting"] = df["NumberOfPersonVisiting"] + df["NumberOfChildrenVisiting"]
    df.drop(columns=["NumberOfPersonVisiting", "NumberOfChildrenVisiting"], inplace=True, errors="ignore")

    return df_raw, df


@st.cache_data
def run_models(df):
    X = df.drop(["ProdTaken"], axis=1)
    y = df["ProdTaken"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    cat_cols = X.select_dtypes(include="object").columns
    num_cols = X.select_dtypes(exclude="object").columns

    preprocessor = ColumnTransformer([
        ("ohe", OneHotEncoder(drop="first", sparse_output=False), cat_cols),
        ("scaler", StandardScaler(), num_cols)
    ])

    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t  = preprocessor.transform(X_test)

    models = {
        "Logistic Regression":  LogisticRegression(max_iter=1000),
        "Decision Tree":        DecisionTreeClassifier(random_state=42),
        "Random Forest":        RandomForestClassifier(n_estimators=100, random_state=42),
        "AdaBoost":             AdaBoostClassifier(random_state=42),
        "Gradient Boosting":    GradientBoostingClassifier(random_state=42),
        "XGBoost":              XGBClassifier(eval_metric="logloss", random_state=42),
    }

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
            "Test ROC-AUC":    round(roc_auc_score(y_test, y_te_pred), 4),
            "Confusion":       confusion_matrix(y_test, y_te_pred),
        }
        if y_te_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_te_prob)
            roc_data[name] = (fpr, tpr)

    # Feature importance from RF
    rf = models["Random Forest"]
    try:
        ohe_cols = preprocessor.named_transformers_["ohe"].get_feature_names_out(cat_cols)
    except Exception:
        ohe_cols = []
    feat_names = list(ohe_cols) + list(num_cols)
    importances = pd.DataFrame({
        "Feature":    feat_names[:len(rf.feature_importances_)],
        "Importance": rf.feature_importances_[:len(feat_names)]
    }).sort_values("Importance", ascending=False).head(15)

    return results, roc_data, importances, X_test, y_test, preprocessor, models, cat_cols, num_cols


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 10px;">
        <div style="font-size:2.4rem;">✈️</div>
        <div style="font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:800; color:#f0e6ff;">
            TravelML
        </div>
        <div style="font-size:0.75rem; color:#9d89c4; margin-top:4px;">
            Holiday Package Predictor
        </div>
    </div>
    <hr style="border-color:rgba(255,255,255,0.08); margin:10px 0 20px;">
    """, unsafe_allow_html=True)

    nav = st.radio(
        "Navigate",
        ["📊 Data Overview", "🔍 EDA", "🤖 Model Comparison", "📈 ROC & Metrics", "🎯 Predict"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color:rgba(255,255,255,0.08); margin:20px 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.72rem; color:#6b5b95; text-align:center;">
        Dataset: Trips & Travel.Com<br>Kaggle Holiday Package Prediction
    </div>
    """, unsafe_allow_html=True)


# ─── LOAD DATA ───────────────────────────────────────────────────────────────
with st.spinner("Loading data & training models…"):
    df_raw, df = load_and_clean()
    results, roc_data, importances, X_test_df, y_test, preprocessor, models, cat_cols, num_cols = run_models(df)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DATA OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if "Data Overview" in nav:
    st.markdown('<p class="section-title">📊 Data Overview</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Holiday Package Purchase Prediction — Trips & Travel.Com</p>', unsafe_allow_html=True)

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

    # Target distribution
    st.markdown("#### Target Variable Distribution")
    tgt = df["ProdTaken"].value_counts().reset_index()
    tgt.columns = ["ProdTaken", "Count"]
    tgt["Label"] = tgt["ProdTaken"].map({0: "Not Purchased", 1: "Purchased"})
    fig_tgt = px.pie(
        tgt, names="Label", values="Count",
        color_discrete_sequence=["#7c3aed", "#a855f7"],
        hole=0.55
    )
    fig_tgt.update_traces(textfont_size=14)
    fig_tgt.update_layout(**PLOTLY_LAYOUT, title="Package Purchase Rate", height=320)
    st.plotly_chart(fig_tgt, use_container_width=True)

    # Missing value heatmap
    st.markdown("#### Missing Values (Raw Data)")
    miss = df_raw.isnull().mean().reset_index()
    miss.columns = ["Feature", "Missing %"]
    miss["Missing %"] = (miss["Missing %"] * 100).round(2)
    miss = miss[miss["Missing %"] > 0].sort_values("Missing %", ascending=False)
    if not miss.empty:
        fig_miss = px.bar(
            miss, x="Feature", y="Missing %",
            color="Missing %", color_continuous_scale=["#4c1d95","#a855f7","#ec4899"],
            text="Missing %"
        )
        fig_miss.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_miss.update_layout(**PLOTLY_LAYOUT, title="Missing Value Percentages", height=320)
        st.plotly_chart(fig_miss, use_container_width=True)

    # Descriptive stats
    with st.expander("📋 Descriptive Statistics"):
        st.dataframe(df.describe().T.round(2), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA
# ═══════════════════════════════════════════════════════════════════════════════
elif "EDA" in nav:
    st.markdown('<p class="section-title">🔍 Exploratory Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Uncover patterns in your travel data</p>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📦 Distributions", "🔗 Correlations", "🎯 Target Insights", "📐 Boxplots"])

    # ── DISTRIBUTIONS
    with tab1:
        num_cols_list = df.select_dtypes(include=np.number).columns.tolist()
        num_cols_list = [c for c in num_cols_list if c != "ProdTaken"]
        col = st.selectbox("Select Numeric Feature", num_cols_list, key="dist_col")
        bins = st.slider("Bins", 10, 80, 30, key="bins")
        fig_h = px.histogram(
            df, x=col, color="ProdTaken",
            barmode="overlay", nbins=bins,
            color_discrete_map={0: "#7c3aed", 1: "#ec4899"},
            opacity=0.75, labels={"ProdTaken": "Purchased"}
        )
        fig_h.update_layout(**PLOTLY_LAYOUT, title=f"Distribution of {col} by Purchase", height=380)
        st.plotly_chart(fig_h, use_container_width=True)

        cat_cols_list = df.select_dtypes(include="object").columns.tolist()
        cat_col = st.selectbox("Select Categorical Feature", cat_cols_list, key="cat_col")
        grp = df.groupby([cat_col, "ProdTaken"]).size().reset_index(name="Count")
        grp["ProdTaken"] = grp["ProdTaken"].map({0: "Not Purchased", 1: "Purchased"})
        fig_bar = px.bar(
            grp, x=cat_col, y="Count", color="ProdTaken", barmode="group",
            color_discrete_map={"Not Purchased": "#7c3aed", "Purchased": "#ec4899"}
        )
        fig_bar.update_layout(**PLOTLY_LAYOUT, title=f"{cat_col} vs Purchase", height=350)
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── CORRELATION
    with tab2:
        corr_df = df.select_dtypes(include=np.number)
        corr_mat = corr_df.corr()
        fig_corr = px.imshow(
            corr_mat, text_auto=".2f", aspect="auto",
            color_continuous_scale=["#1e1245", "#7c3aed", "#ec4899"]
        )
        fig_corr.update_layout(**PLOTLY_LAYOUT, title="Correlation Heatmap", height=520)
        st.plotly_chart(fig_corr, use_container_width=True)

        # Scatter with colour
        c1, c2 = st.columns(2)
        x_ax = c1.selectbox("X-axis", corr_df.columns.tolist(), index=0)
        y_ax = c2.selectbox("Y-axis", corr_df.columns.tolist(), index=1)
        fig_sc = px.scatter(
            df, x=x_ax, y=y_ax, color=df["ProdTaken"].astype(str),
            color_discrete_map={"0": "#7c3aed", "1": "#ec4899"},
            opacity=0.6, labels={"color": "Purchased"},
            trendline="ols"
        )
        fig_sc.update_layout(**PLOTLY_LAYOUT, title=f"{x_ax} vs {y_ax}", height=380)
        st.plotly_chart(fig_sc, use_container_width=True)

    # ── TARGET INSIGHTS
    with tab3:
        cols_for_insight = ["Occupation", "MaritalStatus", "Gender", "ProductPitched", "Designation"]
        cols_for_insight = [c for c in cols_for_insight if c in df.columns]

        for col in cols_for_insight:
            rate = df.groupby(col)["ProdTaken"].mean().reset_index()
            rate.columns = [col, "Purchase Rate"]
            rate = rate.sort_values("Purchase Rate", ascending=False)
            fig_r = px.bar(
                rate, x=col, y="Purchase Rate",
                color="Purchase Rate", color_continuous_scale=["#4c1d95","#a855f7","#ec4899"],
                text="Purchase Rate"
            )
            fig_r.update_traces(texttemplate="%{text:.1%}", textposition="outside")
            fig_r.update_layout(**PLOTLY_LAYOUT, title=f"Purchase Rate by {col}", height=300)
            st.plotly_chart(fig_r, use_container_width=True)

    # ── BOXPLOTS
    with tab4:
        num_feats = [c for c in df.select_dtypes(include=np.number).columns if c != "ProdTaken"]
        box_col = st.selectbox("Feature", num_feats, key="box_col")
        fig_bx = px.box(
            df, x="ProdTaken", y=box_col,
            color="ProdTaken",
            color_discrete_map={0: "#7c3aed", 1: "#ec4899"},
            labels={"ProdTaken": "Purchased"},
            points="outliers"
        )
        fig_bx.update_layout(**PLOTLY_LAYOUT, title=f"{box_col} Distribution by Purchase", height=400)
        st.plotly_chart(fig_bx, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
elif "Model Comparison" in nav:
    st.markdown('<p class="section-title">🤖 Model Comparison</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Six classifiers evaluated on test set performance</p>', unsafe_allow_html=True)

    metric_choice = st.selectbox(
        "Sort by metric",
        ["Test Accuracy", "Test F1", "Test Precision", "Test Recall", "Test ROC-AUC"]
    )

    res_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})
    res_df = res_df.drop(columns=["Confusion"])
    res_df = res_df.sort_values(metric_choice, ascending=False)

    # Bar comparison
    fig_cmp = go.Figure()
    for m, col in zip(
        ["Test Accuracy", "Test F1", "Test Precision", "Test Recall", "Test ROC-AUC"],
        PALETTE
    ):
        fig_cmp.add_trace(go.Bar(
            name=m, x=res_df["Model"], y=res_df[m],
            marker_color=col, opacity=0.85
        ))
    fig_cmp.update_layout(
        **PLOTLY_LAYOUT,
        barmode="group",
        title="All Models — Test Set Metrics",
        height=420,
        xaxis_tickangle=-20
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    # Radar chart for best model
    best_model = res_df.iloc[0]["Model"]
    best = res_df.iloc[0]
    radar_metrics = ["Test Accuracy", "Test F1", "Test Precision", "Test Recall", "Test ROC-AUC"]
    radar_vals = [float(best[m]) for m in radar_metrics]

    fig_radar = go.Figure(go.Scatterpolar(
        r=radar_vals + [radar_vals[0]],
        theta=radar_metrics + [radar_metrics[0]],
        fill="toself",
        fillcolor="rgba(168,85,247,0.2)",
        line=dict(color="#a855f7", width=2),
        name=best_model
    ))
    fig_radar.update_layout(
        **PLOTLY_LAYOUT,
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(255,255,255,0.1)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
            bgcolor="rgba(0,0,0,0)"
        ),
        title=f"Best Model Radar: {best_model}",
        height=400
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Metrics table
    st.markdown("#### 📋 Full Results Table")
    st.dataframe(
        res_df.drop(columns=[], errors="ignore").style.background_gradient(
            subset=["Test Accuracy", "Test F1", "Test ROC-AUC"],
            cmap="Purples"
        ).format({c: "{:.4f}" for c in res_df.columns if c != "Model"}),
        use_container_width=True, height=280
    )

    # Feature importance
    st.markdown("#### 🌲 Random Forest — Feature Importances")
    fig_fi = px.bar(
        importances.sort_values("Importance"),
        x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale=["#4c1d95","#a855f7","#ec4899"]
    )
    fig_fi.update_layout(**PLOTLY_LAYOUT, title="Top 15 Features (Random Forest)", height=460)
    st.plotly_chart(fig_fi, use_container_width=True)

    # Confusion matrices
    st.markdown("#### 🔲 Confusion Matrices")
    sel_model = st.selectbox("Select Model", list(results.keys()), key="cm_sel")
    cm = results[sel_model]["Confusion"]
    fig_cm = px.imshow(
        cm, text_auto=True, aspect="equal",
        color_continuous_scale=["#1e1245", "#7c3aed", "#ec4899"],
        labels=dict(x="Predicted", y="Actual"),
        x=["Not Purchased", "Purchased"],
        y=["Not Purchased", "Purchased"]
    )
    fig_cm.update_layout(**PLOTLY_LAYOUT, title=f"Confusion Matrix — {sel_model}", height=380)
    st.plotly_chart(fig_cm, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ROC & METRICS
# ═══════════════════════════════════════════════════════════════════════════════
elif "ROC" in nav:
    st.markdown('<p class="section-title">📈 ROC Curves & Metrics</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Receiver Operating Characteristic curves for all models</p>', unsafe_allow_html=True)

    # Multi-model ROC
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dash", color="rgba(255,255,255,0.3)"),
        name="Random", showlegend=True
    ))
    for (name, (fpr, tpr)), col in zip(roc_data.items(), PALETTE):
        auc_val = roc_auc_score(y_test, fpr)  # placeholder — use actual AUC stored in results
        auc_val = results[name]["Test ROC-AUC"]
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"{name} (AUC={auc_val:.3f})",
            line=dict(color=col, width=2.5)
        ))
    fig_roc.update_layout(
        **PLOTLY_LAYOUT,
        title="ROC Curves — All Models",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=500
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    # Precision-Recall trade-off
    st.markdown("#### Precision vs Recall Trade-off")
    pr_df = pd.DataFrame([
        {
            "Model": name,
            "Precision": results[name]["Test Precision"],
            "Recall": results[name]["Test Recall"],
            "F1": results[name]["Test F1"],
            "AUC": results[name]["Test ROC-AUC"]
        }
        for name in results
    ])
    fig_pr = px.scatter(
        pr_df, x="Recall", y="Precision", size="F1",
        color="Model", text="Model",
        color_discrete_sequence=PALETTE, size_max=30
    )
    fig_pr.update_traces(textposition="top center", textfont_size=10)
    fig_pr.update_layout(**PLOTLY_LAYOUT, title="Precision–Recall Scatter (bubble = F1)", height=420)
    st.plotly_chart(fig_pr, use_container_width=True)

    # AUC ranking
    auc_df = pr_df.sort_values("AUC", ascending=True)
    fig_auc = px.bar(
        auc_df, x="AUC", y="Model", orientation="h",
        color="AUC", color_continuous_scale=["#4c1d95","#7c3aed","#ec4899"],
        text="AUC"
    )
    fig_auc.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig_auc.update_layout(**PLOTLY_LAYOUT, title="ROC-AUC Ranking", height=350)
    st.plotly_chart(fig_auc, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
elif "Predict" in nav:
    st.markdown('<p class="section-title">🎯 Predict Package Purchase</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Enter customer details and get an instant prediction</p>', unsafe_allow_html=True)

    sel_model_pred = st.selectbox("Choose Model", list(models.keys()))

    with st.form("pred_form"):
        c1, c2, c3 = st.columns(3)
        age             = c1.number_input("Age", 18, 65, 35)
        monthly_income  = c2.number_input("Monthly Income (₹)", 10000, 100000, 23000, step=1000)
        duration_pitch  = c3.number_input("Duration of Pitch (min)", 5, 60, 15)

        c4, c5, c6 = st.columns(3)
        num_followups   = c4.selectbox("Number of Followups", [1,2,3,4,5,6])
        num_trips       = c5.selectbox("Number of Trips", [1,2,3,4,5,6,7])
        city_tier       = c6.selectbox("City Tier", [1,2,3])

        c7, c8, c9 = st.columns(3)
        occupation      = c7.selectbox("Occupation", ["Salaried","Small Business","Large Business","Free Lancer"])
        gender          = c8.selectbox("Gender", ["Male","Female"])
        marital_status  = c9.selectbox("Marital Status", ["Married","Unmarried","Divorced"])

        c10, c11, c12 = st.columns(3)
        product_pitched = c10.selectbox("Product Pitched", ["Basic","Standard","Deluxe","Super Deluxe","King"])
        designation     = c11.selectbox("Designation", ["Executive","Manager","Senior Manager","AVP","VP"])
        type_contact    = c12.selectbox("Type of Contact", ["Self Enquiry","Company Invited"])

        c13, c14, c15 = st.columns(3)
        prop_star       = c13.selectbox("Preferred Property Star", [3,4,5])
        passport        = c14.selectbox("Has Passport?", [0,1], format_func=lambda x: "Yes" if x else "No")
        own_car         = c15.selectbox("Owns Car?", [0,1], format_func=lambda x: "Yes" if x else "No")

        c16, c17 = st.columns(2)
        num_person      = c16.number_input("Number of Persons Visiting", 1, 8, 2)
        num_children    = c17.number_input("Number of Children Visiting", 0, 5, 0)
        pitch_score     = st.slider("Pitch Satisfaction Score (1–5)", 1, 5, 3)

        submitted = st.form_submit_button("🔮 Predict Now")

    if submitted:
        total_visiting = num_person + num_children

        input_dict = {
            "Age": [age],
            "TypeofContact": [type_contact],
            "CityTier": [city_tier],
            "DurationOfPitch": [duration_pitch],
            "Occupation": [occupation],
            "Gender": [gender],
            "NumberOfFollowups": [float(num_followups)],
            "ProductPitched": [product_pitched],
            "PreferredPropertyStar": [float(prop_star)],
            "MaritalStatus": [marital_status],
            "NumberOfTrips": [float(num_trips)],
            "Passport": [passport],
            "PitchSatisfactionScore": [pitch_score],
            "OwnCar": [own_car],
            "Designation": [designation],
            "MonthlyIncome": [monthly_income],
            "TotalVisiting": [float(total_visiting)],
        }
        input_df = pd.DataFrame(input_dict)

        # Align columns with training
        X_template = df.drop(["ProdTaken"], axis=1)
        for col in X_template.columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[X_template.columns]

        try:
            X_input = preprocessor.transform(input_df)
            mdl = models[sel_model_pred]
            pred = mdl.predict(X_input)[0]
            prob = mdl.predict_proba(X_input)[0][1] if hasattr(mdl, "predict_proba") else None

            if pred == 1:
                st.success("✅ **This customer is LIKELY to purchase the package!**")
            else:
                st.warning("❌ **This customer is UNLIKELY to purchase the package.**")

            if prob is not None:
                st.markdown(f"**Purchase Probability: `{prob*100:.1f}%`**")
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=prob * 100,
                    delta={"reference": 50, "valueformat": ".1f"},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#9d89c4"},
                        "bar": {"color": "#a855f7"},
                        "bgcolor": "rgba(0,0,0,0)",
                        "bordercolor": "rgba(255,255,255,0.1)",
                        "steps": [
                            {"range": [0, 30], "color": "rgba(124,58,237,0.2)"},
                            {"range": [30, 70], "color": "rgba(168,85,247,0.2)"},
                            {"range": [70, 100], "color": "rgba(236,72,153,0.2)"},
                        ],
                        "threshold": {"line": {"color": "#ec4899", "width": 3}, "thickness": 0.75, "value": 50}
                    },
                    number={"suffix": "%", "font": {"color": "#f0e6ff", "family": "Syne"}}
                ))
                fig_gauge.update_layout(
                    **PLOTLY_LAYOUT,
                    title=f"Purchase Probability — {sel_model_pred}",
                    height=300
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")
