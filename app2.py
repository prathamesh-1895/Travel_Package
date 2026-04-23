import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Holiday Package Prediction",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background: #0f1117; }
    .block-container { padding: 2rem 2.5rem; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1d2e 0%, #12141f 100%);
        border-right: 1px solid #2d3047;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e2139 0%, #262a45 100%);
        border: 1px solid #3d4266;
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        transition: transform .2s, box-shadow .2s;
    }
    .metric-card:hover { transform: translateY(-3px); box-shadow: 0 8px 30px rgba(0,0,0,0.5); }
    .metric-title { color: #8b92c4; font-size: 0.78rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; margin-bottom: .4rem; }
    .metric-value { font-size: 2rem; font-weight: 700; }
    .metric-sub { color: #6b7280; font-size: 0.72rem; margin-top: .3rem; }

    /* Section headers */
    .section-header {
        font-size: 1.35rem; font-weight: 700; color: #e8eaff;
        border-left: 4px solid #6366f1;
        padding-left: .85rem; margin: 1.8rem 0 1.2rem;
    }

    /* Badge */
    .badge {
        display: inline-block; border-radius: 20px; padding: .25rem .75rem;
        font-size: .72rem; font-weight: 600; letter-spacing: .05em;
    }
    .badge-green  { background: #14532d; color: #4ade80; }
    .badge-blue   { background: #1e3a5f; color: #60a5fa; }
    .badge-purple { background: #3b1d6e; color: #c084fc; }
    .badge-orange { background: #7c2d12; color: #fb923c; }
    .badge-yellow { background: #713f12; color: #fbbf24; }

    /* Tab override */
    div[data-testid="stTabs"] [role="tab"] { font-weight: 600; font-size: .9rem; color: #8b92c4; }
    div[data-testid="stTabs"] [role="tab"][aria-selected="true"] { color: #818cf8; border-bottom-color: #6366f1 !important; }

    /* Divider */
    hr { border-color: #2d3047; }

    /* Prediction box */
    .pred-box {
        border-radius: 16px; padding: 1.6rem;
        text-align: center; margin-top: .5rem;
    }
    .pred-yes { background: linear-gradient(135deg, #14532d, #166534); border: 1px solid #22c55e; }
    .pred-no  { background: linear-gradient(135deg, #450a0a, #7f1d1d); border: 1px solid #ef4444; }
    .pred-title { font-size: 1.4rem; font-weight: 700; margin-bottom: .4rem; }
    .pred-prob  { font-size: 2.8rem; font-weight: 800; }
    .pred-sub   { font-size: .8rem; color: #9ca3af; margin-top: .4rem; }
</style>
""", unsafe_allow_html=True)

# ── Synthetic dataset (matches notebook's Travel.csv schema) ───────────────────
@st.cache_data
def generate_data(n=4888, seed=42):
    rng = np.random.default_rng(seed)
    age        = rng.integers(18, 65, n)
    income     = rng.normal(50000, 18000, n).clip(15000, 120000)
    trips      = rng.integers(1, 9, n)
    duration   = rng.integers(1, 14, n)
    pitch      = rng.integers(2, 12, n)
    followup   = rng.integers(1, 6, n)
    persons    = rng.integers(1, 6, n)
    children   = rng.choice([0,1,2,3], n, p=[.45,.30,.17,.08])
    passport   = rng.choice([0,1], n, p=[.71,.29])
    occupation = rng.choice(['Salaried','Self Employed','Small Business','Large Business','Free Lancer'], n)
    gender     = rng.choice(['Male','Female'], n, p=[.60,.40])
    marital    = rng.choice(['Married','Unmarried','Divorced'], n, p=[.50,.38,.12])
    designation= rng.choice(['Executive','Manager','Senior Manager','AVP','VP'], n)
    prod       = rng.choice(['Basic','Standard','Deluxe','Super Deluxe','King'], n)
    city_tier  = rng.choice([1,2,3], n, p=[.37,.33,.30])

    income_per_trip = income / trips
    proba = (
        0.05
        + 0.12 * (income > 60000)
        + 0.10 * (passport == 1)
        + 0.08 * (trips >= 5)
        + 0.06 * (age < 35)
        + 0.05 * (children > 0)
        - 0.04 * (marital == 'Divorced')
        + 0.03 * (pitch > 7)
        + rng.normal(0, 0.06, n)
    ).clip(0, 1)

    target = (rng.random(n) < proba).astype(int)

    df = pd.DataFrame({
        'Age': age, 'MonthlyIncome': income.astype(int),
        'NumberOfTrips': trips, 'DurationOfPitch': duration,
        'PitchSatisfactionScore': pitch, 'NumberOfFollowups': followup,
        'TotalVisiting': persons + children,
        'Passport': passport, 'Occupation': occupation,
        'Gender': gender, 'MaritalStatus': marital,
        'Designation': designation, 'ProductPitched': prod,
        'CityTier': city_tier, 'ProdTaken': target
    })
    df['IncomePerTrip'] = (income / trips).astype(int)
    df['AgeGroup'] = pd.cut(age, bins=[0,25,35,45,60,100], labels=['<25','25-35','35-45','45-60','60+'])
    return df

@st.cache_resource
def train_models(df):
    X = df.drop('ProdTaken', axis=1)
    y = df['ProdTaken']

    cat_cols = X.select_dtypes(include=['object','category']).columns.tolist()
    num_cols = X.select_dtypes(exclude=['object','category']).columns.tolist()

    num_pipe = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())])
    cat_pipe = Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                         ('enc', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))])
    preprocessor = ColumnTransformer([('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)])

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    X_tr_p = preprocessor.fit_transform(X_tr)
    X_te_p  = preprocessor.transform(X_te)

    smote = SMOTE(random_state=42)
    X_tr_sm, y_tr_sm = smote.fit_resample(X_tr_p, y_tr)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'Decision Tree':        DecisionTreeClassifier(class_weight='balanced', random_state=42),
        'Random Forest':        RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1),
        'Gradient Boosting':    GradientBoostingClassifier(n_estimators=150, random_state=42),
        'XGBoost':              XGBClassifier(n_estimators=150, scale_pos_weight=4, use_label_encoder=False,
                                              eval_metric='logloss', random_state=42, n_jobs=-1),
    }

    results, trained = [], {}
    for name, m in models.items():
        m.fit(X_tr_sm, y_tr_sm)
        yp = m.predict(X_te_p)
        ypr = m.predict_proba(X_te_p)[:,1]
        results.append({'Model':name,
            'Accuracy':  round(accuracy_score(y_te,yp),4),
            'Precision': round(precision_score(y_te,yp,zero_division=0),4),
            'Recall':    round(recall_score(y_te,yp,zero_division=0),4),
            'F1':        round(f1_score(y_te,yp,zero_division=0),4),
            'ROC-AUC':   round(roc_auc_score(y_te,ypr),4)})
        trained[name] = (m, yp, ypr)

    best_name = max(results, key=lambda r: r['ROC-AUC'])['Model']
    best_m, best_yp, best_ypr = trained[best_name]

    precisions, recalls, thresholds = precision_recall_curve(y_te, best_ypr)
    f1s = 2*precisions*recalls/(precisions+recalls+1e-9)
    opt_thr = thresholds[np.argmax(f1s)]

    try:
        cat_names = preprocessor.named_transformers_['cat']['enc'].get_feature_names_out(cat_cols).tolist()
    except:
        cat_names = [f'cat_{i}' for i in range(len(best_m.feature_importances_)-len(num_cols))]
    feat_names = num_cols + cat_names

    return {
        'results_df':   pd.DataFrame(results).set_index('Model'),
        'trained':      trained,
        'preprocessor': preprocessor,
        'best_name':    best_name,
        'best_ypr':     best_ypr,
        'best_yp':      best_yp,
        'y_te':         y_te,
        'X_te':         X_te,
        'opt_thr':      opt_thr,
        'feat_names':   feat_names,
        'num_cols':     num_cols,
        'cat_cols':     cat_cols,
        'X_tr_sm':      X_tr_sm,
        'y_tr_sm':      y_tr_sm,
    }

# ── Load data & train ──────────────────────────────────────────────────────────
df = generate_data()
mdata = train_models(df)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌴 Holiday Predictor")
    st.markdown("**Trips & Travel.Com**")
    st.markdown("*Wellness Tourism Package*")
    st.divider()
    page = st.radio("Navigate",
        ["📊 Dashboard", "🔍 Exploratory Analysis", "🤖 Model Comparison",
         "📈 Model Evaluation", "🎯 Predict Customer"],
        label_visibility="collapsed")
    st.divider()
    st.markdown('<span class="badge badge-blue">4,888 Customers</span>&nbsp;<span class="badge badge-orange">~18% Purchased</span>', unsafe_allow_html=True)
    st.caption("Random Forest · SMOTE · Threshold Tuning")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — Dashboard
# ─────────────────────────────────────────────────────────────────────────────
if page == "📊 Dashboard":
    st.markdown("# 🌴 Holiday Package Prediction Dashboard")
    st.markdown("Real-time insights on customer purchase behaviour for the **Wellness Tourism Package**.")

    purchased = df['ProdTaken'].sum()
    not_purch  = len(df) - purchased
    best_auc   = mdata['results_df']['ROC-AUC'].max()
    best_f1    = mdata['results_df']['F1'].max()

    c1,c2,c3,c4,c5 = st.columns(5)
    for col, title, val, sub, color in [
        (c1, "Total Customers",  f"{len(df):,}", "In dataset", "#60a5fa"),
        (c2, "Purchased",        f"{purchased:,}", f"{purchased/len(df)*100:.1f}% of total", "#4ade80"),
        (c3, "Not Purchased",    f"{not_purch:,}", f"{not_purch/len(df)*100:.1f}% of total", "#f87171"),
        (c4, "Best ROC-AUC",     f"{best_auc:.3f}", "Random Forest", "#a78bfa"),
        (c5, "Best F1 Score",    f"{best_f1:.3f}", "Threshold-tuned", "#fbbf24"),
    ]:
        col.markdown(f"""<div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value" style="color:{color}">{val}</div>
            <div class="metric-sub">{sub}</div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Customer Demographics at a Glance</div>', unsafe_allow_html=True)
    r1c1, r1c2, r1c3 = st.columns(3)

    with r1c1:
        fig = px.pie(df, names='Gender', title='Gender Distribution',
                     color_discrete_sequence=['#6366f1','#f472b6'],
                     hole=.45)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color='#cdd0e8', title_font_size=14, margin=dict(t=40,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with r1c2:
        fig = px.pie(df, names='MaritalStatus', title='Marital Status',
                     color_discrete_sequence=['#22d3ee','#f59e0b','#f87171'],
                     hole=.45)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color='#cdd0e8', title_font_size=14, margin=dict(t=40,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with r1c3:
        occ = df['Occupation'].value_counts().reset_index()
        occ.columns = ['Occupation','Count']
        fig = px.bar(occ, x='Count', y='Occupation', orientation='h',
                     title='Occupation Breakdown',
                     color='Count', color_continuous_scale='Viridis')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color='#cdd0e8', title_font_size=14,
                          coloraxis_showscale=False, margin=dict(t=40,b=10),
                          yaxis_title='', xaxis_title='Count')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Purchase Rate by Segment</div>', unsafe_allow_html=True)
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        seg = df.groupby('AgeGroup')['ProdTaken'].mean().reset_index()
        seg.columns = ['Age Group','Purchase Rate']
        seg['Purchase Rate'] = (seg['Purchase Rate']*100).round(1)
        fig = px.bar(seg, x='Age Group', y='Purchase Rate',
                     title='Purchase Rate by Age Group (%)',
                     color='Purchase Rate', color_continuous_scale='Plasma',
                     text='Purchase Rate')
        fig.update_traces(texttemplate='%{text}%', textposition='outside')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color='#cdd0e8', title_font_size=14,
                          coloraxis_showscale=False, margin=dict(t=40,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with r2c2:
        seg2 = df.groupby('Occupation')['ProdTaken'].mean().reset_index()
        seg2.columns = ['Occupation','Purchase Rate']
        seg2['Purchase Rate'] = (seg2['Purchase Rate']*100).round(1)
        seg2 = seg2.sort_values('Purchase Rate', ascending=True)
        fig = px.bar(seg2, x='Purchase Rate', y='Occupation', orientation='h',
                     title='Purchase Rate by Occupation (%)',
                     color='Purchase Rate', color_continuous_scale='Turbo',
                     text='Purchase Rate')
        fig.update_traces(texttemplate='%{text}%', textposition='outside')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color='#cdd0e8', title_font_size=14,
                          coloraxis_showscale=False, margin=dict(t=40,b=10),
                          yaxis_title='')
        st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — EDA
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔍 Exploratory Analysis":
    st.markdown("# 🔍 Exploratory Data Analysis")

    tab1, tab2, tab3, tab4 = st.tabs(["📦 Distributions", "🔗 Correlations", "💡 Key Insights", "📋 Raw Data"])

    with tab1:
        col_sel = st.selectbox("Select numerical feature", ['Age','MonthlyIncome','NumberOfTrips','IncomePerTrip',
                                                             'PitchSatisfactionScore','NumberOfFollowups'])
        fc1, fc2 = st.columns(2)
        with fc1:
            fig = px.histogram(df, x=col_sel, color='ProdTaken',
                               barmode='overlay', nbins=35,
                               color_discrete_map={0:'#6366f1',1:'#22c55e'},
                               title=f'{col_sel} Distribution by Purchase',
                               labels={'ProdTaken':'Purchased'})
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              font_color='#cdd0e8', title_font_size=14,
                              legend=dict(bgcolor='rgba(0,0,0,0)'))
            st.plotly_chart(fig, use_container_width=True)
        with fc2:
            fig = px.box(df, x='ProdTaken', y=col_sel,
                         color='ProdTaken',
                         color_discrete_map={0:'#6366f1',1:'#22c55e'},
                         title=f'{col_sel} Boxplot by Purchase',
                         labels={'ProdTaken':'Purchased (0=No, 1=Yes)'})
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              font_color='#cdd0e8', title_font_size=14,
                              showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        cat_sel = st.selectbox("Select categorical feature", ['Gender','MaritalStatus','Occupation',
                                                               'Designation','ProductPitched','AgeGroup'])
        ct = df.groupby([cat_sel, 'ProdTaken']).size().reset_index(name='Count')
        ct['ProdTaken'] = ct['ProdTaken'].map({0:'Not Purchased', 1:'Purchased'})
        fig = px.bar(ct, x=cat_sel, y='Count', color='ProdTaken',
                     barmode='group',
                     color_discrete_map={'Not Purchased':'#6366f1','Purchased':'#22c55e'},
                     title=f'Purchase Distribution by {cat_sel}')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color='#cdd0e8', title_font_size=14,
                          legend=dict(bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        num_df = df.select_dtypes(include='number').drop(columns=['ProdTaken'])
        corr = num_df.corr()
        fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r',
                        title='Correlation Heatmap — Numerical Features',
                        zmin=-1, zmax=1)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color='#cdd0e8', title_font_size=15,
                          margin=dict(t=50,b=20))
        st.plotly_chart(fig, use_container_width=True)

        sc1, sc2 = st.columns(2)
        sc_x = sc1.selectbox("X axis", ['MonthlyIncome','Age','NumberOfTrips','IncomePerTrip'], key='sx')
        sc_y = sc2.selectbox("Y axis", ['IncomePerTrip','Age','MonthlyIncome','NumberOfTrips'], index=1, key='sy')
        fig = px.scatter(df.sample(800, random_state=1), x=sc_x, y=sc_y,
                         color=df.loc[df.sample(800, random_state=1).index,'ProdTaken'].map({0:'No',1:'Yes'}),
                         color_discrete_map={'No':'#6366f1','Yes':'#22c55e'},
                         opacity=.65, title=f'{sc_x} vs {sc_y}')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color='#cdd0e8', legend=dict(bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        ins1, ins2 = st.columns(2)
        with ins1:
            pp = df.groupby('Passport')['ProdTaken'].mean().reset_index()
            pp['Passport'] = pp['Passport'].map({0:'No Passport',1:'Has Passport'})
            pp['Rate'] = (pp['ProdTaken']*100).round(1)
            fig = px.bar(pp, x='Passport', y='Rate', color='Rate',
                         title='Passport Holders: Purchase Rate (%)',
                         color_continuous_scale='Greens', text='Rate')
            fig.update_traces(texttemplate='%{text}%', textposition='outside')
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              font_color='#cdd0e8', title_font_size=13,
                              coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        with ins2:
            ct2 = df.groupby('CityTier')['ProdTaken'].mean().reset_index()
            ct2['Rate'] = (ct2['ProdTaken']*100).round(1)
            ct2['CityTier'] = ct2['CityTier'].astype(str)
            fig = px.bar(ct2, x='CityTier', y='Rate', color='Rate',
                         title='City Tier vs Purchase Rate (%)',
                         color_continuous_scale='Blues', text='Rate')
            fig.update_traces(texttemplate='%{text}%', textposition='outside')
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              font_color='#cdd0e8', title_font_size=13,
                              coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        # Income distribution
        fig = px.violin(df, x='AgeGroup', y='MonthlyIncome', color='ProdTaken',
                        color_discrete_map={0:'#6366f1',1:'#22c55e'},
                        box=True, title='Monthly Income Distribution by Age Group & Purchase',
                        labels={'ProdTaken':'Purchased'})
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color='#cdd0e8', title_font_size=14,
                          legend=dict(bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.dataframe(df.head(100).style.background_gradient(cmap='Blues', subset=['MonthlyIncome','IncomePerTrip']),
                     use_container_width=True, height=420)
        st.caption("Showing first 100 rows of 4,888 total records.")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — Model Comparison
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🤖 Model Comparison":
    st.markdown("# 🤖 Model Comparison")
    st.markdown("Five models trained on **SMOTE-balanced** data, evaluated on the held-out test set.")

    res = mdata['results_df'].reset_index()
    best = mdata['best_name']

    # Radar chart
    metrics = ['Accuracy','Precision','Recall','F1','ROC-AUC']
    colors  = ['#6366f1','#22c55e','#f59e0b','#f87171','#22d3ee']
    fig = go.Figure()
    for i, row in res.iterrows():
        vals = [row[m] for m in metrics] + [row[metrics[0]]]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=metrics+[metrics[0]],
            name=row['Model'], fill='toself', opacity=.6,
            line=dict(color=colors[i], width=2)
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,1], color='#8b92c4'),
                   bgcolor='rgba(0,0,0,0)'),
        paper_bgcolor='rgba(0,0,0,0)', font_color='#cdd0e8',
        title='Model Performance Radar', title_font_size=15,
        legend=dict(bgcolor='rgba(30,33,57,0.9)', bordercolor='#3d4266'),
        height=420
    )
    st.plotly_chart(fig, use_container_width=True)

    # Bar comparison
    res_melt = res.melt(id_vars='Model', value_vars=metrics, var_name='Metric', value_name='Score')
    fig2 = px.bar(res_melt, x='Model', y='Score', color='Metric',
                  barmode='group', title='Detailed Metric Comparison',
                  color_discrete_sequence=['#6366f1','#22d3ee','#22c55e','#fbbf24','#f87171'])
    fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                       font_color='#cdd0e8', title_font_size=15,
                       legend=dict(bgcolor='rgba(0,0,0,0)'),
                       xaxis_tickangle=-20, yaxis_range=[0,1.05])
    st.plotly_chart(fig2, use_container_width=True)

    # Table
    st.markdown('<div class="section-header">Score Summary Table</div>', unsafe_allow_html=True)
    def highlight_best(s):
        return ['background-color: #1a3a2a; color: #4ade80; font-weight:700'
                if v == s.max() else '' for v in s]
    styled = res.set_index('Model')[metrics].style\
        .apply(highlight_best)\
        .format("{:.4f}")
    st.dataframe(styled, use_container_width=True)

    st.success(f"🏆 **Best Model: {best}** — ROC-AUC = {mdata['results_df'].loc[best,'ROC-AUC']:.4f}")

    # Feature importance
    best_m = mdata['trained'][best][0]
    if hasattr(best_m, 'feature_importances_'):
        fi = pd.DataFrame({'Feature': mdata['feat_names'],
                           'Importance': best_m.feature_importances_})\
               .sort_values('Importance', ascending=False).head(20)
        fig3 = px.bar(fi, x='Importance', y='Feature', orientation='h',
                      title=f'Top 20 Feature Importances — {best}',
                      color='Importance', color_continuous_scale='Viridis')
        fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           font_color='#cdd0e8', title_font_size=14,
                           coloraxis_showscale=False,
                           yaxis=dict(autorange='reversed'))
        st.plotly_chart(fig3, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — Model Evaluation
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📈 Model Evaluation":
    st.markdown("# 📈 Model Evaluation")

    best   = mdata['best_name']
    y_te   = mdata['y_te']
    y_proba= mdata['best_ypr']
    y_pred = mdata['best_yp']
    opt_thr= mdata['opt_thr']

    m1,m2 = st.columns(2)

    # ROC curve
    with m1:
        fpr, tpr, _ = roc_curve(y_te, y_proba)
        auc = roc_auc_score(y_te, y_proba)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                  name=f'RF (AUC={auc:.3f})',
                                  line=dict(color='#6366f1', width=2.5),
                                  fill='tozeroy', fillcolor='rgba(99,102,241,0.15)'))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                                  name='Random', line=dict(color='#6b7280', dash='dash', width=1.5)))
        fig.update_layout(title='ROC-AUC Curve', xaxis_title='False Positive Rate',
                           yaxis_title='True Positive Rate',
                           paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           font_color='#cdd0e8', title_font_size=14,
                           legend=dict(bgcolor='rgba(0,0,0,0)'), height=350)
        st.plotly_chart(fig, use_container_width=True)

    # Precision-Recall
    with m2:
        precs, recs, thrs = precision_recall_curve(y_te, y_proba)
        f1s = 2*precs*recs/(precs+recs+1e-9)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=thrs, y=precs[:-1], name='Precision', line=dict(color='#22d3ee',width=2)))
        fig.add_trace(go.Scatter(x=thrs, y=recs[:-1],  name='Recall',    line=dict(color='#f59e0b',width=2)))
        fig.add_trace(go.Scatter(x=thrs, y=f1s[:-1],   name='F1',        line=dict(color='#22c55e',width=2,dash='dash')))
        fig.add_vline(x=opt_thr, line=dict(color='#f87171', dash='dot', width=1.5),
                      annotation_text=f"Optimal={opt_thr:.2f}", annotation_font_color='#f87171')
        fig.update_layout(title='Precision / Recall / F1 vs Threshold',
                           xaxis_title='Threshold', yaxis_title='Score',
                           paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           font_color='#cdd0e8', title_font_size=14,
                           legend=dict(bgcolor='rgba(0,0,0,0)'), height=350)
        st.plotly_chart(fig, use_container_width=True)

    # Confusion matrices side by side
    st.markdown('<div class="section-header">Confusion Matrix — Default vs Tuned Threshold</div>', unsafe_allow_html=True)
    y_tuned = (y_proba >= opt_thr).astype(int)

    cf1, cf2 = st.columns(2)
    for col, yp, title, cs in [
        (cf1, y_pred,  'Default Threshold (0.50)', 'Blues'),
        (cf2, y_tuned, f'Tuned Threshold ({opt_thr:.2f})', 'Greens')
    ]:
        cm = confusion_matrix(y_te, yp)
        fig = px.imshow(cm, text_auto=True, color_continuous_scale=cs,
                        x=['Pred: No','Pred: Yes'], y=['Act: No','Act: Yes'],
                        title=title, aspect='auto')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color='#cdd0e8', title_font_size=13,
                          coloraxis_showscale=False, height=300)
        col.plotly_chart(fig, use_container_width=True)

    # Score delta table
    st.markdown('<div class="section-header">Metrics: Before & After Threshold Tuning</div>', unsafe_allow_html=True)
    delta = pd.DataFrame({
        'Metric': ['Accuracy','Precision','Recall','F1'],
        'Default (0.50)': [accuracy_score(y_te,y_pred), precision_score(y_te,y_pred,zero_division=0),
                           recall_score(y_te,y_pred,zero_division=0), f1_score(y_te,y_pred,zero_division=0)],
        f'Tuned ({opt_thr:.2f})': [accuracy_score(y_te,y_tuned), precision_score(y_te,y_tuned,zero_division=0),
                                   recall_score(y_te,y_tuned,zero_division=0), f1_score(y_te,y_tuned,zero_division=0)],
    })
    delta['Δ Change'] = delta[f'Tuned ({opt_thr:.2f})'] - delta['Default (0.50)']
    st.dataframe(delta.style.format({'Default (0.50)':'{:.4f}', f'Tuned ({opt_thr:.2f})':'{:.4f}', 'Δ Change':'{:+.4f}'})\
                 .applymap(lambda v: 'color:#4ade80' if isinstance(v,float) and v>0
                           else ('color:#f87171' if isinstance(v,float) and v<0 else ''),
                           subset=['Δ Change']),
                 use_container_width=True)

    # Probability distribution
    st.markdown('<div class="section-header">Predicted Probability Distribution</div>', unsafe_allow_html=True)
    prob_df = pd.DataFrame({'Probability': y_proba, 'Actual': y_te.values})
    fig = px.histogram(prob_df, x='Probability', color='Actual',
                       color_discrete_map={0:'#6366f1',1:'#22c55e'},
                       nbins=40, barmode='overlay', opacity=.72,
                       title='Distribution of Predicted Probabilities')
    fig.add_vline(x=opt_thr, line=dict(color='#f87171', dash='dash'),
                  annotation_text=f'Threshold={opt_thr:.2f}', annotation_font_color='#f87171')
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font_color='#cdd0e8', legend=dict(bgcolor='rgba(0,0,0,0)'))
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 — Predict Customer
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🎯 Predict Customer":
    st.markdown("# 🎯 Predict Customer Purchase")
    st.markdown("Fill in a customer's profile below to get a real-time purchase probability.")

    with st.form("predict_form"):
        st.markdown("#### 👤 Customer Details")
        c1, c2, c3 = st.columns(3)
        age     = c1.slider("Age", 18, 65, 32)
        income  = c2.number_input("Monthly Income (₹)", 15000, 120000, 55000, step=1000)
        trips   = c3.slider("Number of Past Trips", 1, 8, 3)

        c4, c5, c6 = st.columns(3)
        duration= c4.slider("Pitch Duration (min)", 1, 14, 6)
        pitch_s = c5.slider("Pitch Satisfaction (1-10)", 1, 10, 6)
        followup= c6.slider("Number of Follow-ups", 1, 6, 3)

        c7, c8, c9 = st.columns(3)
        total_v = c7.slider("Total Persons Visiting", 1, 8, 3)
        passport= c8.selectbox("Passport", ["No Passport (0)", "Has Passport (1)"])
        city_t  = c9.selectbox("City Tier", [1, 2, 3])

        c10, c11, c12 = st.columns(3)
        occupation = c10.selectbox("Occupation", ['Salaried','Self Employed','Small Business','Large Business','Free Lancer'])
        gender     = c11.selectbox("Gender", ['Male','Female'])
        marital    = c12.selectbox("Marital Status", ['Married','Unmarried','Divorced'])

        c13, c14 = st.columns(2)
        designation = c13.selectbox("Designation", ['Executive','Manager','Senior Manager','AVP','VP'])
        product_p   = c14.selectbox("Product Pitched", ['Basic','Standard','Deluxe','Super Deluxe','King'])

        submitted = st.form_submit_button("🔮 Predict", use_container_width=True, type="primary")

    if submitted:
        passport_val = 1 if "1" in passport else 0
        age_grp = '<25' if age<25 else '25-35' if age<35 else '35-45' if age<45 else '45-60' if age<60 else '60+'
        income_per_trip = int(income / trips)

        new_cust = pd.DataFrame([{
            'Age': age, 'MonthlyIncome': income, 'NumberOfTrips': trips,
            'DurationOfPitch': duration, 'PitchSatisfactionScore': pitch_s,
            'NumberOfFollowups': followup, 'TotalVisiting': total_v,
            'Passport': passport_val, 'CityTier': city_t,
            'Occupation': occupation, 'Gender': gender,
            'MaritalStatus': marital, 'Designation': designation,
            'ProductPitched': product_p,
            'IncomePerTrip': income_per_trip,
            'AgeGroup': age_grp
        }])

        best_m = mdata['trained'][mdata['best_name']][0]
        prep   = mdata['preprocessor']
        thr    = mdata['opt_thr']

        new_proc = prep.transform(new_cust)
        prob = float(best_m.predict_proba(new_proc)[0,1])
        pred = int(prob >= thr)

        pc1, pc2, pc3 = st.columns([1,2,1])
        with pc2:
            if pred == 1:
                st.markdown(f"""<div class="pred-box pred-yes">
                    <div class="pred-title">✅ Likely to Purchase</div>
                    <div class="pred-prob" style="color:#4ade80">{prob*100:.1f}%</div>
                    <div class="pred-sub">Purchase probability · Threshold = {thr:.2f}</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="pred-box pred-no">
                    <div class="pred-title">❌ Unlikely to Purchase</div>
                    <div class="pred-prob" style="color:#f87171">{prob*100:.1f}%</div>
                    <div class="pred-sub">Purchase probability · Threshold = {thr:.2f}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 📊 Probability Gauge")
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob*100,
            domain={'x':[0,1],'y':[0,1]},
            title={'text':'Purchase Probability (%)','font':{'color':'#cdd0e8','size':16}},
            delta={'reference': thr*100, 'valueformat':'.1f',
                   'increasing':{'color':'#22c55e'},'decreasing':{'color':'#f87171'}},
            number={'suffix':'%','font':{'color':'#e8eaff','size':36}},
            gauge={
                'axis':{'range':[0,100],'tickcolor':'#6b7280','tickfont':{'color':'#6b7280'}},
                'bar':{'color':'#6366f1','thickness':.35},
                'bgcolor':'#1e2139',
                'bordercolor':'#3d4266',
                'steps':[
                    {'range':[0, thr*100],'color':'rgba(248,113,113,0.25)'},
                    {'range':[thr*100,100],'color':'rgba(34,197,94,0.25)'}
                ],
                'threshold':{'line':{'color':'#f59e0b','width':3},'thickness':.85,'value':thr*100}
            }
        ))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=280, margin=dict(t=30,b=10))
        st.plotly_chart(fig, use_container_width=True)

        # Key factors
        st.markdown("#### 💡 Key Factors for This Customer")
        factors = []
        if income > 60000: factors.append(("💰 High Income", "Significantly above average", "#22c55e"))
        if passport_val:   factors.append(("🛂 Has Passport", "Higher propensity to travel", "#22c55e"))
        if trips >= 5:     factors.append(("✈️ Frequent Traveller", f"{trips} past trips", "#22c55e"))
        if age < 35:       factors.append(("🧑 Young Demographic", "Higher purchase rate group", "#22c55e"))
        if pitch_s >= 8:   factors.append(("⭐ High Pitch Score", f"Satisfaction: {pitch_s}/10", "#22c55e"))
        if income <= 40000:factors.append(("📉 Lower Income", "Below purchasing threshold", "#f87171"))
        if not passport_val:factors.append(("🚫 No Passport", "Lower travel intent signal", "#f87171"))
        if trips < 2:      factors.append(("🔁 Infrequent Traveller", "Limited trip history", "#f87171"))

        if factors:
            cols = st.columns(min(len(factors), 3))
            for i, (title, desc, color) in enumerate(factors):
                cols[i%3].markdown(f"""<div class="metric-card" style="text-align:left;">
                    <div style="font-size:.95rem;font-weight:700;color:{color}">{title}</div>
                    <div style="font-size:.78rem;color:#8b92c4;margin-top:.3rem">{desc}</div>
                </div>""", unsafe_allow_html=True)