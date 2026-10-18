"""
QuickLoan · The Great Model Bake-Off — Champion Dashboard
Run:  streamlit run app.py      (keep loans.csv next to this file, or upload it in the sidebar)

The app re-creates the notebook's pipeline exactly (same split, seeds, models, OOF threshold
tuning on TRAIN only) and caches the result, so no joblib file is required.
"""
import os, io, hashlib, warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import joblib

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (VotingClassifier, RandomForestClassifier, BaggingClassifier,
                              AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier)
from sklearn.metrics import (accuracy_score, recall_score, precision_score, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.inspection import permutation_importance

try:
    from xgboost import XGBClassifier
except ImportError:  # same fallback as the notebook
    from sklearn.ensemble import HistGradientBoostingClassifier as XGBClassifier

SEED, INTEREST = 42, 0.20
CUR = "AED"

# ───────────────────────────── Page + theme ─────────────────────────────
st.set_page_config(page_title="QuickLoan · Champion Model", page_icon="🏆", layout="wide",
                   initial_sidebar_state="expanded")

INK, MUTED = "#EAF2F8", "#93A9BA"
BG, PANEL, EDGE = "#0A1220", "rgba(18,30,50,.82)", "#243750"
TEAL, SKY, CORAL, GOLD, VIOLET, SLATE = "#2DD4BF", "#38BDF8", "#FB7185", "#FBBF24", "#A78BFA", "#7B8FA3"
FAM_COLOR = {"Baseline": "#475569", "Single": SLATE, "Voting": GOLD, "Bagging": SKY,
             "Boosting": TEAL, "Stacking": VIOLET}
FAMILY = {"Decision Tree (depth 3)": "Single", "Logistic Regression": "Single", "KNN (k=25)": "Single",
          "Voting (LR+DT+KNN, soft)": "Voting", "Random Forest": "Bagging",
          "Bagging (100 deep trees)": "Bagging", "AdaBoost": "Boosting", "Gradient Boosting": "Boosting",
          "XGBoost": "Boosting", "Stacking (RF+GB+XGB → LR)": "Stacking"}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp {{ font-family: 'Plus Jakarta Sans', system-ui, sans-serif; color:{INK}; }}
.stApp {{ background: radial-gradient(1100px 600px at 6% -10%, rgba(45,212,191,.16) 0%, transparent 60%),
                     radial-gradient(900px 500px at 100% 0%, rgba(167,139,250,.16) 0%, transparent 55%),
                     linear-gradient(180deg, #0A1220 0%, #0C1627 60%, #0E1A2E 100%) !important; color-scheme: dark; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 1.6rem; max-width: 1280px; }}
section[data-testid="stSidebar"] > div {{ background: linear-gradient(180deg,#0C1A2B 0%,#0F1D33 100%); border-right:1px solid #1E3048; }}
h1,h2,h3,h4 {{ color:{INK}; letter-spacing:-0.02em; font-weight:800; }}
@property --i {{ syntax:'<integer>'; initial-value:0; inherits:false; }}
@property --f {{ syntax:'<integer>'; initial-value:0; inherits:false; }}
@keyframes cu {{ from {{ --i:0; --f:0; }} to {{ --i:var(--ti); --f:var(--tf); }} }}
@keyframes rise {{ from {{ opacity:0; transform:translateY(18px) scale(.985); }} to {{ opacity:1; transform:none; }} }}
@keyframes floaty {{ 0%,100% {{ transform:translateY(0) rotate(-4deg); }} 50% {{ transform:translateY(-8px) rotate(4deg); }} }}
@keyframes shine {{ 0% {{ background-position:-200% 0; }} 100% {{ background-position:200% 0; }} }}
@keyframes pulse {{ 0% {{ box-shadow:0 0 0 0 rgba(255,122,92,.45); }} 100% {{ box-shadow:0 0 0 16px rgba(255,122,92,0); }} }}
.cu {{ animation: cu 1.9s cubic-bezier(.16,.84,.3,1) .15s both; counter-reset: ci var(--i) cf var(--f); }}
.cu2::after {{ content: counter(ci) "." counter(cf, decimal-leading-zero) var(--suf); }}
.cu1::after {{ content: counter(ci) "." counter(cf) var(--suf); }}
.cu0::after {{ content: counter(ci) var(--suf); }}
.hero {{ position:relative; overflow:hidden; border-radius:28px; padding:34px 38px; color:white; animation:rise .8s both;
        background: linear-gradient(120deg,#0B5E63 0%,#0E7C86 38%,#4C3FA8 100%); box-shadow:0 24px 60px -24px rgba(45,212,191,.45); border:1px solid rgba(255,255,255,.12); }}
.hero::before {{ content:""; position:absolute; inset:0; background:linear-gradient(110deg,transparent 30%,rgba(255,255,255,.22) 48%,transparent 66%);
        background-size:200% 100%; animation:shine 5.5s linear infinite; }}
.hero .blob {{ position:absolute; border-radius:50%; filter:blur(2px); opacity:.18; background:#fff; }}
.hero h1 {{ color:white; font-size:2.5rem; margin:.1rem 0 .3rem 0; line-height:1.1; }}
.hero .kicker {{ text-transform:uppercase; letter-spacing:.16em; font-weight:700; font-size:.78rem; opacity:.9; }}
.hero .trophy {{ position:absolute; right:42px; top:22px; font-size:5.2rem; animation:floaty 4.2s ease-in-out infinite; filter:drop-shadow(0 10px 18px rgba(0,0,0,.25)); }}
.chip {{ display:inline-block; padding:6px 14px; border-radius:999px; background:rgba(255,255,255,.2); border:1px solid rgba(255,255,255,.45);
        font-weight:700; font-size:.88rem; margin:10px 8px 0 0; backdrop-filter:blur(4px); }}
.tile {{ background:{PANEL}; border:1px solid {EDGE}; border-radius:22px; padding:20px 22px; min-height:168px;
        box-shadow:0 10px 30px -18px rgba(0,0,0,.7); animation:rise .7s both; transition:transform .25s, box-shadow .25s; backdrop-filter:blur(6px); }}
.tile:hover {{ transform:translateY(-4px); box-shadow:0 18px 38px -18px rgba(0,0,0,.8); }}
.tile .lab {{ font-size:.78rem; text-transform:uppercase; letter-spacing:.1em; color:{MUTED}; font-weight:700; }}
.tile .val {{ font-size:2.15rem; font-weight:800; letter-spacing:-.03em; margin:4px 0 2px 0; line-height:1.15; }}
.tile .sub {{ font-size:.84rem; color:{MUTED}; }}
.tile .bar {{ height:6px; border-radius:6px; margin-bottom:12px; }}
.card {{ background:{PANEL}; border:1px solid {EDGE}; border-radius:22px; padding:20px 24px; animation:rise .7s both;
        box-shadow:0 10px 30px -20px rgba(14,47,59,.35); }}
.insight {{ border-left:5px solid {TEAL}; background:linear-gradient(90deg,rgba(45,212,191,.16),rgba(18,30,50,.5)); border-radius:14px; padding:14px 18px; margin:10px 0; animation:rise .7s both; }}
.insight.warn {{ border-left-color:{GOLD}; background:linear-gradient(90deg,rgba(251,191,36,.16),rgba(18,30,50,.5)); }}
.insight.bad {{ border-left-color:{CORAL}; background:linear-gradient(90deg,rgba(251,113,133,.17),rgba(18,30,50,.5)); }}
.insight b {{ color:{INK}; }}
.decision {{ border-radius:24px; padding:26px; text-align:center; color:white; font-weight:800; animation:rise .6s both; }}
.decision.ok {{ background:linear-gradient(120deg,#0F8F84,#1FB89E); box-shadow:0 18px 40px -18px #2DD4BF; }}
.decision.no {{ background:linear-gradient(120deg,#E11D48,#FB7185); box-shadow:0 18px 40px -18px #FB7185; animation:rise .6s both, pulse 1.6s 2; }}
.decision .big {{ font-size:2.4rem; letter-spacing:-.02em; }}
.stTabs [role="tablist"], .stTabs [data-baseweb="tab-list"] {{ gap:6px; background:{PANEL}; padding:6px; border-radius:18px; border:1px solid {EDGE}; flex-wrap:wrap; }}
.stTabs [role="tab"] {{ border-radius:14px !important; padding:10px 18px !important; font-weight:700; color:{MUTED}; height:auto; transition:all .25s; cursor:pointer; }}
.stTabs [role="tab"] p {{ font-weight:700; margin:0; }}
.stTabs [role="tab"]:hover {{ background:#17283F; }}
.stTabs [role="tab"][aria-selected="true"] {{ background:linear-gradient(120deg,{TEAL},{SKY}) !important; color:#06202A !important; box-shadow:0 8px 18px -10px {TEAL}; }}
.stTabs [role="tab"][aria-selected="true"] p {{ color:#06202A !important; }}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"], .stTabs .react-aria-SelectionIndicator, .stTabs [role="tablist"]::after {{ display:none !important; }}
div[data-testid="stPlotlyChart"] {{ background:{PANEL}; border:1px solid {EDGE}; border-radius:22px; padding:8px 8px 0 8px; animation:rise .8s both;
        box-shadow:0 10px 30px -22px rgba(14,47,59,.4); }}
div[data-testid="stDataFrame"] {{ border-radius:18px; overflow:hidden; border:1px solid {EDGE}; }}
.stButton>button {{ border-radius:14px; border:0; font-weight:700; padding:.55rem 1.1rem; color:#06202A;
        background:linear-gradient(120deg,{TEAL},{SKY}); box-shadow:0 10px 22px -12px {TEAL}; transition:transform .2s; }}
.stButton>button:hover {{ transform:translateY(-2px) scale(1.02); color:#06202A; }}
div[data-baseweb="slider"] div[role="slider"] {{ background:{TEAL} !important; box-shadow:0 0 0 4px rgba(15,168,154,.18) !important; }}
div[data-testid="stSliderThumbValue"] {{ color:{TEAL} !important; }}
.stApp, .stApp p, .stApp label, .stApp span, .stApp li, .stApp div[data-testid="stMarkdownContainer"], .stApp [data-testid="stWidgetLabel"] p {{ color:{INK}; }}
.stApp .small, .stApp .small b {{ color:{MUTED}; }}
.stApp .small b {{ color:{INK}; }}
.stApp [data-testid="stCaptionContainer"] p {{ color:{MUTED}; }}
.stApp .tile .lab, .stApp .tile .sub {{ color:{MUTED}; }}
.stApp .hero, .stApp .hero p, .stApp .hero span, .stApp .hero h1, .stApp .hero div, .stApp .decision, .stApp .decision div {{ color:white; }}
.stApp input, .stApp textarea, .stApp div[data-baseweb="select"] > div, .stApp div[data-baseweb="input"], .stApp div[data-baseweb="base-input"] {{ background:#15233A !important; color:{INK} !important; border-color:{EDGE} !important; }}
.stApp div[data-baseweb="select"] span, .stApp div[data-baseweb="select"] svg {{ color:{INK} !important; fill:{INK} !important; }}
div[data-baseweb="popover"] ul, div[data-baseweb="popover"] li {{ background:#15233A !important; color:{INK} !important; }}
.stNumberInput button {{ background:#1C2E49 !important; color:{INK} !important; }}
div[data-testid="stExpander"] {{ background:{PANEL}; border:1px solid {EDGE}; border-radius:16px; }}
div[data-testid="stExpander"] summary p, div[data-testid="stExpander"] summary span {{ color:{INK}; }}
div[data-baseweb="slider"] > div > div {{ background:#243750; }}
div[data-testid="stSliderTickBarMin"], div[data-testid="stSliderTickBarMax"] {{ color:{MUTED} !important; }}
div[data-testid="stRadio"] label div[data-baseweb="radio"] > div:first-child {{ border-color:{TEAL} !important; background:#15233A !important; }}
div[data-testid="stSegmentedControl"] button {{ background:#15233A !important; color:{INK} !important; border-color:{EDGE} !important; }}
div[data-testid="stSegmentedControl"] button[aria-checked="true"], div[data-testid="stSegmentedControl"] button[aria-pressed="true"] {{ background:linear-gradient(120deg,{TEAL},{SKY}) !important; color:#06202A !important; }}
.lb {{ overflow-x:auto; border:1px solid {EDGE}; border-radius:18px; background:{PANEL}; animation:rise .8s both; }}
.lb table {{ width:100%; border-collapse:collapse; font-size:.9rem; }}
.lb th {{ text-align:left; padding:12px 14px; color:{MUTED}; font-size:.74rem; text-transform:uppercase; letter-spacing:.08em; border-bottom:1px solid {EDGE}; background:#0F1B2E; }}
.lb td {{ padding:10px 14px; border-bottom:1px solid #1A2B43; color:{INK}; }}
.lb tr:hover td {{ background:rgba(45,212,191,.07); }}
.small {{ color:{MUTED}; font-size:.86rem; }}
.sect {{ font-size:1.25rem; font-weight:800; margin:.6rem 0 .1rem 0; }}
</style>
""", unsafe_allow_html=True)


def style_fig(fig, h=420, legend=True):
    fig.update_layout(template="plotly_dark", height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Plus Jakarta Sans, sans-serif", color=INK, size=13),
                      margin=dict(l=12, r=12, t=52, b=76 if legend else 12), showlegend=legend,
                      title=dict(font=dict(size=17, color=INK), x=0.01),
                      hoverlabel=dict(bgcolor="#15233A", font_size=13, bordercolor="#243750", font_color="#EAF2F8"),
                      legend=dict(orientation="h", y=-0.15, x=0))
    fig.update_xaxes(gridcolor="#1E3048", zerolinecolor="#2C4260")
    fig.update_yaxes(gridcolor="#1E3048", zerolinecolor="#2C4260")
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def fmt_aed(v, short=False):
    if short:
        return f"{CUR} {v/1e6:,.2f}M" if abs(v) >= 1e6 else f"{CUR} {v/1e3:,.0f}K"
    return f"{CUR} {v:,.0f}"


def tile(label, value, sub="", prefix="", suffix="", dec=0, color=TEAL, delay=0.0):
    v = round(float(value), dec)
    iv = int(v); fv = int(round((v - iv) * 10 ** dec)) if dec else 0
    cls = {0: "cu0", 1: "cu1", 2: "cu2"}[dec]
    pre = f"{prefix} " if prefix else ""
    return (f'<div class="tile" style="animation-delay:{delay}s"><div class="bar" style="background:linear-gradient(90deg,{color},{color}55)"></div>'
            f'<div class="lab">{label}</div><div class="val" style="color:{INK}">{pre}'
            f'<span class="cu {cls}" style="--ti:{iv};--tf:{fv};--suf:\'{suffix}\';--dot:\'.\'"></span></div>'
            f'<div class="sub">{sub}</div></div>')


def insight(text, kind=""):
    st.markdown(f'<div class="insight {kind}">{text}</div>', unsafe_allow_html=True)


# ───────────────────────────── Data ─────────────────────────────
def money_saved(y_true, y_pred, loan, interest=INTEREST):
    y_true, y_pred, loan = np.asarray(y_true), np.asarray(y_pred), np.asarray(loan)
    caught = (y_pred == 1) & (y_true == 1)
    wrong = (y_pred == 1) & (y_true == 0)
    return loan[caught].sum() - interest * loan[wrong].sum()


def add_flags(d):
    d = d.copy()
    d["flag_employment_gt_adult_life"] = (d.employment_years > d.age - 16).astype(int)
    d["flag_credit_gt_adult_life"] = (d.credit_history_months / 12 > d.age - 18).astype(int)
    return d


def load_data():
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (os.path.join(here, "loans.csv"), "loans.csv", "/mnt/user-data/uploads/loans.csv"):
        if os.path.exists(p):
            return pd.read_csv(p)
    up = st.sidebar.file_uploader("Upload loans.csv", type="csv")
    if up is not None:
        return pd.read_csv(up)
    return None


# ───────────────────────────── Training (mirrors the notebook) ─────────────────────────────
@st.cache_resource(show_spinner=False)
def run_bakeoff(raw_csv_bytes: bytes):
    df = pd.read_csv(io.BytesIO(raw_csv_bytes))
    clean = add_flags(df)
    X, y = clean.drop(columns="default"), clean["default"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
    loan_train, loan_test = X_train["loan_amount"].values, X_test["loan_amount"].values

    cat_cols = ["home_ownership"]
    num_feats = [c for c in X.columns if c not in cat_cols]

    def make(clf, scaled=False):
        num = StandardScaler() if scaled else "passthrough"
        pre = ColumnTransformer([("num", num, num_feats),
                                 ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols)])
        return Pipeline([("pre", pre), ("clf", clf)])

    xgb_kw = dict(n_estimators=400, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8,
                  eval_metric="logloss", n_jobs=-1, random_state=SEED)
    try:
        XGBClassifier(**xgb_kw)
    except TypeError:  # HistGradientBoosting fallback has different args
        xgb_kw = dict(max_iter=400, learning_rate=0.05, max_depth=3, random_state=SEED)
    gb_kw = dict(n_estimators=200, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=SEED)

    singles = {
        "Decision Tree (depth 3)": make(DecisionTreeClassifier(max_depth=3, random_state=SEED)),
        "Logistic Regression": make(LogisticRegression(max_iter=2000), scaled=True),
        "KNN (k=25)": make(KNeighborsClassifier(n_neighbors=25), scaled=True),
    }
    voting = VotingClassifier(
        estimators=[("lr", make(LogisticRegression(max_iter=2000), scaled=True)),
                    ("dt", make(DecisionTreeClassifier(max_depth=5, random_state=SEED))),
                    ("knn", make(KNeighborsClassifier(n_neighbors=25), scaled=True))], voting="soft")
    bag_models = {
        "Random Forest": make(RandomForestClassifier(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=SEED)),
        "Bagging (100 deep trees)": make(BaggingClassifier(DecisionTreeClassifier(random_state=SEED), n_estimators=100,
                                                           n_jobs=-1, random_state=SEED)),
    }
    boost_models = {
        "AdaBoost": make(AdaBoostClassifier(n_estimators=200, learning_rate=0.5, random_state=SEED)),
        "Gradient Boosting": make(GradientBoostingClassifier(**gb_kw)),
        "XGBoost": make(XGBClassifier(**xgb_kw)),
    }
    stacking = StackingClassifier(
        estimators=[("rf", make(RandomForestClassifier(n_estimators=200, min_samples_leaf=3, n_jobs=-1, random_state=SEED))),
                    ("gb", make(GradientBoostingClassifier(**gb_kw))),
                    ("xgb", make(XGBClassifier(**xgb_kw)))],
        final_estimator=LogisticRegression(max_iter=2000),
        cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=-1)
    all_models = {**singles, "Voting (LR+DT+KNN, soft)": voting, **bag_models, **boost_models,
                  "Stacking (RF+GB+XGB → LR)": stacking}

    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    grid = np.round(np.arange(0.03, 0.81, 0.01), 2)
    y_te = y_test.values
    probs, curves, fitted = {}, {}, {}
    for name, m in all_models.items():
        fitted[name] = clone(m).fit(X_train, y_train)
        probs[name] = fitted[name].predict_proba(X_test)[:, 1]
        oof = cross_val_predict(clone(m), X_train, y_train, cv=skf, method="predict_proba", n_jobs=1)[:, 1]
        curves[name] = np.array([money_saved(y_train, (oof >= t).astype(int), loan_train) for t in grid])

    def metrics(p, thr):
        pred = (p >= thr).astype(int)
        return dict(Accuracy=accuracy_score(y_te, pred), Recall=recall_score(y_te, pred),
                    Precision=precision_score(y_te, pred, zero_division=0),
                    Money=money_saved(y_te, pred, loan_test),
                    Caught=int(((pred == 1) & (y_te == 1)).sum()), Wrong=int(((pred == 1) & (y_te == 0)).sum()))

    rows = []
    for name in all_models:
        t_best = float(grid[curves[name].argmax()])
        base = dict(Model=name, Family=FAMILY[name], CV_money=float(curves[name].max()), Best_thr=t_best,
                    AUC=roc_auc_score(y_te, probs[name]))
        rows.append({**base, "Cutoff": "default", "Threshold": 0.5, **metrics(probs[name], 0.5)})
        rows.append({**base, "Cutoff": "tuned", "Threshold": t_best, **metrics(probs[name], t_best)})
    res = pd.DataFrame(rows)

    tuned = res[res.Cutoff == "tuned"].sort_values("CV_money", ascending=False)
    champ = tuned.iloc[0].Model
    champ_thr = float(tuned.iloc[0].Threshold)

    pi = permutation_importance(fitted[champ], X_test, y_test, scoring="roc_auc", n_repeats=5,
                                random_state=SEED, n_jobs=1)
    importance = pd.Series(pi.importances_mean, index=X_test.columns).sort_values()

    return dict(df=df, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, loan_test=loan_test,
                probs=probs, curves=curves, grid=grid, res=res, champ=champ, champ_thr=champ_thr,
                model=fitted[champ], importance=importance)


# ───────────────────────────── Boot ─────────────────────────────
df_in = load_data()
st.sidebar.markdown("## 💰 QuickLoan")
st.sidebar.caption("Champion-model command centre")
if df_in is None:
    st.markdown('<div class="hero"><div class="kicker">QuickLoan · Bake-Off</div><h1>Drop in your data to begin</h1>'
                '<div>Place <b>loans.csv</b> next to app.py or upload it from the sidebar.</div></div>', unsafe_allow_html=True)
    st.stop()

csv_bytes = df_in.to_csv(index=False).encode()
with st.spinner("🏁 Running the bake-off: training 10 models, tuning thresholds on out-of-fold data… (first load only, ~1–3 min)"):
    S = run_bakeoff(csv_bytes)

df, res, champ, champ_thr = S["df"], S["res"], S["champ"], S["champ_thr"]
y_te, p_te, loan_te = S["y_test"].values, S["probs"][S["champ"]], S["loan_test"]
pred_c = (p_te >= champ_thr).astype(int)
cm = confusion_matrix(y_te, pred_c)
tn, fp, fn, tp = cm.ravel()
champ_money = money_saved(y_te, pred_c, loan_te)
total_def_principal = loan_te[y_te == 1].sum()
champ_row = res[(res.Model == champ) & (res.Cutoff == "tuned")].iloc[0]
single_best_default = res[(res.Family == "Single") & (res.Cutoff == "default")].Money.max()
single_best_row = res[(res.Family == "Single") & (res.Cutoff == "default")].sort_values("Money", ascending=False).iloc[0]
ens_default = res[(res.Family != "Single") & (res.Cutoff == "default")].sort_values("Money", ascending=False).iloc[0]
most_acc = res.sort_values("Accuracy", ascending=False).iloc[0]
lost_interest = INTEREST * loan_te[(pred_c == 1) & (y_te == 0)].sum()
caught_principal = loan_te[(pred_c == 1) & (y_te == 1)].sum()

# Sidebar facts
st.sidebar.markdown("---")
st.sidebar.markdown(f"**🏆 Champion**  \n{champ}")
st.sidebar.markdown(f"**🎯 Reject if**  \nP(default) ≥ **{champ_thr:.2f}**")
st.sidebar.markdown(f"**💸 Saved**  \n{fmt_aed(champ_money)}")
st.sidebar.markdown("---")
st.sidebar.markdown(f"<div class='small'>Dataset: <b>{len(df):,}</b> loans · default rate <b>{df.default.mean():.1%}</b><br>"
                    f"Test set: <b>{len(y_te):,}</b> loans (30 %, stratified)<br>"
                    f"Money rule: caught defaulter = <b>+loan</b>; wrongly rejected = <b>−{INTEREST:.0%} × loan</b></div>",
                    unsafe_allow_html=True)
if st.sidebar.button("🎉 Celebrate the champion"):
    st.balloons()

# ───────────────────────────── Hero ─────────────────────────────
st.markdown(f"""
<div class="hero">
<div class="blob" style="width:220px;height:220px;right:-40px;bottom:-90px"></div>
<div class="blob" style="width:120px;height:120px;right:240px;top:-50px"></div>
<div class="trophy">🏆</div>
<div class="kicker">QuickLoan · The Great Model Bake-Off</div>
<h1>{champ}</h1>
<div style="font-size:1.08rem;max-width:760px;opacity:.96">is our champion. Picked by <b>cross-validated money on the training set</b> (never the test set),
it protects <b>{fmt_aed(champ_money, True)}</b> versus approving everyone.</div>
<span class="chip">🎯 Reject when P(default) ≥ {champ_thr:.2f}</span>
<span class="chip">📈 AUC {champ_row.AUC:.3f}</span>
<span class="chip">🧪 Tested on {len(y_te):,} unseen loans</span>
</div>
""", unsafe_allow_html=True)
st.write("")

tabs = st.tabs(["🏆 Champion", "📊 Bake-off", "🎚️ Threshold lab", "🔍 Under the hood", "🧮 Applicant simulator", "🗂️ Data explorer"])

# ═════════════════════ TAB 1 · CHAMPION ═════════════════════
with tabs[0]:
    c = st.columns(4)
    c[0].markdown(tile("Money saved vs approving everyone", champ_money / 1e6, f"{fmt_aed(champ_money)} on the test set",
                       prefix=CUR, suffix="M", dec=2, color=TEAL, delay=0.0), unsafe_allow_html=True)
    c[1].markdown(tile("Defaulter principal protected", 100 * champ_money / total_def_principal,
                       f"of {fmt_aed(total_def_principal, True)} held by defaulters", suffix="%", dec=1, color=SKY, delay=.08), unsafe_allow_html=True)
    c[2].markdown(tile("Defaulters caught", 100 * tp / (tp + fn), f"{tp} of {tp + fn} defaulters rejected",
                       suffix="%", dec=1, color=CORAL, delay=.16), unsafe_allow_html=True)
    c[3].markdown(tile("Good borrowers wrongly rejected", 100 * fp / (fp + tn), f"{fp} of {fp + tn} · cost {fmt_aed(lost_interest, True)} in interest",
                       suffix="%", dec=1, color=GOLD, delay=.24), unsafe_allow_html=True)
    st.write("")

    l, r = st.columns([1.15, 1])
    with l:
        wf = go.Figure(go.Waterfall(
            orientation="v", measure=["relative", "relative", "total"],
            x=["Loans saved from<br>caught defaulters", "Interest lost on<br>wrongly rejected", "Net money saved"],
            y=[caught_principal, -lost_interest, 0],
            text=[f"+{fmt_aed(caught_principal, True)}", f"−{fmt_aed(lost_interest, True)}", fmt_aed(champ_money, True)],
            textposition="outside", connector=dict(line=dict(color="#4B6580", dash="dot")),
            increasing=dict(marker=dict(color=TEAL)), decreasing=dict(marker=dict(color=CORAL)),
            totals=dict(marker=dict(color=SKY))))
        wf.update_layout(title="Where the money comes from", yaxis_title=f"{CUR}")
        wf.update_yaxes(range=[0, caught_principal * 1.18])
        show(style_fig(wf, 430, legend=False))
    with r:
        lad = pd.DataFrame({
            "Step": ["Approve everyone", f"Best single model<br>({single_best_row.Model.split(' (')[0]})",
                     f"Best ensemble @0.5<br>({ens_default.Model.split(' (')[0]})", f"Champion, tuned<br>cut-off {champ_thr:.2f}"],
            "Money": [0, single_best_default, ens_default.Money, champ_money],
            "Color": [FAM_COLOR["Baseline"], SLATE, SKY, TEAL]})
        fig = go.Figure(go.Bar(x=lad.Step, y=lad.Money, marker_color=lad.Color, marker_cornerradius=10,
                               text=[fmt_aed(v, True) if v else f"{CUR} 0" for v in lad.Money], textposition="outside",
                               hovertemplate="%{x}<br>%{y:,.0f}<extra></extra>"))
        fig.update_layout(title="The money ladder")
        fig.update_yaxes(range=[0, champ_money * 1.2], title=CUR)
        show(style_fig(fig, 430, legend=False))

    a, b = st.columns([1, 1.25])
    with a:
        d = go.Figure(go.Pie(labels=["Caught (rejected)", "Missed (approved, then default)"], values=[tp, fn], hole=.66,
                             marker=dict(colors=[TEAL, CORAL], line=dict(color="#12203A", width=3)),
                             textinfo="percent", textfont=dict(size=15, color="#06202A"), sort=False, pull=[0.03, 0]))
        d.add_annotation(text=f"<b>{tp + fn}</b><br><span style='font-size:12px'>defaulters<br>in test</span>", showarrow=False, font=dict(size=24))
        d.update_layout(title="Of every defaulter in the test set…")
        show(style_fig(d, 380, legend=True))
    with b:
        st.markdown('<div class="sect">What the numbers say</div>', unsafe_allow_html=True)
        insight(f"<b>Accuracy would have picked the wrong model.</b> The most accurate option ({most_acc.Model}, cut-off {most_acc.Threshold:.2f}) "
                f"scores {most_acc.Accuracy:.1%} but saves only {fmt_aed(most_acc.Money, True)}. Our champion is {champ_row.Accuracy:.1%} accurate and saves "
                f"<b>{fmt_aed(champ_money, True)}</b>: every missed defaulter costs 100 % of a loan, every false alarm only 20 %.", "bad")
        insight(f"<b>+{fmt_aed(champ_money - single_best_default, True)}</b> beyond the best weak single model at the default cut-off "
                f"({single_best_row.Model}: {fmt_aed(single_best_default, True)}). Most of this comes from <b>ensembles + a money-aware threshold</b>.")
        top5 = res[res.Cutoff == "tuned"].sort_values("Money", ascending=False).head(5).Money
        insight(f"<b>Honest reading:</b> after tuning, the top 5 models sit within <b>{100 * (top5.max() - top5.min()) / top5.max():.1f}%</b> of each other "
                f"({fmt_aed(top5.min(), True)}–{fmt_aed(top5.max(), True)}), inside the noise of a single {len(y_te):,}-row test set. "
                "A tuned Logistic Regression is a strong, explainable fallback. Re-validate on fresh data before go-live.", "warn")

# ═════════════════════ TAB 2 · BAKE-OFF ═════════════════════
with tabs[1]:
    st.markdown('<div class="sect">The leaderboard, animated</div>', unsafe_allow_html=True)
    st.caption("Press ▶ to watch every model jump from the naive 0.5 cut-off to its money-optimal cut-off. Same test set for everyone.")
    metric = st.radio("Metric", ["Money saved", "Recall", "Accuracy", "Precision"], horizontal=True, label_visibility="collapsed")
    mcol = {"Money saved": "Money", "Recall": "Recall", "Accuracy": "Accuracy", "Precision": "Precision"}[metric]
    order = res[res.Cutoff == "tuned"].sort_values("Money")["Model"].tolist()
    d0 = res[res.Cutoff == "default"].set_index("Model").loc[order]
    d1 = res[res.Cutoff == "tuned"].set_index("Model").loc[order]
    f = (lambda v: fmt_aed(v, True)) if mcol == "Money" else (lambda v: f"{v:.1%}")
    colors = [FAM_COLOR[FAMILY[m]] for m in order]
    outline = [CORAL if m == champ else "rgba(0,0,0,0)" for m in order]

    def bars(d, label):
        return go.Bar(y=order, x=d[mcol], orientation="h", marker=dict(color=colors, cornerradius=8,
                      line=dict(color=outline, width=3)), text=[f(v) for v in d[mcol]], textposition="outside",
                      customdata=np.c_[d.Threshold], hovertemplate="%{y}<br>%{text} @ cut-off %{customdata[0]:.2f}<extra></extra>", name=label)

    xmax = max(d0[mcol].max(), d1[mcol].max()) * 1.22
    xmin = 0 if mcol != "Accuracy" else 0.75
    fig = go.Figure(data=[bars(d0, "default")], frames=[go.Frame(data=[bars(d0, "default")], name="Default 0.5 cut-off"),
                                                        go.Frame(data=[bars(d1, "tuned")], name="Money-optimal cut-off")])
    fig.update_layout(
        title=f"{metric}: default cut-off → money-optimal cut-off (🔶 = champion)",
        xaxis=dict(range=[xmin, xmax]),
        updatemenus=[dict(type="buttons", direction="left", x=0.0, y=1.11, showactive=False, pad=dict(t=0, r=6),
                          buttons=[dict(label="▶  Tune the cut-offs", method="animate",
                                        args=[["Money-optimal cut-off"], dict(frame=dict(duration=1300, redraw=True), transition=dict(duration=1300, easing="cubic-in-out"), fromcurrent=True)]),
                                   dict(label="↺  Reset", method="animate",
                                        args=[["Default 0.5 cut-off"], dict(frame=dict(duration=600, redraw=True), transition=dict(duration=600, easing="cubic-in-out"))])])],
        sliders=[dict(active=0, x=0.30, len=0.6, y=1.11, pad=dict(t=0, b=0), currentvalue=dict(visible=False), ticklen=0,
                      steps=[dict(label=n, method="animate", args=[[n], dict(mode="immediate", frame=dict(duration=900, redraw=True), transition=dict(duration=900))])
                             for n in ["Default 0.5 cut-off", "Money-optimal cut-off"]])])
    show(style_fig(fig, 600, legend=False).update_layout(margin=dict(t=130), title=dict(y=0.97, yanchor="top")))
    leg = " ".join(f"<span style='display:inline-block;width:11px;height:11px;border-radius:4px;background:{c};margin:0 5px 0 12px'></span>{k}"
                   for k, c in FAM_COLOR.items() if k != "Baseline")
    st.markdown(f"<div class='small'>{leg}</div>", unsafe_allow_html=True)
    st.write("")

    cl, cr = st.columns(2)
    with cl:
        sc = go.Figure()
        for cutoff, sym in [("default", "circle-open"), ("tuned", "circle")]:
            d = res[res.Cutoff == cutoff]
            sc.add_trace(go.Scatter(x=d.Accuracy, y=d.Money, mode="markers", name=f"{cutoff} cut-off",
                                    marker=dict(size=17, symbol=sym, color=[FAM_COLOR[x] for x in d.Family], line=dict(width=2.5, color=[FAM_COLOR[x] for x in d.Family])),
                                    text=d.Model, hovertemplate="%{text}<br>Accuracy %{x:.1%}<br>Money %{y:,.0f}<extra></extra>"))
        sc.add_annotation(x=champ_row.Accuracy, y=champ_row.Money, text="🏆 champion", showarrow=True, arrowcolor=CORAL, ay=-40, font=dict(color=CORAL, size=13))
        sc.update_layout(title="The accuracy trap: higher accuracy ≠ more money", xaxis_title="Accuracy", yaxis_title=f"Money saved ({CUR})", xaxis_tickformat=".0%")
        show(style_fig(sc, 430))
    with cr:
        ens = res[res.Cutoff == "default"]
        gain = res.pivot(index="Model", columns="Cutoff", values="Money")
        gain["Gain"] = gain.tuned - gain.default
        gain = gain.sort_values("Gain")
        g = go.Figure(go.Bar(y=gain.index, x=gain.Gain, orientation="h", marker=dict(color=[FAM_COLOR[FAMILY[m]] for m in gain.index], cornerradius=8),
                             text=[fmt_aed(v, True) for v in gain.Gain], textposition="outside"))
        g.update_layout(title="Money unlocked by threshold tuning alone")
        g.update_xaxes(range=[0, gain.Gain.max() * 1.5])
        show(style_fig(g, 430, legend=False))

    st.markdown('<div class="sect">Full leaderboard</div>', unsafe_allow_html=True)
    cut = st.segmented_control("Cut-off", ["tuned", "default"], default="tuned") or "tuned"
    t = res[res.Cutoff == cut].sort_values("Money", ascending=False)[["Model", "Family", "Threshold", "Accuracy", "Recall", "Precision", "Money", "Caught", "Wrong", "AUC"]]
    t = t.rename(columns={"Threshold": "Cut-off", "Money": f"Money saved ({CUR})", "Caught": "Defaulters caught", "Wrong": "Good rejected"})
    mcol_ = f"Money saved ({CUR})"
    sty = (t.style.format({"Cut-off": "{:.2f}", "Accuracy": "{:.1%}", "Recall": "{:.1%}", "Precision": "{:.1%}", mcol_: "{:,.0f}", "AUC": "{:.3f}"})
           .bar(subset=["Recall", "Precision"], color="#14796F", vmin=0, vmax=1)
           .bar(subset=["Accuracy"], color="#1B6E9B", vmin=0.8, vmax=1)
           .bar(subset=[mcol_], color="#1F8F83", vmin=0, vmax=float(res.Money.max())))
    st.markdown(f'<div class="lb">{sty.hide(axis="index").to_html()}</div>', unsafe_allow_html=True)

# ═════════════════════ TAB 3 · THRESHOLD LAB ═════════════════════
with tabs[2]:
    st.markdown('<div class="sect">Find the money-optimal cut-off</div>', unsafe_allow_html=True)
    st.caption(f"Theory: reject when P(default) × loan > (1 − P) × {INTEREST:.0%} × loan → threshold = {INTEREST / (1 + INTEREST):.3f}. "
               "In practice we tune it on out-of-fold TRAIN predictions, then apply it once to the test set.")
    cA, cB = st.columns([1, 2])
    with cA:
        mname = st.selectbox("Model", list(S["probs"].keys()), index=list(S["probs"].keys()).index(champ))
    best_t = float(res[(res.Model == mname) & (res.Cutoff == "tuned")].Threshold.iloc[0])
    with cB:
        thr = st.slider("Rejection threshold on P(default)", 0.03, 0.80, float(best_t), 0.01,
                        help="Drag to see what happens to the test-set money.")
    pm = S["probs"][mname]
    predm = (pm >= thr).astype(int)
    mm = money_saved(y_te, predm, loan_te)
    cmm = confusion_matrix(y_te, predm).ravel()
    k = st.columns(5)
    k[0].markdown(tile(f"Money saved ({CUR}, test)", mm / 1e6, "", suffix="M", dec=2, color=TEAL), unsafe_allow_html=True)
    k[1].markdown(tile("Recall", 100 * recall_score(y_te, predm), "", suffix="%", dec=1, color=CORAL), unsafe_allow_html=True)
    k[2].markdown(tile("Accuracy", 100 * accuracy_score(y_te, predm), "", suffix="%", dec=1, color=SKY), unsafe_allow_html=True)
    k[3].markdown(tile("Defaulters caught", cmm[3], f"of {cmm[2] + cmm[3]}", dec=0, color=VIOLET), unsafe_allow_html=True)
    k[4].markdown(tile("Good rejected", cmm[1], f"of {cmm[0] + cmm[1]}", dec=0, color=GOLD), unsafe_allow_html=True)

    st.write("")
    grid = S["grid"]
    scale = len(y_te) / len(S["y_train"])
    test_curve = np.array([money_saved(y_te, (pm >= t_).astype(int), loan_te) for t_ in grid])
    cur = go.Figure()
    cur.add_trace(go.Scatter(x=grid, y=S["curves"][mname] * scale, name="Out-of-fold TRAIN, scaled to test size (used to choose)", line=dict(color=TEAL, width=4),
                             fill="tozeroy", fillcolor="rgba(45,212,191,.12)"))
    cur.add_trace(go.Scatter(x=grid, y=test_curve, name="TEST (illustration only)", line=dict(color=SKY, width=3, dash="dot")))
    cur.add_trace(go.Scatter(x=[best_t], y=[S["curves"][mname].max() * scale], mode="markers+text", name="Tuned optimum", text=[f" optimum {best_t:.2f}"],
                             textposition="top right", marker=dict(size=16, color=CORAL, line=dict(width=3, color="#12203A"))))
    cur.add_trace(go.Scatter(x=[thr], y=[test_curve[int(np.argmin(np.abs(grid - thr)))]], mode="markers", name="Your threshold",
                             marker=dict(size=18, color=GOLD, symbol="diamond", line=dict(width=3, color="#12203A"))))
    cur.add_vline(x=0.5, line_dash="dash", line_color="#7B8FA3", annotation_text="naive 0.5", annotation_position="top")
    cur.add_vline(x=INTEREST / (1 + INTEREST), line_dash="dot", line_color=INK, annotation_text=f"theory {INTEREST / (1 + INTEREST):.3f}", annotation_position="top left")
    cur.update_layout(title=f"Money vs. threshold · {mname}", xaxis_title="Rejection threshold on P(default)", yaxis_title=f"Money saved ({CUR})")
    show(style_fig(cur, 470))

    # animated sweep: composition of decisions as the threshold sweeps
    sw_grid = np.round(np.arange(0.03, 0.81, 0.03), 2)
    comp = []
    for t_ in sw_grid:
        pr = (pm >= t_).astype(int)
        comp.append((loan_te[(pr == 1) & (y_te == 1)].sum(), INTEREST * loan_te[(pr == 1) & (y_te == 0)].sum(),
                     money_saved(y_te, pr, loan_te)))
    comp = np.array(comp)

    def sweep_data(i):
        return [go.Bar(x=["Loans saved", "Interest lost", "Net saved"], y=[comp[i, 0], -comp[i, 1], comp[i, 2]],
                       marker=dict(color=[TEAL, CORAL, SKY], cornerradius=10),
                       text=[fmt_aed(comp[i, 0], True), "−" + fmt_aed(comp[i, 1], True), fmt_aed(comp[i, 2], True)], textposition="outside")]
    sf = go.Figure(data=sweep_data(0), frames=[go.Frame(data=sweep_data(i), name=f"{t_:.2f}") for i, t_ in enumerate(sw_grid)])
    sf.update_layout(title="▶ Watch the trade-off as the threshold sweeps from strict to lenient",
                     updatemenus=[dict(type="buttons", x=0, y=1.12, showactive=False, direction="left",
                                       buttons=[dict(label="▶ Play sweep", method="animate", args=[None, dict(frame=dict(duration=450, redraw=True), transition=dict(duration=350), fromcurrent=True)]),
                                                dict(label="⏸ Pause", method="animate", args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False))])])],
                     sliders=[dict(x=0.32, len=0.66, y=1.12, currentvalue=dict(prefix="threshold = ", font=dict(size=13)), pad=dict(t=0),
                                   steps=[dict(label=f"{t_:.2f}", method="animate", args=[[f"{t_:.2f}"], dict(mode="immediate", frame=dict(duration=300, redraw=True), transition=dict(duration=250))]) for t_ in sw_grid])])
    sf.update_yaxes(range=[-comp[:, 1].max() * 1.3, max(comp[:, 0].max(), 1) * 1.2], title=CUR)
    show(style_fig(sf, 500, legend=False).update_layout(margin=dict(t=130), title=dict(y=0.97, yanchor="top")))
    insight("<b>Why the cut-off falls so far below 0.5:</b> a missed defaulter costs a whole loan while a false alarm costs only 20 % of one, "
            "so it pays to reject at much lower risk. Push the threshold too low, though, and the interest you give up starts to eat the savings.")

# ═════════════════════ TAB 4 · UNDER THE HOOD ═════════════════════
with tabs[3]:
    a, b = st.columns(2)
    with a:
        z = np.array([[tn, fp], [fn, tp]])
        hm = go.Figure(go.Heatmap(z=z, x=["Approve", "Reject"], y=["Repays", "Defaults"], colorscale=[[0, "#14243B"], [1, "#0F9C8E"]],
                                  text=[[f"<b>{tn}</b><br>approved ✔" , f"<b>{fp}</b><br>false alarm"], [f"<b>{fn}</b><br>missed defaulter", f"<b>{tp}</b><br>caught ✔"]],
                                  texttemplate="%{text}", showscale=False, hovertemplate="%{y} → %{x}: %{z}<extra></extra>", xgap=5, ygap=5))
        hm.update_layout(title=f"Confusion matrix · {champ} @ {champ_thr:.2f}", xaxis_title="Model decision", yaxis_title="Actual", yaxis_autorange="reversed")
        show(style_fig(hm, 400, legend=False))
    with b:
        imp = S["importance"]
        fi = go.Figure(go.Bar(y=imp.index, x=imp.values, orientation="h", marker=dict(color=imp.values, colorscale=[[0, "#14796F"], [1, TEAL]], cornerradius=7),
                              text=[f"{v:.3f}" for v in imp.values], textposition="outside"))
        fi.update_layout(title="What drives the champion? (AUC drop when shuffled)")
        fi.update_xaxes(range=[0, max(imp.max(), 1e-3) * 1.2])
        show(style_fig(fi, 400, legend=False))

    a, b = st.columns(2)
    with a:
        hist = go.Figure()
        hist.add_trace(go.Histogram(x=p_te[y_te == 0], name="Repaid", marker_color=TEAL, opacity=.75, xbins=dict(start=0, end=1, size=0.025)))
        hist.add_trace(go.Histogram(x=p_te[y_te == 1], name="Defaulted", marker_color=CORAL, opacity=.8, xbins=dict(start=0, end=1, size=0.025)))
        hist.add_vline(x=champ_thr, line_color=INK, line_dash="dash", annotation_text=f"reject ≥ {champ_thr:.2f}")
        hist.update_layout(barmode="overlay", title="Risk scores: how well are the two groups separated?", xaxis_title="Predicted P(default)", yaxis_title="Applicants", yaxis_type="log")
        show(style_fig(hist, 400))
    with b:
        roc = go.Figure()
        top = res[res.Cutoff == "tuned"].sort_values("AUC", ascending=False).head(5).Model
        for m in top:
            fpr, tpr, _ = roc_curve(y_te, S["probs"][m])
            roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{m.split(' (')[0]} (AUC {roc_auc_score(y_te, S['probs'][m]):.3f})",
                                     line=dict(width=4 if m == champ else 2, color=CORAL if m == champ else None)))
        roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(color="#4B6580", dash="dash"), name="coin flip"))
        roc.update_layout(title="ROC curves · top models by AUC", xaxis_title="False-positive rate", yaxis_title="Recall")
        show(style_fig(roc, 400))

    # gains curve (money-weighted)
    order_i = np.argsort(-p_te)
    cum_def = np.cumsum((loan_te * y_te)[order_i]) / total_def_principal
    pct = np.arange(1, len(p_te) + 1) / len(p_te)
    gn = go.Figure()
    gn.add_trace(go.Scatter(x=pct, y=cum_def, name="Champion ranking", line=dict(color=TEAL, width=4), fill="tozeroy", fillcolor="rgba(45,212,191,.12)"))
    gn.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random review", line=dict(color="#4B6580", dash="dash")))
    rej_share = pred_c.mean()
    gn.add_trace(go.Scatter(x=[rej_share], y=[caught_principal / total_def_principal], mode="markers+text", name="Operating point",
                            text=[f"  reject top {rej_share:.0%} → protect {caught_principal / total_def_principal:.0%} of defaulter principal"], textposition="middle right",
                            marker=dict(size=16, color=CORAL, line=dict(color="#12203A", width=3))))
    gn.update_layout(title="Gains: review the riskiest applicants first", xaxis_title="Share of applicants rejected (riskiest first)",
                     yaxis_title="Share of defaulter principal caught", xaxis_tickformat=".0%", yaxis_tickformat=".0%")
    show(style_fig(gn, 430))

    st.markdown('<div class="sect">Fairness check · who gets rejected?</div>', unsafe_allow_html=True)
    t = S["X_test"].copy(); t["y"] = y_te; t["pred"] = pred_c
    t["Age group"] = pd.cut(t.age, [17, 30, 40, 50, 65], labels=["≤30", "31-40", "41-50", "51-64"])
    t["Income band"] = pd.qcut(t.annual_income, 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
    t["Home"] = t.home_ownership
    fc = st.columns(3)
    for col, g in zip(fc, ["Age group", "Home", "Income band"]):
        grp = t.groupby(g, observed=True)
        d = pd.DataFrame({"Actual default rate": grp.y.mean(), "Rejection rate": grp.pred.mean(), "n": grp.size()}).reset_index()
        fg = go.Figure()
        fg.add_trace(go.Bar(x=d[g], y=d["Actual default rate"], name="Actual default rate", marker=dict(color=CORAL, cornerradius=7)))
        fg.add_trace(go.Bar(x=d[g], y=d["Rejection rate"], name="Rejection rate", marker=dict(color=TEAL, cornerradius=7)))
        fg.update_layout(barmode="group", title=g, yaxis_tickformat=".0%")
        with col:
            show(style_fig(fg, 330))
    insight("A healthy model rejects more where the <b>actual default rate</b> is higher. Age is flat. Renters and low-income applicants are rejected more, "
            "which tracks their higher real default rates, but gaps like these are worth monitoring and keeping explainable (small groups are noisy).", "warn")

# ═════════════════════ TAB 5 · SIMULATOR ═════════════════════
with tabs[4]:
    st.markdown('<div class="sect">Score a live applicant with the champion</div>', unsafe_allow_html=True)
    st.caption("Debt-to-income and the two data-quality flags are derived automatically, exactly as in the training data.")
    ci, co = st.columns([1.05, 1])
    with ci:
        r1 = st.columns(2)
        age = r1[0].slider("Age", 21, 64, 35)
        inc = r1[1].number_input(f"Annual income ({CUR})", 20000, 400000, 60000, 1000)
        r2 = st.columns(2)
        loan = r2[0].number_input(f"Loan amount ({CUR})", 5000, 250000, 90000, 1000)
        term = r2[1].select_slider("Loan term (months)", [12, 24, 36, 48, 60], 48)
        r3 = st.columns(2)
        emp = r3[0].slider("Employment (years)", 0.0, 35.0, 3.0, 0.1)
        hist_m = r3[1].slider("Credit history (months)", 0, 299, 60)
        r4 = st.columns(3)
        nloans = r4[0].slider("Existing loans", 0, 6, 3)
        pdef = r4[1].slider("Past defaults", 0, 4, 1)
        home = r4[2].selectbox("Home", ["rent", "mortgage", "own"])
        dti = round(loan / inc, 2)
        st.markdown(f"<div class='small'>Derived → debt-to-income <b>{dti:.2f}</b></div>", unsafe_allow_html=True)

    app = add_flags(pd.DataFrame([dict(age=age, annual_income=float(inc), loan_amount=float(loan), loan_term_months=term, employment_years=emp,
                                       credit_history_months=hist_m, num_existing_loans=nloans, past_defaults=pdef, debt_to_income=dti,
                                       home_ownership=home, default=0)])).drop(columns="default")
    p_app = float(S["model"].predict_proba(app)[0, 1])
    reject = p_app >= champ_thr
    with co:
        gauge = go.Figure(go.Indicator(mode="gauge+number", value=p_app * 100, number=dict(suffix="%", font=dict(size=46, color=INK)),
                                       title=dict(text="Predicted probability of default", font=dict(size=15)),
                                       gauge=dict(axis=dict(range=[0, 100], ticksuffix="%"), bar=dict(color=CORAL if reject else TEAL, thickness=.3),
                                                  bgcolor="rgba(0,0,0,0)", borderwidth=0,
                                                  steps=[dict(range=[0, champ_thr * 100], color="rgba(45,212,191,.22)"), dict(range=[champ_thr * 100, 100], color="rgba(251,113,133,.25)")],
                                                  threshold=dict(line=dict(color=INK, width=4), thickness=.85, value=champ_thr * 100))))
        gauge.update_layout(transition=dict(duration=700, easing="cubic-in-out"))
        show(style_fig(gauge, 300, legend=False))
        if reject:
            st.markdown(f'<div class="decision no"><div class="big">🚫 REJECT</div>risk {p_app:.1%} ≥ cut-off {champ_thr:.0%}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="decision ok"><div class="big">✅ APPROVE</div>risk {p_app:.1%} &lt; cut-off {champ_thr:.0%}</div>', unsafe_allow_html=True)

    exp_approve = p_app * loan
    exp_reject = (1 - p_app) * INTEREST * loan
    e1, e2 = st.columns([1, 1.3])
    with e1:
        ev = go.Figure(go.Bar(x=["If we APPROVE", "If we REJECT"], y=[exp_approve, exp_reject], marker=dict(color=[TEAL, CORAL], cornerradius=10),
                              text=[fmt_aed(exp_approve), fmt_aed(exp_reject)], textposition="outside"))
        ev.update_layout(title="Expected loss of each decision", yaxis_title=CUR)
        ev.update_yaxes(range=[0, max(exp_approve, exp_reject) * 1.25 + 1])
        show(style_fig(ev, 370, legend=False))
    with e2:
        ls = np.linspace(5000, 250000, 60)
        sens = pd.DataFrame([dict(age=age, annual_income=float(inc), loan_amount=float(x), loan_term_months=term, employment_years=emp,
                                  credit_history_months=hist_m, num_existing_loans=nloans, past_defaults=pdef, debt_to_income=round(x / inc, 2),
                                  home_ownership=home, default=0) for x in ls])
        sens = add_flags(sens).drop(columns="default")
        ps = S["model"].predict_proba(sens)[:, 1]
        sf2 = go.Figure()
        sf2.add_trace(go.Scatter(x=ls, y=ps, line=dict(color=TEAL, width=4, shape="spline"), name="P(default)", fill="tozeroy", fillcolor="rgba(45,212,191,.12)"))
        sf2.add_hline(y=champ_thr, line_dash="dash", line_color=INK, annotation_text=f"reject ≥ {champ_thr:.2f}")
        sf2.add_trace(go.Scatter(x=[loan], y=[p_app], mode="markers", name="This applicant",
                                 marker=dict(size=17, color=CORAL, line=dict(color="#12203A", width=3))))
        sf2.update_layout(title="What-if: how does the loan size change this applicant's risk?", xaxis_title=f"Loan amount ({CUR})", yaxis_title="P(default)", yaxis_tickformat=".0%")
        show(style_fig(sf2, 370))
    insight("The decision rule is economic: <b>reject when the expected loss from approving exceeds the expected interest lost by rejecting</b>. "
            f"Here that is {fmt_aed(exp_approve)} vs {fmt_aed(exp_reject)} → <b>{'reject' if exp_approve > exp_reject else 'approve'}</b> "
            f"(the model uses a tuned cut-off of {champ_thr:.2f} instead of the theoretical {INTEREST / (1 + INTEREST):.3f}).")

# ═════════════════════ TAB 6 · DATA EXPLORER ═════════════════════
with tabs[5]:
    k = st.columns(4)
    k[0].markdown(tile("Loans", len(df), "6 000 historical applications".replace("6 000", f"{len(df):,}"), dec=0, color=TEAL), unsafe_allow_html=True)
    k[1].markdown(tile("Default rate", 100 * df.default.mean(), "class imbalance: why accuracy misleads", suffix="%", dec=1, color=CORAL), unsafe_allow_html=True)
    k[2].markdown(tile("Avg loan", df.loan_amount.mean() / 1000, f"{CUR} thousand", suffix="K", dec=1, color=SKY), unsafe_allow_html=True)
    k[3].markdown(tile("Avg income", df.annual_income.mean() / 1000, f"{CUR} thousand / year", suffix="K", dec=1, color=GOLD), unsafe_allow_html=True)
    st.write("")

    e = df.copy()
    e["Income quartile"] = pd.qcut(e.annual_income, 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
    _q = pd.qcut(e.debt_to_income, 5, duplicates="drop")
    e["Debt-to-income"] = _q.map(lambda iv: f"{max(iv.left, 0):.2f}–{iv.right:.2f}").astype(str)
    cols = st.columns(2)
    spec = [("Income quartile", cols[0]), ("Debt-to-income", cols[1]), ("past_defaults", cols[0]), ("home_ownership", cols[1])]
    for g, col in spec:
        d = e.groupby(g, observed=True).default.agg(["mean", "size"]).reset_index()
        bg = go.Figure(go.Bar(x=d[g].astype(str), y=d["mean"], marker=dict(color=d["mean"], colorscale=[[0, "#14B8A6"], [.5, GOLD], [1, CORAL]], cornerradius=8),
                              text=[f"{v:.1%}" for v in d["mean"]], textposition="outside", customdata=d["size"],
                              hovertemplate="%{x}<br>default rate %{y:.1%}<br>n=%{customdata}<extra></extra>"))
        bg.update_layout(title=f"Default rate by {g.replace('_', ' ')}", yaxis_tickformat=".0%")
        bg.update_yaxes(range=[0, d["mean"].max() * 1.25])
        with col:
            show(style_fig(bg, 340, legend=False))

    a, b = st.columns([1.2, 1])
    with a:
        sm = e.sample(min(2500, len(e)), random_state=SEED).sort_values("default")
        sp = px.scatter(sm, x="annual_income", y="loan_amount", color=sm.default.map({0: "Repaid", 1: "Defaulted"}), opacity=.65,
                        color_discrete_map={"Repaid": TEAL, "Defaulted": CORAL}, labels={"annual_income": f"Annual income ({CUR})", "loan_amount": f"Loan amount ({CUR})", "color": ""})
        sp.update_traces(marker=dict(size=7))
        sp.update_layout(title="Big loans on small incomes are where defaults live")
        show(style_fig(sp, 430))
    with b:
        num = [c for c in df.columns if c != "home_ownership"]
        cr = df[num].corr().round(2)
        ch = go.Figure(go.Heatmap(z=cr.values, x=cr.columns, y=cr.columns, zmin=-1, zmax=1, colorscale=[[0, SKY], [.5, "#14243B"], [1, CORAL]],
                                  hovertemplate="%{y} × %{x}: %{z}<extra></extra>", xgap=2, ygap=2))
        ch.update_layout(title="Correlation matrix", yaxis_autorange="reversed")
        show(style_fig(ch, 430, legend=False))
    insight(f"<b>Data quality:</b> no missing values or duplicates. {int((add_flags(df).flag_credit_gt_adult_life).sum()):,} rows have a credit history longer than adult life "
            f"and {int((add_flags(df).flag_employment_gt_adult_life).sum()):,} have employment longer than working age; we kept them and added two flag columns. "
            "<b>debt_to_income ≈ loan ÷ income</b> (correlation 0.999), so it is redundant but harmless for tree models.", "warn")
    with st.expander("Peek at the raw data"):
        st.dataframe(df.head(200), width="stretch", hide_index=True)

st.markdown("<div class='small' style='text-align:center;margin:28px 0 8px 0'>Built from the QuickLoan Bake-Off notebook · champion chosen on out-of-fold training money, evaluated once on a held-out stratified test set.</div>",
            unsafe_allow_html=True)
