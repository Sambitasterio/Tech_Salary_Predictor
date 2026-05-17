import json
import numpy as np
import pandas as pd
import shap
import streamlit as st
from xgboost import XGBRegressor

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Tech Salary Predictor", page_icon="💰", layout="wide")

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("models/model_meta.json") as f:
        meta = json.load(f)
    xgb = XGBRegressor()
    xgb.load_model("models/xgb_model.json")
    return xgb, meta["feature_names"], meta["median_ed"]

xgb, feature_names, median_ed = load_model()

# ── Constants ─────────────────────────────────────────────────────────────────
REGION_MAP = {
    "United States of America": "North_America",
    "Canada": "North_America",
    "Mexico": "North_America",
    "Germany": "Western_Europe",
    "France": "Western_Europe",
    "United Kingdom of Great Britain and Northern Ireland": "Western_Europe",
    "Netherlands": "Western_Europe",
    "Switzerland": "Western_Europe",
    "Austria": "Western_Europe",
    "Belgium": "Western_Europe",
    "Sweden": "Western_Europe",
    "Denmark": "Western_Europe",
    "Norway": "Western_Europe",
    "Finland": "Western_Europe",
    "Ireland": "Western_Europe",
    "Italy": "Western_Europe",
    "Spain": "Western_Europe",
    "Portugal": "Western_Europe",
    "Greece": "Western_Europe",
    "Ukraine": "Eastern_Europe",
    "Poland": "Eastern_Europe",
    "Romania": "Eastern_Europe",
    "Czech Republic": "Eastern_Europe",
    "Hungary": "Eastern_Europe",
    "Russia": "Eastern_Europe",
    "Bulgaria": "Eastern_Europe",
    "Serbia": "Eastern_Europe",
    "Croatia": "Eastern_Europe",
    "India": "South_Asia",
    "Pakistan": "South_Asia",
    "Bangladesh": "South_Asia",
    "Sri Lanka": "South_Asia",
    "Nepal": "South_Asia",
    "Brazil": "Latin_America",
    "Argentina": "Latin_America",
    "Colombia": "Latin_America",
    "Chile": "Latin_America",
    "Peru": "Latin_America",
    "Israel": "Middle_East_Africa",
    "Turkey": "Middle_East_Africa",
    "Nigeria": "Middle_East_Africa",
    "South Africa": "Middle_East_Africa",
    "Egypt": "Middle_East_Africa",
    "United Arab Emirates": "Middle_East_Africa",
    "Saudi Arabia": "Middle_East_Africa",
    "Australia": "Oceania",
    "New Zealand": "Oceania",
    "China": "East_SE_Asia",
    "Japan": "East_SE_Asia",
    "South Korea": "East_SE_Asia",
    "Singapore": "East_SE_Asia",
    "Vietnam": "East_SE_Asia",
    "Indonesia": "East_SE_Asia",
    "Philippines": "East_SE_Asia",
    "Malaysia": "East_SE_Asia",
    "Thailand": "East_SE_Asia",
}

ED_MAP = {
    "Primary / elementary school": 0,
    "Secondary school": 1,
    "Some college (no degree)": 2,
    "Associate degree": 3,
    "Bachelor's degree": 4,
    "Master's degree": 5,
    "PhD / Professional degree": 6,
}

LANGUAGES = sorted([c.replace("lang_", "").replace("_", " ")
                    for c in feature_names if c.startswith("lang_")])
DATABASES = sorted([c.replace("db_", "").replace("_", " ")
                    for c in feature_names if c.startswith("db_")])
FRAMEWORKS = sorted([c.replace("frame_", "").replace("_", " ")
                     for c in feature_names if c.startswith("frame_")])
ROLES = sorted([c.replace("role_", "")
                for c in feature_names if c.startswith("role_")])
COUNTRIES = sorted(REGION_MAP.keys()) + ["Other"]

MAE_USD = 32_370  # from Phase 7

# ── Helper ────────────────────────────────────────────────────────────────────
def build_feature_row(country, years_exp, ed_level, role, languages, databases, frameworks):
    row = {col: 0 for col in feature_names}
    row["WorkExp"] = min(float(years_exp), 50)
    row["EdLevel_ord"] = ED_MAP.get(ed_level, 4)

    def _set(prefix, values):
        for v in values:
            col = prefix + v.replace(" ", "_").replace("/", "_").replace("-", "_") \
                           .replace(".", "").replace("(", "").replace(")", "")
            if col in row:
                row[col] = 1

    _set("lang_",  languages)
    _set("db_",    databases)
    _set("frame_", frameworks)

    role_col = "role_" + role
    if role_col in row:
        row[role_col] = 1

    region = REGION_MAP.get(country, "Other")
    reg_col = "region_" + region
    if reg_col in row:
        row[reg_col] = 1

    return pd.DataFrame([row]), region


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("💰 Tech Salary Predictor")
st.caption("Stack Overflow Developer Survey 2025 · XGBoost · R²=0.56 · MAE=$32,370")

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("Your Profile")

    country      = st.selectbox("Country", COUNTRIES, index=COUNTRIES.index("United States of America"))
    years_exp    = st.slider("Years of professional experience", 0, 50, 5)
    ed_level     = st.selectbox("Education level", list(ED_MAP.keys()), index=4)
    role         = st.selectbox("Primary role", ROLES, index=ROLES.index("Developer, full-stack") if "Developer, full-stack" in ROLES else 0)
    languages    = st.multiselect("Languages used", LANGUAGES, default=["Python", "JavaScript"] if "Python" in LANGUAGES else [])
    databases    = st.multiselect("Databases used", DATABASES, default=["PostgreSQL"] if "PostgreSQL" in DATABASES else [])
    frameworks   = st.multiselect("Frameworks used", FRAMEWORKS, default=[])

    predict_btn  = st.button("Predict Salary", type="primary", use_container_width=True)

with col_right:
    st.subheader("Prediction")

    if predict_btn:
        X_input, region = build_feature_row(
            country, years_exp, ed_level, role, languages, databases, frameworks
        )
        log_pred = xgb.predict(X_input)[0]
        salary   = int(np.expm1(log_pred))
        lo, hi   = max(0, salary - MAE_USD), salary + MAE_USD

        st.metric("Predicted Annual Salary (USD)", f"${salary:,}")
        st.caption(f"Confidence band: ${lo:,} – ${hi:,}  (± MAE of $32,370)")
        st.caption(f"Region mapped to: **{region.replace('_', ' ')}**")

        st.divider()

        # SHAP waterfall
        st.markdown("**What's driving this prediction?**")
        try:
            explainer   = shap.TreeExplainer(xgb)
            shap_values = explainer(X_input)
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 5))
            shap.plots.waterfall(shap_values[0], max_display=10, show=False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.warning(f"SHAP plot unavailable: {e}")

        st.divider()
        st.markdown("**Key factors for your profile:**")
        exp_pct  = min(100, int(years_exp / 50 * 100))
        st.progress(exp_pct, text=f"Experience: {years_exp} yrs")
        n_tech   = len(languages) + len(databases) + len(frameworks)
        st.info(f"Tech stack: {n_tech} tools selected — more niche tools (Snowflake, Go, Scala) boost salary")

    else:
        st.info("Fill in your profile and click **Predict Salary**.")
        st.markdown("""
**Model trained on 17,679 full-time employed developers.**

Key salary drivers (SHAP):
- 🌍 **Location** — 38% of prediction power
- 🛠️ **Tech stack** — 35%
- 📅 **Experience** — 17%
- 👔 **Role** — 9%
- 🎓 **Education** — 1%

Top salary boosters: Snowflake · Ruby · Databricks · Go · Elixir
        """)

st.divider()
st.caption("Data: Stack Overflow Developer Survey 2025 · Model: XGBoost (n_estimators=278, max_depth=5) · Built with Streamlit")
