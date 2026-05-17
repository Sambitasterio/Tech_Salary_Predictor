# Tech Salary Predictor

Predicts developer annual salary (USD) from skills, experience, and location using the **Stack Overflow Developer Survey 2025** (49,191 respondents).

---

## Key Findings

| Insight | Detail |
|---------|--------|
| USA earns 5x India | Median $150,000 vs $30,222 |
| Go adds $29k over PHP | Median $98,612 vs $69,609 |
| 10+ yrs experience doubles salary | $45,088 (0-2 yrs) → $97,452 (11-20 yrs) |
| Engineering Manager earns $56k more than Full-stack | $135,000 vs $79,062 |
| Location and tech stack are near-equal drivers | 38% vs 35% of SHAP prediction power |

**Top skill boosters:** Snowflake (+$56k) · Ruby/Rails (+$34k) · Databricks SQL (+$29k) · Go (+$17k)

**Top skill dampeners:** Laravel (-$21k) · Dart (-$19k) · PHP (-$17k) · MySQL (-$12k)

---

## Model Performance

| Model | R² Test | MAE ($) | RMSE ($) |
|-------|---------|---------|----------|
| Linear Regression (baseline) | 0.506 | $34,683 | $54,731 |
| Ridge (α=10) | 0.506 | $34,684 | $54,753 |
| Lasso (α=0.0025) | 0.491 | $35,185 | $55,640 |
| **XGBoost (best)** | **0.563** | **$32,370** | **$52,123** |

- 76.2% of predictions within $40k of actual salary
- 95.5% of predictions within $80k of actual salary

---

## Project Structure

```
Tech_Salary_Predictor/
├── data/
│   ├── results.txt          # Raw survey data (Stack Overflow 2025)
│   ├── schema.txt           # Column descriptions
│   ├── cleaned.csv          # Phase 3 output — 17,679 rows
│   ├── features.csv         # Phase 4 output — 136 features
│   ├── X_train/test.csv     # Train/test splits
│   └── y_train/test.csv     # Log-salary targets
├── models/
│   ├── xgb_model.json       # Trained XGBoost model
│   └── model_meta.json      # Feature names + imputation values
├── notebooks/
│   ├── 01_data_inspection.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_cleaning.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_baseline_modeling.ipynb
│   ├── 06_advanced_modeling.ipynb
│   ├── 07_model_evaluation.ipynb
│   ├── 08_shap_interpretability.ipynb
│   └── 09_insight_narrative.ipynb
├── reports/                 # All saved charts (.png)
├── app.py                   # Streamlit salary predictor
├── requirements.txt
└── README.md
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit app

```bash
streamlit run app.py
```

### 3. Run notebooks in order

Open Jupyter and run notebooks `01` through `09` sequentially. Each phase saves its outputs for the next.

---

## Pipeline Summary

| Phase | Description | Output |
|-------|-------------|--------|
| 1 | Data inspection | First-pass look at 49,191 rows × 172 cols |
| 2 | EDA | Salary distributions, country/role/language analysis |
| 3 | Cleaning | 17,679 rows after outlier removal + imputation |
| 4 | Feature engineering | 136 features: multi-hot tech, regions, roles |
| 5 | Baseline modeling | Linear Regression R²=0.506, MAE=$34,683 |
| 6 | Advanced modeling | XGBoost best: R²=0.563, MAE=$32,370 |
| 7 | Model evaluation | Subgroup analysis — hardest: Middle East/Africa, easy: Oceania |
| 8 | SHAP interpretability | Location=38%, Tech=35%, Experience=17%, Role=9%, Education=1% |
| 9 | Insight narrative | 5 headline stats + prediction function |
| 10 | Portfolio polish | README + Streamlit app |

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| pandas / numpy | Data manipulation |
| matplotlib / seaborn | Visualization |
| scikit-learn | Linear, Ridge, Lasso |
| xgboost | Best model (R²=0.563) |
| shap | Model interpretability |
| streamlit | Interactive demo |

---

## Dataset

- **Source:** [Stack Overflow Developer Survey 2025](https://survey.stackoverflow.co/)
- **Raw size:** 49,191 respondents × 172 columns
- **After cleaning:** 17,679 full-time employed developers with valid USD salary
- **Salary range (cleaned):** $10,000 – $500,000
- **Target:** `ConvertedCompYearly` (log-transformed for modelling)
