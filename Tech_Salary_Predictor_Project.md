# Tech Salary Predictor — Project Plan
**Dataset:** Stack Overflow Developer Survey (Kaggle)  
**Goal:** Predict developer salary given skills, experience, and location  
**Timeline:** 1–2 weeks

---

## Progress Tracker

| Phase | Status |
|-------|--------|
| 1. Setup & Data Acquisition | ✅ Complete |
| 2. Exploratory Data Analysis (EDA) | ✅ Complete |
| 3. Data Cleaning & Preprocessing | ✅ Complete |
| 4. Feature Engineering | 🔄 In Progress |
| 5. Baseline Modeling | ⬜ Not Started |
| 6. Advanced Modeling | ⬜ Not Started |
| 7. Model Evaluation | ⬜ Not Started |
| 8. SHAP Interpretability | ⬜ Not Started |
| 9. Insight Narrative | ⬜ Not Started |
| 10. Final Presentation / Portfolio Polish | ⬜ Not Started |

---

## Phase 1 — Setup & Data Acquisition
- [x] Create project folder structure (`/data`, `/notebooks`, `/models`, `/reports`)
- [x] Download Stack Overflow Developer Survey 2025 (`results.txt`, `schema.txt`, `survey.pdf`)
- [x] Set up virtual environment and install dependencies (`pandas`, `numpy`, `scikit-learn`, `xgboost`, `shap`, `matplotlib`, `seaborn`)
- [x] Load dataset and do a first-pass inspection (`df.head()`, `df.info()`, `df.describe()`)

> **2025 column name changes:** delimiter is `;` not `|`; `YearsCodePro` → `WorkExp`; `FrameworkHaveWorkedWith` → `WebframeHaveWorkedWith`; files are `.txt` not `.csv`

---

## Phase 2 — Exploratory Data Analysis (EDA)
- [ ] Identify target variable (`ConvertedCompYearly` or equivalent salary column)
- [ ] Plot salary distribution — check for skew, outliers, log-scale need
- [ ] Salary by **country** — bar/box plots; identify top-paying regions
- [ ] Salary by **years of professional coding experience** (`YearsCodePro`)
- [ ] Salary by **role/job title** (`DevType`)
- [ ] Salary by **tech stack** (languages, databases, frameworks used)
- [ ] Correlation heatmap for numeric features
- [ ] Document 3–5 key EDA findings as markdown cells in the notebook

---

## Phase 3 — Data Cleaning & Preprocessing
- [ ] Drop rows where target salary is null or extreme outlier (e.g., < $10k or > $500k)
- [ ] Filter to full-time employed respondents only
- [ ] Handle missing values:
  - Numeric cols → median imputation
  - Categorical cols → mode or "Unknown" fill
- [ ] Normalize salary to USD if multi-currency (use `ConvertedCompYearly`)
- [ ] Remove duplicate rows

---

## Phase 4 — Feature Engineering
- [ ] **Multi-hot encode** technology columns (`LanguageHaveWorkedWith`, `DatabaseHaveWorkedWith`, `FrameworkHaveWorkedWith`) — split pipe-delimited strings into binary columns
- [ ] **Bucket experience** into bands (0–2, 3–5, 6–10, 11–20, 20+ years)
- [ ] **Country clustering** — group countries into regions (North America, Western Europe, South/SE Asia, etc.) to reduce cardinality
- [ ] Ordinal encode education level (`EdLevel`)
- [ ] One-hot encode role/job type (`DevType`)
- [ ] Log-transform salary target (improves regression performance on skewed data)
- [ ] Split into train/test sets (80/20, stratified by region)

---

## Phase 5 — Baseline Modeling
- [ ] Train **Linear Regression** baseline
- [ ] Evaluate: R², MAE, RMSE on test set
- [ ] Plot residuals — check for patterns/heteroscedasticity
- [ ] Document baseline results in a comparison table

---

## Phase 6 — Advanced Modeling
- [ ] Train **Ridge Regression** (L2) — tune `alpha` with cross-validation
- [ ] Train **Lasso Regression** (L1) — observe feature sparsity
- [ ] Compare Ridge vs Lasso coefficients — which skills survive regularization?
- [ ] Train **XGBoost Regressor** — tune `n_estimators`, `max_depth`, `learning_rate`
- [ ] (Optional) Train **Random Forest Regressor** for ensemble comparison

---

## Phase 7 — Model Evaluation
- [ ] Side-by-side comparison table: Linear / Ridge / Lasso / XGBoost on R², MAE, RMSE
- [ ] Residual plots for best model — show where predictions break down
- [ ] Learning curves — diagnose overfitting/underfitting
- [ ] Identify subgroups where model underperforms (e.g., certain countries, senior roles)

---

## Phase 8 — SHAP Interpretability
- [ ] Install and import `shap`
- [ ] Compute SHAP values for XGBoost model
- [ ] **Summary plot** — global feature importance ranked by mean |SHAP|
- [ ] **Beeswarm plot** — direction and magnitude of each feature's effect
- [ ] **Waterfall plot** — single prediction breakdown (pick 2–3 example candidates)
- [ ] **Dependence plot** for top 3 features (e.g., Python, YearsCodePro, Region)

---

## Phase 9 — Insight Narrative
- [ ] Write a findings summary (can be in notebook or separate `report.md`):
  - Which skills add the most salary premium?
  - How much does location affect salary vs. skills?
  - Where does the model struggle and why?
- [ ] Create 3–5 "headline insights" (e.g., _"Python adds ~$8k vs Java in the same region"_)
- [ ] Build a simple prediction function: input a profile, get a salary estimate + confidence

---

## Phase 10 — Final Presentation / Portfolio Polish
- [ ] Clean and comment all notebooks — remove dead cells, add markdown headers
- [ ] Write a `README.md` with: project overview, how to run, key findings, model performance
- [ ] Export key charts as `.png` for portfolio/slides
- [ ] (Optional) Build a minimal Streamlit app — form inputs → salary prediction + SHAP explanation
- [ ] Upload to GitHub with clean commit history
- [ ] (Optional) Write a short Medium/LinkedIn post summarizing findings

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| `pandas` / `numpy` | Data manipulation |
| `matplotlib` / `seaborn` | Visualization |
| `scikit-learn` | Linear, Ridge, Lasso, preprocessing |
| `xgboost` | Gradient boosting regressor |
| `shap` | Model interpretability |
| `streamlit` | (Optional) interactive demo |
| `jupyter` | Notebook environment |

---

## Key Metrics to Hit

| Metric | Target |
|--------|--------|
| R² (test set) | > 0.60 |
| MAE | < $20,000 |
| SHAP plots | At least 3 types |
| Insight headlines | At least 3 clear statements |

---

*Update status in the Progress Tracker after completing each phase.*
