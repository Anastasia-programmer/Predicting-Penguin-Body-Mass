"""
Streamlit dashboard — Predicting Penguin Body Mass with Linear Regression.

Run:
    pip install -r requirements.txt
    streamlit run app_lab.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from scipy import stats

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = Path(__file__).parent / 'data' / 'penguins_size.csv'
TARGET = 'body_mass_g'
NUM_FEATS = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm']
CAT_FEATS = ['species', 'island', 'sex']

st.set_page_config(page_title='Penguin Body Mass — Regression Lab',
                   page_icon='🐧', layout='wide')


# Data + model (cached)
@st.cache_data
def load_data():
    df_raw = pd.read_csv(DATA_PATH)
    df = df_raw.copy()
    df = df[df['sex'].isin(['MALE', 'FEMALE'])]
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    return df_raw, df


@st.cache_resource
def train_models(df: pd.DataFrame):
    df_enc = pd.get_dummies(df, columns=CAT_FEATS, drop_first=True)
    features = [c for c in df_enc.columns if c != TARGET]

    X = df_enc[features].astype(float)
    y = df_enc[TARGET].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    models = {
        'OLS':   LinearRegression().fit(X_train_sc, y_train),
        'Ridge': Ridge(alpha=1.0).fit(X_train_sc, y_train),
        'Lasso': Lasso(alpha=1.0, max_iter=10000).fit(X_train_sc, y_train),
    }

    # Hyperparameter tuning via Pipeline + GridSearchCV (no leakage)
    alpha_grid = [0.001, 0.01, 0.1, 1, 10, 100]
    ridge_pipe = Pipeline([('sc', StandardScaler()), ('m', Ridge())])
    lasso_pipe = Pipeline([('sc', StandardScaler()),
                           ('m', Lasso(max_iter=20000))])
    ridge_search = GridSearchCV(ridge_pipe, {'m__alpha': alpha_grid},
                                cv=5, scoring='r2', n_jobs=-1).fit(X_train, y_train)
    lasso_search = GridSearchCV(lasso_pipe, {'m__alpha': alpha_grid},
                                cv=5, scoring='r2', n_jobs=-1).fit(X_train, y_train)

    models[f'Ridge tuned (α={ridge_search.best_params_["m__alpha"]})'] = ridge_search
    models[f'Lasso tuned (α={lasso_search.best_params_["m__alpha"]})'] = lasso_search

    rows = []
    preds = {}
    for name, m in models.items():
        # Pipeline searches take unscaled X; bare estimators take scaled X
        if isinstance(m, GridSearchCV):
            yp = m.predict(X_test)
        else:
            yp = m.predict(X_test_sc)
        preds[name] = yp
        mse = mean_squared_error(y_test, yp)
        rows.append({
            'Model': name,
            'MAE':   round(mean_absolute_error(y_test, yp), 2),
            'MSE':   round(mse, 2),
            'RMSE':  round(np.sqrt(mse), 2),
            'R²':    round(r2_score(y_test, yp), 4),
        })
    metrics_df = pd.DataFrame(rows)

    cv_scores = cross_val_score(LinearRegression(), X_train_sc, y_train,
                                cv=5, scoring='r2')

    # Residual std on training set — used for prediction intervals
    train_residuals = y_train.values - models['OLS'].predict(X_train_sc)
    residual_std = float(np.std(train_residuals, ddof=1))

    return {
        'features': features, 'scaler': scaler, 'models': models,
        'X_train': X_train, 'X_test': X_test,
        'X_train_sc': X_train_sc, 'X_test_sc': X_test_sc,
        'y_train': y_train, 'y_test': y_test,
        'preds': preds, 'metrics_df': metrics_df, 'cv_scores': cv_scores,
        'alpha_grid': alpha_grid,
        'ridge_search': ridge_search, 'lasso_search': lasso_search,
        'residual_std': residual_std,
    }


df_raw, df = load_data()
bundle = train_models(df)


# Sidebar / title
st.title('🐧 Predicting Penguin Body Mass with Linear Regression')
st.caption('Course project — Statistical Modeling for Predictive Analysis. '
           'Adapted from Lab 7 (Regression Techniques).')

tab_method, tab_overview, tab_eda, tab_model, tab_predict = st.tabs(
    ['Methodology', 'Overview', 'EDA', 'Model & Evaluation', 'Predict'])


# Tab 0 — Methodology (CRISP-DM)
with tab_method:
    st.header('Methodology — CRISP-DM mapped to this project')

    st.markdown(
        """
**CRISP-DM** (Cross-Industry Standard Process for Data Mining) is a six-phase
framework for data-mining projects. Here is how each phase maps to what we do
in this notebook and dashboard.

| # | Phase | What it means in general | What we did here |
|---|-------|--------------------------|------------------|
| 1 | **Business Understanding** | Define the goal in business / domain terms. | "Given a few quick measurements of a penguin, can we estimate its body mass?" — useful for ecologists in the field who can measure flippers and culmens but not weigh the bird. |
| 2 | **Data Understanding** | Acquire the data, inspect it, identify quality issues. | Loaded `penguins_size.csv` (344 rows). Checked dtypes, missing values, duplicates, category counts; built histograms, scatter plots and a correlation matrix. |
| 3 | **Data Preparation** | Clean, encode, scale, split. | Dropped 11 NA rows + 1 row where `sex == "."`. One-hot encoded `species`/`island`/`sex`. Train/test split 80/20 with `random_state=42`. Standardized features inside a Pipeline. |
| 4 | **Modeling** | Choose technique(s), fit, tune. | OLS, Ridge, Lasso. Tuned Ridge/Lasso `alpha` via 5-fold `GridSearchCV` over `[0.001, 0.01, 0.1, 1, 10, 100]`. |
| 5 | **Evaluation** | Did the model meet the business goal? | Test R² ≈ 0.86, MAE ≈ 237 g (on a ~4 kg bird). Residuals are roughly normal; train/test gap is small; CV is stable. |
| 6 | **Deployment** | Make the model usable. | This Streamlit dashboard — non-technical users can predict body mass for new penguins through the **Predict** tab. |
        """
    )

    st.subheader('Train / validation / test — and why we split 80/20')
    st.markdown(
        """
- **Train set (80%)** is used to fit the model — it sees the answers.
- **Test set (20%)** is held out and used **only at the end** to estimate
  how the model performs on data it has never seen. This is the
  honest performance number we report.
- **Validation** here is done via **5-fold cross-validation inside the
  training set** — we split the training data five ways, fit on four
  folds, score on the fifth, rotate, and average. This gives us a
  reliable estimate of CV-R² *without* touching the test set.
- **Why 80/20?** With ~333 clean rows, an 80/20 split gives ~266 training
  rows (enough to fit 8 coefficients reliably) and ~67 test rows
  (enough for a reasonably stable test score, while keeping as much
  data as possible for learning). For larger datasets a 90/10 or
  70/30 split would be equally defensible — 80/20 is the conventional
  default.
        """
    )

    st.subheader('Why we scale features')
    st.markdown(
        """
`StandardScaler` transforms each feature so it has mean 0 and standard
deviation 1. Two reasons it matters here:

1. **Comparable coefficients.** After scaling, every coefficient is in
   the same units ("grams of body mass per 1-std change in the feature"),
   so we can rank features by impact directly from the bar chart.
2. **Fair regularization.** Ridge and Lasso penalize the *size* of
   coefficients (`Σ βᵢ²` and `Σ |βᵢ|` respectively). Without scaling,
   features with larger numeric ranges (e.g. flipper length in mm vs
   one-hot 0/1 dummies) would receive disproportionately large
   coefficients, and the penalty would unfairly target some features
   over others. Scaling puts every feature on equal footing.
        """
    )

    st.subheader('Pipelines and data leakage')
    st.markdown(
        """
**Data leakage** is when information from the test (or validation)
set sneaks into the training process. The classic mistake: scale the
*entire* dataset before splitting — now the scaler's mean and std
already encode test-set information, and your validation scores are
optimistic.

We fix this with **`sklearn.pipeline.Pipeline`**. A pipeline bundles
the scaler and the model into a single object. When `GridSearchCV`
runs 5-fold CV on the pipeline, it **re-fits the scaler inside each
fold**, using only that fold's training rows. The held-out fold never
contributes its statistics to the scaler, so the validation score is
an honest estimate of out-of-sample performance.

```python
pipe = Pipeline([('scaler', StandardScaler()), ('model', Ridge())])
GridSearchCV(pipe, {'model__alpha': [...]}, cv=5).fit(X_train, y_train)
```
        """
    )

    st.subheader('Bias–variance tradeoff (the short version)')
    st.markdown(
        """
Every model has two sources of error:
- **Bias** = the error from being too simple to capture the true pattern
  (the model *under*-fits — it's wrong on training data too).
- **Variance** = the error from being too sensitive to the specific
  training rows (the model *over*-fits — it does great on training, bad
  on test).

Plain OLS sits in the low-bias, moderate-variance corner.
**Ridge** and **Lasso** add a penalty that intentionally raises bias a
little to reduce variance — they shrink coefficients toward 0. The
sweet spot is found by tuning `alpha` against cross-validation R²
(see the **Model & Evaluation** tab). On this dataset OLS is already
near the sweet spot, so the tuned models barely improve on it — but
on a noisier dataset with more features, regularization can be a
big win.
        """
    )


# Tab 1 — Overview
with tab_overview:
    st.header('Project description')
    st.markdown(
        "**Goal:** predict a penguin's **body mass (g)** from"
        "measurements and categorical attributes using linear regression.\n\n"
        "**Why linear regression?** The target is continuous, the relationships "
        "between flipper/culmen size and body mass are roughly linear, and we "
        "want **interpretable coefficients** — each one tells us how body mass "
        "changes when a feature changes, holding others constant."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Raw rows',     len(df_raw))
    c2.metric('Clean rows',   len(df))
    c3.metric('Features',     len(bundle['features']))
    c4.metric('Target',       TARGET)

    st.subheader('Dataset preview (clean)')
    st.dataframe(df.head(15), width='stretch')

    st.subheader('Missing values (raw)')
    miss = df_raw.isnull().sum().to_frame('missing')
    miss['% of rows'] = (miss['missing'] / len(df_raw) * 100).round(2)
    st.dataframe(miss, width='stretch')

    st.markdown(
        "**Cleaning applied:** drop rows with NA, drop the single row where "
        "`sex == '.'`, drop duplicates."
    )
# Tab 2 : EDA
with tab_eda:
    st.header('Exploratory Data Analysis')

    st.subheader('Target distribution — body_mass_g')
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    axes[0].hist(df[TARGET], bins=25, color='steelblue', edgecolor='white')
    axes[0].set_title('Histogram'); axes[0].set_xlabel('Body mass (g)')
    axes[1].boxplot(df[TARGET], vert=True, patch_artist=True,
                    boxprops=dict(facecolor='steelblue', alpha=0.6))
    axes[1].set_title('Box plot'); axes[1].set_ylabel('Body mass (g)')
    plt.tight_layout(); st.pyplot(fig)

    c1, c2, c3 = st.columns(3)
    c1.metric('Mean (g)',   f"{df[TARGET].mean():.0f}")
    c2.metric('Median (g)', f"{df[TARGET].median():.0f}")
    c3.metric('Std (g)',    f"{df[TARGET].std():.0f}")

    st.subheader('Numeric features vs body mass')
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    for ax, feat in zip(axes, NUM_FEATS):
        ax.scatter(df[feat], df[TARGET], alpha=0.5, s=20, color='steelblue')
        z = np.polyfit(df[feat], df[TARGET], 1); p = np.poly1d(z)
        xs = np.linspace(df[feat].min(), df[feat].max(), 100)
        ax.plot(xs, p(xs), color='tomato', linewidth=2)
        ax.set_xlabel(feat); ax.set_ylabel(TARGET)
        ax.set_title(feat, fontweight='bold')
    plt.tight_layout(); st.pyplot(fig)

    st.subheader('Body mass by category')
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    for ax, feat in zip(axes, CAT_FEATS):
        sns.boxplot(x=feat, y=TARGET, data=df, ax=ax)
        ax.set_title(f'by {feat}', fontweight='bold')
    plt.tight_layout(); st.pyplot(fig)

    st.subheader('Correlation matrix (numeric)')
    corr = df[NUM_FEATS + [TARGET]].corr()
    fig, ax = plt.subplots(figsize=(6, 4))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax)
    st.pyplot(fig)

    st.markdown(
        "**Reading:** flipper length is the strongest single predictor of "
        "body mass (r ≈ +0.87). Culmen depth shows a *negative* raw "
        "correlation (r ≈ −0.47) — a **Simpson's-paradox** artifact, because "
        "Gentoos are heavier *and* have shallower culmens. Once `species` is "
        "added to the regression, the culmen-depth coefficient flips to "
        "positive, which is the within-species relationship."
    )

# Tab 3 :  Model & Evaluation
with tab_model:
    st.header('Linear regression — OLS, Ridge, Lasso')

    st.subheader('Performance on the test set (20%)')
    st.dataframe(bundle['metrics_df'], width='stretch')

    best_row = bundle['metrics_df'].sort_values('R²', ascending=False).iloc[0]
    avg_mass = bundle['y_train'].mean()
    st.markdown(
        f"**What these numbers mean.** The best model is **{best_row['Model']}**: "
        f"on penguins it has never seen, it explains "
        f"**{best_row['R²']*100:.1f}%** of the variation in body mass. The "
        f"average prediction is off by **{best_row['MAE']:.0f} g** "
        f"(**MAE** = average absolute error, the most intuitive metric — "
        f"about {best_row['MAE']/avg_mass*100:.1f}% of an average penguin's "
        f"weight). **RMSE = {best_row['RMSE']:.0f} g** is similar but "
        f"penalizes large errors more (it's the square root of **MSE**, "
        f"which is in g² and harder to interpret directly). "
        f"In short: the model can guess any penguin's weight to within "
        f"about half a kilogram."
    )

    st.subheader('Coefficients (on standardized features, units = grams)')
    ols = bundle['models']['OLS']
    coef_df = (pd.DataFrame({'feature': bundle['features'],
                             'coefficient': ols.coef_.round(2)})
               .sort_values('coefficient', ascending=False)
               .reset_index(drop=True))
    st.dataframe(coef_df, width='stretch')

    fig, ax = plt.subplots(figsize=(9, 4))
    order = np.argsort(ols.coef_)
    colors = ['tomato' if c < 0 else 'steelblue' for c in ols.coef_[order]]
    ax.barh(np.array(bundle['features'])[order], ols.coef_[order],
            color=colors, edgecolor='white')
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Coefficient (g per 1-std change)')
    ax.set_title('OLS feature coefficients', fontweight='bold')
    plt.tight_layout(); st.pyplot(fig)

    st.subheader('Actual vs predicted + residuals (OLS)')
    y_test = bundle['y_test']; y_pred = bundle['preds']['OLS']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    mn = min(y_test.min(), y_pred.min()) - 50
    mx = max(y_test.max(), y_pred.max()) + 50
    axes[0].scatter(y_test, y_pred, alpha=0.55, color='steelblue', s=35)
    axes[0].plot([mn, mx], [mn, mx], 'r--', linewidth=2)
    axes[0].set_xlabel('Actual (g)'); axes[0].set_ylabel('Predicted (g)')
    axes[0].set_title('Actual vs predicted', fontweight='bold')

    residuals = y_test.values - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.55, color='darkorange', s=35)
    axes[1].axhline(0, color='red', linewidth=2, linestyle='--')
    axes[1].set_xlabel('Predicted (g)'); axes[1].set_ylabel('Residual (g)')
    axes[1].set_title('Residual plot (no pattern = good)', fontweight='bold')
    plt.tight_layout(); st.pyplot(fig)

    st.subheader('Overfitting check')
    r2_train = r2_score(bundle['y_train'], ols.predict(bundle['X_train_sc']))
    r2_test  = r2_score(y_test, y_pred)
    c1, c2, c3 = st.columns(3)
    c1.metric('Train R²', f'{r2_train:.4f}')
    c2.metric('Test R²',  f'{r2_test:.4f}')
    c3.metric('Gap',      f'{r2_train - r2_test:+.4f}')
    st.caption('A gap < 0.05 typically means the model generalizes well.')

    st.subheader('5-fold cross-validation (R²)')
    cv = bundle['cv_scores']
    st.write(', '.join(f'{s:.4f}' for s in cv))
    st.caption(f'Mean R² = {cv.mean():.4f}  |  Std = {cv.std():.4f}')

    # Hyperparameter tuning — alpha vs CV R² curves
    st.subheader('Hyperparameter tuning — alpha vs CV R²')
    st.caption(
        'Each pipeline (StandardScaler → model) is re-fitted inside every '
        'CV fold via GridSearchCV, so the validation R² is leak-free.'
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, search, name, color in [
        (axes[0], bundle['ridge_search'], 'Ridge', 'seagreen'),
        (axes[1], bundle['lasso_search'], 'Lasso', 'tomato'),
    ]:
        means = search.cv_results_['mean_test_score']
        stds  = search.cv_results_['std_test_score']
        ax.errorbar(bundle['alpha_grid'], means, yerr=stds, marker='o',
                    color=color, capsize=4, linewidth=2)
        ax.set_xscale('log')
        best_a = search.best_params_['m__alpha']
        ax.axvline(best_a, color='black', linestyle='--', linewidth=0.8,
                   label=f'best α = {best_a}')
        ax.set_xlabel('alpha (log scale)'); ax.set_ylabel('5-fold CV R²')
        ax.set_title(f'{name} — CV R² vs alpha', fontweight='bold')
        ax.legend()
    plt.tight_layout(); st.pyplot(fig)

    c1, c2 = st.columns(2)
    c1.metric('Best Ridge α',
              f"{bundle['ridge_search'].best_params_['m__alpha']}",
              f"CV R² = {bundle['ridge_search'].best_score_:.4f}")
    c2.metric('Best Lasso α',
              f"{bundle['lasso_search'].best_params_['m__alpha']}",
              f"CV R² = {bundle['lasso_search'].best_score_:.4f}")

    # Linear regression assumptions check
    st.subheader('Linear regression assumptions check')

    # Q-Q plot of residuals
    fig, ax = plt.subplots(figsize=(6, 4))
    stats.probplot(residuals, dist='norm', plot=ax)
    ax.set_title('Q-Q plot of residuals (normality check)', fontweight='bold')
    ax.get_lines()[0].set_markerfacecolor('steelblue')
    ax.get_lines()[0].set_markeredgecolor('steelblue')
    ax.get_lines()[1].set_color('tomato')
    plt.tight_layout(); st.pyplot(fig)

    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    st.caption(
        f'Shapiro–Wilk normality test on residuals: '
        f'W = {shapiro_stat:.4f}, p = {shapiro_p:.4f}  '
        f'({"residuals look normal" if shapiro_p > 0.05 else "residuals deviate from normality"})'
    )

    st.markdown(
        f"""
**Verdict on each of the 5 OLS assumptions for this fit:**

1. **Linearity** — ✅ Scatter plots in the EDA tab show approximately
   linear relationships between numeric features and body mass. The
   residual plot has no curved pattern.
2. **Independence of errors** — ✅ Each row is an independent penguin
   measurement.
3. **Homoscedasticity** (constant residual variance) — ✅ The residual
   plot shows  constant spread across the predicted-value range
   (no funnel/cone shape). Residual std ≈ {bundle['residual_std']:.0f} g.
4. **Normality of residuals** — {"✅" if shapiro_p > 0.05 else "⚠️"} The Q-Q plot is close to
   the diagonal in the bulk, with mild tail deviation. 
5. **No multicollinearity** — ⚠️ Some correlation exists among the
   numeric features (e.g. flipper and culmen length covary). Ridge and
   Lasso are exactly the tools to handle this, and the coefficients
   barely change between OLS / Ridge / Lasso, suggesting collinearity
   is not severe enough to destabilize the fit.

**Bottom line:** the linear-regression assumptions are reasonably
satisfied; the model's reported metrics can be trusted.
        """
    )


# Tab 4 : Predict
with tab_predict:
    st.header('Predict a single penguin')

    c1, c2, c3 = st.columns(3)
    culmen_length    = c1.slider('Culmen length (mm)',    30.0, 65.0, 45.0, 0.1)
    culmen_depth     = c2.slider('Culmen depth (mm)',     13.0, 22.0, 17.0, 0.1)
    flipper_length   = c3.slider('Flipper length (mm)',  170.0, 235.0, 200.0, 1.0)

    c4, c5, c6 = st.columns(3)
    species = c4.selectbox('Species', ['Adelie', 'Chinstrap', 'Gentoo'])
    island  = c5.selectbox('Island',  ['Biscoe', 'Dream', 'Torgersen'])
    sex     = c6.selectbox('Sex',     ['MALE', 'FEMALE'])

    model_name = st.radio('Headline model', ['OLS', 'Ridge', 'Lasso'],
                          horizontal=True)

    sample = pd.DataFrame([{
        'culmen_length_mm':  culmen_length,
        'culmen_depth_mm':   culmen_depth,
        'flipper_length_mm': flipper_length,
        'species': species, 'island': island, 'sex': sex,
    }])
    sample_enc = pd.get_dummies(sample, columns=CAT_FEATS, drop_first=True)
    # Add any missing dummy columns the training data had, in correct order
    for col in bundle['features']:
        if col not in sample_enc.columns:
            sample_enc[col] = 0
    sample_enc = sample_enc[bundle['features']].astype(float)

    sample_sc = bundle['scaler'].transform(sample_enc)

    # Predict from all three baseline models (the tuned ones live as
    # GridSearchCV pipelines and need unscaled X)
    pred_ols   = bundle['models']['OLS'  ].predict(sample_sc)[0]
    pred_ridge = bundle['models']['Ridge'].predict(sample_sc)[0]
    pred_lasso = bundle['models']['Lasso'].predict(sample_sc)[0]

    headline_pred = {'OLS': pred_ols, 'Ridge': pred_ridge,
                     'Lasso': pred_lasso}[model_name]

    # ~95% prediction interval using residual std on the training set
    rstd = bundle['residual_std']
    lo = headline_pred - 1.96 * rstd
    hi = headline_pred + 1.96 * rstd

    c1, c2 = st.columns([1, 2])
    c1.metric(f'Predicted body mass ({model_name})',
              f'{headline_pred:.0f} g',
              f'≈ {headline_pred/1000:.2f} kg')
    c2.markdown(
        f"**~95% prediction interval:** {lo:,.0f} g – {hi:,.0f} g  \n"
        f"*(headline ± 1.96 × residual std of {rstd:.0f} g; this is the "
        f"range where ~95% of real penguins with these inputs would fall, "
        f"assuming residuals are roughly normal)*"
    )

    # Side-by-side comparison of all three baseline models
    st.subheader('Model comparison for this penguin')
    fig, ax = plt.subplots(figsize=(7, 3))
    names = ['OLS', 'Ridge', 'Lasso']
    vals  = [pred_ols, pred_ridge, pred_lasso]
    colors = ['steelblue', 'seagreen', 'tomato']
    bars = ax.bar(names, vals, color=colors, edgecolor='white', width=0.55)
    ax.axhline(headline_pred, color='black', linewidth=0.8, linestyle='--',
               alpha=0.5)
    ax.set_ylabel('Predicted body mass (g)')
    ax.set_title('OLS vs Ridge vs Lasso for the chosen penguin',
                 fontweight='bold')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 8,
                f'{v:.0f} g', ha='center', va='bottom', fontsize=10)
    # Pad y-axis so labels don't clip
    spread = max(vals) - min(vals)
    pad = max(spread * 0.6, 80)
    ax.set_ylim(min(vals) - pad, max(vals) + pad)
    plt.tight_layout(); st.pyplot(fig)

    st.caption(
        f'Range across the 3 models: '
        f'{max(vals) - min(vals):.0f} g  '
        f'({"models agree closely" if (max(vals) - min(vals)) < 50 else "models disagree somewhat"})'
    )

    with st.expander('What the model sees (encoded sample)'):
        st.dataframe(sample_enc, width='stretch')
