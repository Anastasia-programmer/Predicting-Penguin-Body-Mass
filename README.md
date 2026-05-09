# Data Analysis Homework — Penguin Dataset

Hello Madame , sorry for late submession i had to learn the libraries , you can view the analysis in two ways:

1. **Terminal (text + plots in pop-up windows)**  
   Run 
   python app.py

2. **Streamlit , (my recommendation )**  
   A web app shows the text and the plots in a scrollable format. To run it:
     pip install streamlit
     streamlit run app_streamlit.py
   The app will open in your browser. I recommend this option for a clearer, scrollable view.

## 1. Dataset

- **Source:** Kaggle (Penguin size dataset ).
- **File:** `data/penguins_size.csv` (species, island, culmen_length_mm, culmen_depth_mm, flipper_length_mm, body_mass_g, sex).

---

## 2. Analysis (by comprehension only)

### What is the type of the dataset?

- **Structured / tabular dataset:** rows = individuals (penguins), columns = variables (features and identifiers).
- **Type:** Observational, cross-sectional (one record per penguin, no time dimension).

### What is the type of data in it?

- **Categorical (qualitative):**
  - **Nominal:** `species`, `island`, `sex` (categories with no order).
- **Numerical (quantitative):**
  - **Continuous (ratio):** `culmen_length_mm`, `culmen_depth_mm`, `flipper_length_mm`, `body_mass_g` (measurements with a true zero and meaningful ratios).

### What could be appropriate mathematical/statistical methods to analyse it, and why?

- **For categorical variables (e.g. species, island, sex):**
  - **Mode** — most frequent category; useful for nominal data where “average” is not defined.
  - **Frequency / count** — how many individuals per category.
  - **Proportions / percentages** — relative share of each category; good for comparing balance across groups.
  - **Bar charts** — to compare counts or proportions across categories.
- **For numerical variables (e.g. body mass, culmen length):**
  - **Mean** — central tendency when distribution is roughly symmetric.
  - **Median** — robust central tendency, especially with skew or outliers.
  - **Min, max, range** — spread and extremes.
  - **Standard deviation and variance** — spread around the mean.
  - **Quantiles (e.g. quartiles)** — distribution shape and spread.
  - **Skewness** — asymmetry of the distribution.
  - **Histograms** — to visualize distribution shape and spread.

---

## 3. What we did (implementation)

- **Libraries used:** Pandas, Matplotlib.
- **Data preparation:** Read CSV, removed missing values (`dropna()`), removed duplicates (`drop_duplicates()`).
- **Exploration:** First 5 and first 10 rows (`head()`), dataset size (`shape`).
- **Categorical study (species):**
  - Number of unique species (`nunique()`), list of categories (`unique()`).
  - Frequency of each category (`value_counts()`).
  - Proportions (`value_counts(normalize=True)`).
  - Mode (`mode()`).
  - Sorted values .
- **Numerical study (body mass in g):**
  - First/last rows (`head()`, `tail()`).
  - Central tendency: mean, median, mode.
  - Min, max, range.
  - Spread: standard deviation, variance.
  - Quantiles (25%, 50%, 75%).
  - Skewness.
  - **Graph 2:** Histogram of body mass with mean and median lines.

---

## 4. Interpretation of the graphs

### Graph 1: Species count (bar chart)

- Adelie penguins have the highest count in the dataset.
- Gentoo penguins are the second most common species.
- Chinstrap penguins have the lowest count.
- The dataset is unevenly distributed across species.

### Graph 2: Body mass distribution (histogram)

- Pe nguin body mass ranges widely from about 2700 g to 6200 g.
- Most penguins cluster in the middle mass range (around 3300–4800 g).
- The mean body mass is higher than the median, indicating right skew.
- A few heavier penguins increase the average body mass.

## TP1: Interpretation of the statistics

*Numerical variable: body mass (g).*

- **Mean** — The typical (average) body mass of a penguin in our sample.
- **Median** — The middle body mass—half of the penguins weigh less, half weigh more; not pulled by a few very light or very heavy penguins.
- **Mode** — The body mass that appears most often among our penguins.
- **Std** — How much penguin body masses tend to differ from the average—whether weights are similar or spread out.
- **Variance** — How spread out body masses are (same idea as std, in squared units).
- **Min / Max** — The lightest and heaviest penguin in our sample.
- **Range** — The full span of body masses from lightest to heaviest penguin.
- **Quartiles** — How body mass is spread in the lighter quarter, middle half, and heavier quarter of our penguins.
- **Skewness** — Whether body mass is symmetric or has a long tail (e.g. more heavy or more light penguins).
- **Count** — How many penguins we have a body mass for in our sample.

*Categorical variable: species.*

- **Mode** — The species we have the most of in our sample.
- **Frequency** — How many penguins belong to each species (Adelie, Chinstrap, Gentoo).
- **Proportions** — What share of our sample is each species (so we can compare species representation).

---

## 5. Final project — Linear Regression on body mass

For the final submission i extended the homework into a full linear regression project. The goal is to predict `body_mass_g` from the other features (culmen length, culmen depth, flipper length, species, island, sex).

### What i added

- A new analysis script `notebook_analysis.py` with the full pipeline (loading, EDA, preprocessing, modeling, evaluation, regularization, key takeaways).
- A new Streamlit dashboard `app_lab.py` with 5 tabs (Methodology, Overview, EDA, Model & Evaluation, Predict).
- A `requirements.txt` with all the libraries.
- The original `app.py` and `app_streamlit.py` are still there for reference (the homework version, not the final one).

### Pipeline (CRISP-DM)

1. **Business understanding** — predict penguin body mass from quick field measurements (flipper, culmen, species, sex). Useful because weighing the bird in the field is the hardest part.
2. **Data understanding** — EDA, correlation matrix, missing values check.
3. **Data preparation** — drop NA rows (only 11/344, ~3%), drop the one row where `sex == "."`, drop duplicates, one-hot encode species/island/sex with `drop_first=True`, 80/20 train/test split, `StandardScaler` fitted on the training set only.
4. **Modeling** — `LinearRegression` (OLS) as baseline, then `Ridge` (L2) and `Lasso` (L1). For Ridge and Lasso i tuned alpha with `GridSearchCV` (5-fold CV) over the grid `[0.001, 0.01, 0.1, 1, 10, 100]`. To avoid leakage i wrapped the scaler and the model in a `Pipeline`, so the scaler is re-fitted inside each CV fold.
5. **Evaluation** — MAE, MSE, RMSE, R², train vs test R² gap, 5-fold cross validation, residual plot, Q-Q plot, Shapiro-Wilk normality test, check on the 5 OLS assumptions.
6. **Deployment** — Streamlit dashboard so it can be used by someone who is not technical.

### Linear regression — interpretation (IMPORTANT)

Because every feature is standardized before fitting, each coefficient says: "if this feature increases by one standard deviation, by how many grams does predicted body mass change, all else equal".

OLS coefficients i got:

| Feature              | Coef (g) | Reading |
|----------------------|---------:|---------|
| `species_Gentoo`     |    +502  | A Gentoo penguin is ~500 g heavier than an Adelie at equal measurements |
| `flipper_length_mm`  |    +230  | Largest numeric effect, matches the EDA scatter plot |
| `sex_MALE`           |    +183  | Males are ~180 g heavier than females at equal measurements |
| `culmen_depth_mm`    |    +156  | Positive once species is controlled for (Simpson's paradox below) |
| `culmen_length_mm`   |     +91  | Modest positive contribution |
| `species_Chinstrap`  |     -92  | Slightly lighter than the Adelie reference |
| `island_Torgersen`   |     -21  | Negligible after the rest is accounted for |
| `island_Dream`       |      -4  | Negligible after the rest is accounted for |

**Simpson's paradox on culmen depth.** The raw correlation between `culmen_depth_mm` and `body_mass_g` is **−0.47** (negative), but the OLS coefficient is **+156 g per std** (positive). The reason is that Gentoo penguins are heavier *and* have shallower culmens, so if you don't control for species the confounder flips the sign. Adding species dummies in the regression resolves it.

### Final results

On the held-out test set (67 penguins):

| Model              | MAE (g) | RMSE (g) | R²     |
|--------------------|--------:|---------:|-------:|
| OLS                | 237.08  | 298.18   | 0.8629 |
| Ridge (α=1.0)      | 236.01  | 297.39   | 0.8636 |
| Lasso (α=1.0)      | 236.88  | 297.79   | 0.8633 |
| Ridge tuned (α=1)  | 236.01  | 297.39   | 0.8636 |
| Lasso tuned (α=1)  | 236.88  | 297.79   | 0.8633 |

Ridge tuned came out best but only by a tiny margin. All 5 models land within 1 g of MAE and 0.001 of R². For practical use any of them works.

The headline: **the model predicts body mass to within ~236 g on average (about 5.6% of the mean weight) and explains 86.4% of the variance on data it has never seen.** Train/test gap is 0.015 and 5-fold CV is stable (mean R² 0.873, std 0.009), so the model generalizes and is not overfitting.

A few things i noticed:
- `flipper_length_mm` is the strongest single numeric predictor, but `species_Gentoo` is the largest coefficient overall (+502 g vs Adelie). Species identity matters more than any one measurement.
- Lasso did not zero out any feature at α=1, so none of the 8 features is redundant enough to drop. The dataset is small but well-behaved, so OLS is already close to optimal and regularization does not change much here.
- Tuning showed that the CV R² curve is flat for small alphas and only drops at α=100, where the penalty becomes too strong.

### How to run the final version

```
pip install -r requirements.txt
streamlit run app_lab.py
```

Or for the analysis script with plots in pop-up windows:

```
python notebook_analysis.py
```

### What is in each Streamlit tab

- **Methodology** — short write-up of CRISP-DM, why i split 80/20, why i scale features, what a Pipeline is and why it prevents leakage, and a sentence on the bias-variance tradeoff.
- **Overview** — dataset preview, missing-value report, cleaning summary.
- **EDA** — distribution of `body_mass_g`, scatter plots vs each numeric feature, box plots by category, correlation heatmap.
- **Model & Evaluation** — the metrics table (MAE, MSE, RMSE, R²), the OLS coefficient bar chart, the actual-vs-predicted and residual plots, the alpha-vs-CV-R² tuning curves, a Q-Q plot of residuals with a Shapiro-Wilk test, and a check on the 5 OLS assumptions.
- **Predict** — sliders and selectors for a custom penguin. Returns the predicted body mass, a 95% prediction interval (predicted ± 1.96 × residual std on training, ≈ 281 g), and a small bar chart comparing OLS / Ridge / Lasso predictions for the same input.
