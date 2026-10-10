# Predicting term-deposit subscription

A supervised learning project, originally a university assignment for the Machine Learning course at UC3M, published here as a personal project: given a bank client's profile and the history of the marketing calls made to them, predict whether they will subscribe to a term deposit (`deposit` = yes/no). It covers the whole workflow: EDA, preprocessing in scikit-learn pipelines, model selection with hyperparameter tuning, an honest estimate of future performance, a saved final model, and a small Streamlit app to use it.

## Data

11,000 clients, 16 input features and the target. The data file (`bank_09.pkl`) was provided by the course. The features look like the well-known Bank Marketing dataset: age, job, marital status, education, balance, housing/personal loans, contact type, month and day of the last contact, call duration, number of contacts in this campaign, days since the previous campaign (`pdays`), number of previous contacts and the outcome of the previous campaign.

Things the EDA found, all checked in code rather than by eye:

- Classification problem, roughly balanced: 52.5% `no`, 47.5% `yes`. So accuracy is the main metric, and the baseline to beat is 52.5%.
- Missing values only in `job` (97) and `marital` (282). `unknown` also shows up as a category in several columns (`job`, `education`, `contact`, `poutcome`).
- No constant or ID-like columns. `job` and `month` have 12 levels each (13 counting the missing values of `job`), the highest-cardinality categorical features. `education` is the only ordinal one.
- `pdays` is -1 for 74.6% of clients (never contacted before), which makes the raw number misleading. It is turned into six ordered levels (0 = never contacted, then 1 to 5 from "more than 210 days ago" up to "within the last 30 days"). The cut points came from plotting subscription rate against `pdays` rather than from a calendar-style split, because the plot didn't show the "more recent is better" pattern one might expect. Note that the plot was made on the whole dataset, before the train/test split.

## Method

- **Evaluation.** Stratified holdout, 2/3 train (7,333 rows) and 1/3 test (3,667 rows), with a fixed seed. The test set is touched once, at the end. Model selection and tuning use 5-fold cross-validation on the training set only (shuffled `KFold` with a fixed seed for every model except KNN, which uses scikit-learn's default stratified 5-fold).
- **Preprocessing** is done inside pipelines, except the `pdays` grouping, which is applied to the dataframe beforehand (see below). Numeric columns are standardized, categorical ones are one-hot encoded after imputing missing values (and `unknown`) with the most frequent value. The details change a little per model:
  - KNN: standard scaling and one-hot encoding for all text columns, `education` included as a plain categorical column (it did better than the ordinal version). For KNN the scaler was chosen by comparison: standard (0.799) and robust (0.797) were close, min-max was clearly worse (0.727).
  - Decision tree: no scaling, one-hot encoding only.
  - Logistic regression, SVM and gradient boosting: median imputation and standard scaling for numeric columns, one-hot encoding for nominal ones and an ordinal encoding (primary < secondary < tertiary) for `education`.
- **Models.** KNN, decision trees, logistic regression, SVM, and gradient boosting as the extra task. Each one was run with default hyperparameters first, then tuned with grid search, with training times recorded and plots of how the hyperparameters affect accuracy.

Cross-validated accuracy on the training set:

| Model | Default | Tuned | Best settings | Tuning time |
|---|---|---|---|---|
| KNN | 0.795 | 0.806 | k = 20, distance weights | 0.7 s |
| Decision tree | 0.771 | 0.815 | entropy, max depth 14, min split 48 | 10 s |
| Logistic regression | 0.812 | 0.814 | C = 0.1, L2 | 15 s |
| SVM | 0.839 | 0.840 | RBF, C = 1, gamma = 0.1 | 112 s |
| Gradient boosting (extra) | 0.838 | 0.852 | 400 trees, depth 3, lr 0.1, subsample 0.8 | 392 s |

The SVM was the model picked as the final one among the four required methods. On the held-out test set it gets **84.5% accuracy** (precision 82.0%, recall 86.1%, F1 84.0%). The recall being a bit higher than precision is a reasonable bias here, since a wasted call costs far less than a missed subscriber. The final model was then retrained on all 11,000 rows, so there is no unseen data left to measure it; the 84.5% is the estimate to quote.

Two smaller findings: for the logistic regression, the strongest coefficients were call duration, a successful previous campaign and the month (March positive, May negative); and tuning helped trees a lot but barely moved the SVM and logistic regression.

## Using the model

The saved model (`modelo_final.pkl`) is a scikit-learn pipeline that takes the original columns, **but `pdays` must already be grouped into the six levels**, because that step was done outside the pipeline when training. `notebook2.ipynb` and the app both apply it before predicting:

```python
df['pdays'] = pd.cut(df['pdays'], bins=[-2, -1, 30, 105, 180, 210, np.inf],
                     labels=[0, 5, 4, 3, 2, 1]).astype(int)
```

If you pass the raw number of days, the model still runs but gives wrong predictions, with no error. Valid values are -1 (never contacted) or a positive number of days; anything below -1 makes the grouping fail.

`mystreamlit.py` is a Streamlit form where you enter a client's data and get a yes/no prediction (the model is an SVM without probabilities, so it doesn't give a probability):

```bash
streamlit run mystreamlit.py
```

`notebook2.ipynb` loads the saved model, predicts the 162 unlabeled clients in `bank_competition.pkl` and writes `predicciones.csv`, plus two hand-made example clients that were also checked in the app.

Quick start, if you only want to try the app (see [Running it](#running-it) for the full setup with a virtual environment and the notebooks):

```bash
pip install -r requirements.txt
streamlit run mystreamlit.py
```

## Repository layout

```
notebook1.ipynb                EDA, evaluation setup, all models, tuning, final model
notebook2.ipynb                loads the model, predicts the competition set, example clients
mystreamlit.py                 Streamlit app
requirements.txt               Python dependencies
modelo_final.pkl               the trained SVM pipeline
bank_09.pkl                    training data
bank_competition.pkl           unlabeled clients to predict
predicciones.csv               predictions for those clients
```

The notebooks and the app are written in Spanish; this README is in English.

## Running it

Pickled models depend on the scikit-learn version, so `requirements.txt` pins `scikit-learn==1.8.0`, the one used to save `modelo_final.pkl`. The project was developed on Python 3.13 and 3.14; the setup below was checked on Python 3.13. Download or clone the repository and run everything from its folder so the relative file names resolve.

Create a virtual environment, install the dependencies and register it as a Jupyter kernel:

```bash
# Windows (PowerShell)
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned   # only if activation is blocked
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
python -m ipykernel install --user --name proyecto-banco --display-name "Python (proyecto-banco)"
```

Then open the notebooks and select the kernel **Python (proyecto-banco)** (in VS Code: *Select Kernel*; in Jupyter: *Kernel > Change Kernel*). A quick check that it is the right one:

```python
import sys, sklearn
print(sys.executable, sklearn.__version__)   # should point to .venv and print 1.8.0
```

`notebook2.ipynb` only needs the saved files and runs in seconds. `notebook1.ipynb` retrains everything and takes around 10 minutes, mostly the gradient boosting search (about 6.5 minutes) and the SVM search (about 2 minutes); it overwrites `modelo_final.pkl` and also writes a `modelo_final.joblib` copy. If loading the model fails with another scikit-learn version, retrain by running `notebook1.ipynb`.

## Notes and limitations

- **`duration` is a leaky feature for real use.** It is the strongest predictor, but the length of a call is only known after the call has happened, so a model that relies on it can't be used to decide whom to call. It's fine for this exercise, not for a production campaign.
- **Gradient boosting scored higher in cross-validation than the SVM** (0.852 vs 0.840) but was treated as the optional extra, so the saved model is the SVM. Its tuning grid was also much bigger. Two of its best values (400 trees, and min split 48 for the decision tree) sit at the edge of the searched grid, so a wider search might improve them.
- **`unknown` values are imputed.** The pipelines replace `unknown` with the most frequent value of each column, so for example `unknown` in `poutcome` (no previous campaign, 75% of rows) becomes `failure`. A quick check outside the notebooks, keeping `unknown` as one more category, gave a higher SVM accuracy (about 0.850 in cross-validation and 0.848 on the test set instead of 0.840 and 0.845). It was not adopted in the final model.
- **The comparison between models is approximate.** KNN is cross-validated with stratified folds and the other models with shuffled, non-stratified ones, so their folds are not exactly the same.