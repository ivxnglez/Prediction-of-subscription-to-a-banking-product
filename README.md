# Predicting term-deposit subscription

A supervised learning project for the Machine Learning course at UC3M: given a bank client's profile and the history of the marketing calls made to them, predict whether they will subscribe to a term deposit (`deposit` = yes/no). It covers the whole workflow: EDA, preprocessing in scikit-learn pipelines, model selection with hyperparameter tuning, an honest estimate of future performance, a saved final model, and a small Streamlit app to use it.

## Data

11,000 clients, 16 input features and the target. The features look like the well-known Bank Marketing dataset: age, job, marital status, education, balance, housing/personal loans, contact type, month and day of the last contact, call duration, number of contacts in this campaign, days since the previous campaign (`pdays`), number of previous contacts and the outcome of the previous campaign.

Things the EDA found, all checked in code rather than by eye:

- Classification problem, roughly balanced: 52.5% `no`, 47.5% `yes`. So accuracy is the main metric, and the baseline to beat is 52.5%.
- Missing values only in `job` (97) and `marital` (282). `unknown` also shows up as a category in several columns.
- No constant or ID-like columns. `job` has 13 levels, the only high-cardinality feature. `education` is the only ordinal one.
- `pdays` is -1 for 74.6% of clients (never contacted before), which makes the raw number misleading. It is turned into six ordered levels (0 = never contacted, then 1 to 5 from "more than 210 days ago" up to "within the last 30 days"). The cut points came from plotting subscription rate against `pdays` rather than from a calendar-style split, because the plot didn't show the "more recent is better" pattern one might expect.

## Method

- **Evaluation.** Stratified holdout, 2/3 train and 1/3 test, with a fixed seed. The test set is touched once, at the end. Model selection and tuning use 5-fold cross-validation on the training set only.
- **Preprocessing** is done inside pipelines: median imputation and standard scaling for numeric columns, most-frequent imputation and one-hot encoding for categorical ones, and an ordinal encoding for `education`. For KNN the scaler was chosen by comparison: standard (0.799) and robust (0.797) were close, min-max was clearly worse (0.727).
- **Models.** KNN, decision trees, logistic regression, SVM, and gradient boosting as the extra task. Each one was run with default hyperparameters first, then tuned with grid search, with training times recorded and plots of how the hyperparameters affect accuracy.

Cross-validated accuracy on the training set:

| Model | Default | Tuned | Best settings | Tuning time |
|---|---|---|---|---|
| KNN | 0.795 | 0.806 | k = 20, distance weights | 0.7 s |
| Decision tree | 0.771 | 0.815 | entropy, max depth 14, min split 48 | 10 s |
| Logistic regression | 0.812 | 0.814 | C = 0.1, L2 | 15 s |
| SVM | 0.839 | 0.840 | RBF, C = 1, gamma = 0.1 | 112 s |
| Gradient boosting (extra) | 0.838 | 0.852 | 400 trees, depth 3, lr 0.1, subsample 0.8 | 392 s |

The SVM was the model picked as the final one among the four required methods. On the held-out test set it gets **84.5% accuracy** (precision 82.0%, recall 86.1%, F1 84.0%). The recall being a bit higher than precision is a reasonable bias here, since a wasted call costs far less than a missed subscriber. The final model was then retrained on all 11,000 rows.

Two smaller findings: for the linear models, the strongest coefficients were call duration, a successful previous campaign and the month (March positive, May negative); and tuning helped trees a lot but barely moved the SVM and logistic regression.

## Using the model

`mystreamlit.py` is a Streamlit form where you enter a client's data and get a yes/no prediction:

```bash
streamlit run mystreamlit.py
```

`100522281_100522287_n2.ipynb` loads the saved model, predicts the 162 unlabeled clients in `bank_competition.pkl` and writes `predicciones.csv`, plus two hand-made example clients that were also checked in the app.

## Repository layout

```
100522281_100522287_n1.ipynb   EDA, evaluation setup, all models, tuning, final model
100522281_100522287_n2.ipynb   loads the model, predicts the competition set, example clients
mystreamlit.py                 Streamlit app
modelo_final.pkl               the trained SVM pipeline
bank_09.pkl                    training data
bank_competition.pkl           unlabeled clients to predict
predicciones.csv               predictions for those clients
```

The notebooks are written in Spanish.

## Running it

```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit
```

The notebooks were run on Python 3.14; the saved model loads fine with scikit-learn 1.8. Pickled models depend on the scikit-learn version, so if loading fails, retrain by running `100522281_100522287_n1.ipynb`. Run the notebooks from the repository folder so the relative file names resolve.

## Notes and limitations

- **`duration` is a leaky feature for real use.** It is the strongest predictor, but the length of a call is only known after the call has happened, so a model that relies on it can't be used to decide whom to call. It's fine for this exercise, not for a production campaign.
- **Gradient boosting scored higher in cross-validation than the SVM** (0.852 vs 0.840) but was treated as the optional extra, so the saved model is the SVM. Its tuning grid was also much bigger.