# Heart Disease Risk Screening

A machine learning project that estimates heart disease risk from clinical measurements, built on the UCI Heart Disease dataset. It compares three models, tunes the decision threshold for a screening use case, tests generalization across hospitals, explains predictions with SHAP, and wraps the result in a Streamlit app.

> **Educational demo. Not a diagnostic tool and not medical advice.**

![App screenshot](docs/app.png)

## Why this project

Most heart disease classifiers stop at "90% accuracy". For screening, that is the wrong goal: missing a sick patient is far worse than a false alarm. This project is built around that idea, so it reports recall, precision and confusion matrices, chooses a threshold deliberately, and checks whether the model holds up on patients from other hospitals.

## Data

[UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease), four sites:

| Site | Patients |
|---|---|
| Cleveland | 303 |
| Hungarian | 294 |
| Switzerland | 123 |
| VA Long Beach | 200 |

The original target (`num`) ranges from 0 to 4. I binarized it: 0 means no disease, 1 to 4 means disease.

Data issues handled:
- Missing values are marked `?` in the raw files.
- `chol = 0` and `trestbps = 0` are placeholders for missing values, not real readings. Switzerland has no usable cholesterol data at all.
- `ca`, `thal` and `slope` are almost entirely missing outside Cleveland.
- Class balance differs a lot by site (Switzerland has only 8 healthy patients out of 123).

All imputation, scaling and encoding happen inside a scikit-learn `Pipeline`, so nothing from the test data leaks into training.

## Results

### 1. Model comparison (Cleveland, 5-fold CV repeated 5 times)

| Model | Recall | Precision | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.781 | 0.856 | 0.913 |
| Random Forest | 0.776 | 0.835 | 0.907 |
| XGBoost | 0.773 | 0.820 | 0.888 |

The three models are effectively tied. On a dataset of about 300 patients, the more complex models gave no measurable benefit, so I used logistic regression: it is the simplest, it scored marginally best, and it is easy to explain.

A single 61-patient test split gave a recall of 0.93 for the same model. Cross-validation puts it near 0.78, so the single split was optimistic. This is why I rely on repeated cross-validation.

### 2. Choosing a threshold

By default a model predicts "sick" above a probability of 0.5. For screening I lowered it. Out-of-fold predictions for all 303 Cleveland patients (139 sick):

| Threshold | Sick patients missed | False alarms | Recall | Precision |
|---|---|---|---|---|
| 0.50 | 29 | 17 | 0.79 | 0.87 |
| 0.40 | 23 | 27 | 0.83 | 0.81 |
| 0.30 | 17 | 44 | 0.88 | 0.74 |
| 0.25 | 13 | 53 | 0.91 | 0.70 |
| 0.20 | 10 | 59 | 0.93 | 0.69 |

I chose **0.30**: it catches about 88% of sick patients while keeping precision near 74%. Below 0.25 the extra false alarms cost a lot for small gains. The right threshold depends on how bad a miss is compared with a false alarm, so this is a judgment call, not a universal answer.

### 3. Does it generalize? (train on Cleveland, test on other hospitals)

Because `ca`, `thal` and `slope` are mostly missing elsewhere, this test uses 10 features without them.

| Site | ROC-AUC | Recall at 0.5 | Recall at 0.3 | Healthy patients flagged at 0.3 |
|---|---|---|---|---|
| Hungarian | 0.878 | 0.78 | 0.87 | 62 of 188 |
| Switzerland | 0.752 | 0.77 | 0.89 | 6 of 8 |
| VA | 0.738 | 0.85 | 0.91 | 35 of 51 |

- Hungarian is the encouraging result: a model trained only on Cleveland still separates sick from healthy well at a different hospital.
- VA looks good on recall, but about 69% of healthy VA patients are flagged, so part of that recall comes from flagging many people.
- Precision is not compared across sites because it depends on how common disease is in each group.
- Switzerland has too few healthy patients for its ROC-AUC or false alarm figures to be reliable.
- A threshold tuned on one hospital does not transfer cleanly: recall holds up, but the false alarm cost varies a lot.

### 4. What drives the predictions (SHAP)

![SHAP summary](docs/shap_summary.png)

- Strongest drivers: number of major vessels with blockage (`ca`), thallium stress test result (`thal`), ST depression (`oldpeak`), and maximum heart rate (`thalach`). Directions match clinical expectations.
- Age and cholesterol have small effects within normal ranges. Very extreme values still push the score up, because logistic regression keeps adding risk as a value grows.
- "Asymptomatic" chest pain (`cp = 4`) pushes strongly toward disease. This is a quirk of the data, not a medical rule: these patients were referred for angiography, so people without symptoms who ended up in the study were often sick.

## Limitations

- **Small data.** About 300 patients in the training set, so every estimate carries wide uncertainty.
- **Threshold chosen on the same data it is measured on**, so the 0.30 numbers are slightly optimistic.
- **Cross-site test uses fewer features** (no `ca`, `thal`, `slope`), so it is not directly comparable to the Cleveland cross-validation.
- **Needs clinical test results.** The strongest features come from imaging and a stress test, so this is built for clinical staff entering results, not for the public to self-check.
- **Additive model.** A patient with one alarming value and otherwise normal results can still score low.
- **Not clinically validated.** The data comes from 1980s referral populations and may not reflect other groups. It has not been checked for performance differences by sex or age.

## Run it locally

```bash
git clone https://github.com/<your-username>/heart-disease-risk-screening.git
cd heart-disease-risk-screening
python -m venv venv
venv\Scripts\activate        # Windows (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

To reproduce the analysis, open `notebook.ipynb` and run it top to bottom. It also regenerates `model.joblib`.

## Project structure

```
├── app.py              Streamlit app
├── notebook.ipynb      Data cleaning, model comparison, threshold, cross-site test, SHAP
├── model.joblib        Trained pipeline (logistic regression on Cleveland, 13 features)
├── requirements.txt
├── data/heart+disease/ UCI files
└── docs/               Screenshots
```

## Tech

Python, pandas, scikit-learn, XGBoost, SHAP, Streamlit.

## Citation

Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). *Heart Disease* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X (CC BY 4.0)

## Author

Faadhil Mohiadeen, Computer Science (Intelligent Computing), Universiti Sains Malaysia.
