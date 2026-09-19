# Personalized Asthma Trigger Prediction from Wearable & Environmental Data

Machine learning pipeline that predicts a patient's daily asthma-trigger label from smartwatch heart rate, air quality and pollen data, and adapts to an individual patient through an online feedback loop.

**Stack:** Python · Pandas · scikit-learn · XGBoost · SHAP · Matplotlib / Seaborn

---

## Overview

Asthma triggers differ from patient to patient and day to day, and the signals that hint at them are spread across separate sources. This project:

1. Merges questionnaire, environmental and smartwatch data into one patient-day dataset.
2. Benchmarks 9 classifiers using a **patient-wise split**, so the test set contains only patients the model has never seen.
3. Simulates **online personalization**: after each day's confirmed label, the model is retrained with that patient's data up-weighted.
4. Explains predictions with SHAP and benchmarks inference latency.

## Dataset

The AAMOS-00 dataset, provided as five CSV files: `dailyquestionnaire`, `environment`, and `smartwatch1/2/3`. *[Add dataset source / citation link here.]*

| Item | Value |
|---|---|
| Patients | 20 |
| Patient-day records (after merge and cleaning) | 1,205 (from 1,602 before dropping missing values) |
| Class balance | 965 positive / 240 negative (about 80 / 20) |
| Target | `daily_triggers`: binary label derived from the daily questionnaire (0 if trigger code `1` is reported, otherwise 1) *[describe what code 1 means]* |
| Features | Temperature, pressure, CO, NO, NO₂, O₃, SO₂, PM2.5, PM10, NH₃, daily mean heart rate, one-hot encoded grass / tree / weed pollen levels |

## Pipeline

1. **Data integration:** join the environment, smartwatch and questionnaire tables on `user_key` and `date`.
2. **Cleaning:** concatenate the three smartwatch files, drop heart-rate readings below 40 bpm (sensor artifacts), aggregate to daily mean heart rate, and drop rows with missing values.
3. **Feature engineering:** one-hot encode pollen levels, standardize continuous features, and drop redundant or leakage-prone columns (raw symptom and inhaler fields, AQI, min/max temperature, wind direction).
4. **Patient-wise split:** 14 patients for training (843 records), 6 for testing (362 records), with zero patient overlap verified in code.
5. **Modelling:** Random Forest, XGBoost, Logistic Regression, Gradient Boosting, MLP, Bernoulli NB, Gaussian NB, Decision Tree and Bagging.
6. **Personalization:** a feedback loop that retrains the model after each day using the patient's confirmed labels (weight ×4).
7. **Interpretability and deployment checks:** SHAP (permutation explainer) and inference latency benchmarks.

## Results

### Population models on unseen patients (362 test records)

Baseline for reference: predicting "1" for every record gives accuracy 0.801 and F1 0.890.

| Model | F1 | Precision | Recall | AUPRC | AUROC |
|---|---|---|---|---|---|
| Bernoulli NB | 0.897 | 0.838 | 0.966 | 0.895 | 0.724 |
| Logistic Regression | 0.894 | 0.808 | 1.000 | 0.829 | 0.623 |
| MLP | 0.892 | 0.807 | 0.997 | 0.834 | 0.638 |
| Random Forest | 0.890 | 0.801 | 1.000 | 0.878 | 0.639 |
| Bagging | 0.883 | 0.806 | 0.976 | 0.806* | 0.516 |
| Gradient Boosting | 0.861 | 0.811 | 0.917 | 0.846 | 0.614 |
| XGBoost | 0.855 | 0.795 | 0.924 | 0.768 | 0.446 |
| Gaussian NB | 0.851 | 0.867 | 0.834 | 0.873 | 0.724 |
| Decision Tree | 0.770 | 0.778 | 0.762 | 0.777 | 0.435 |

\*Bagging's AUPRC was computed from hard predictions rather than probabilities.

**Takeaway:** population-level models roughly match the majority-class baseline on F1, and AUROC is at most 0.72. Predicting triggers for a brand-new patient from environmental and heart-rate data alone is hard with this dataset, which motivates the personalization experiment below.

### Personalization (proof of concept, one held-out patient, 25 days)

The population Bagging model was compared with a version retrained after each day's confirmed label.

| Setting | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Population model | 0.24 | 0.18 | 0.80 | 0.30 |
| Personalized, original day order | 0.48 | 0.17 | 0.40 | 0.24 |
| Personalized, 4 shuffled orders | 0.56 – 0.68 | 0.00 – 0.13 | 0.00 – 0.20 | 0.00 – 0.15 |

Accuracy improved after personalization, but minority-class recall and F1 did not improve consistently. This is a single patient with 25 records, so treat it as an early signal rather than a validated result.

### Inference benchmark (Bagging model, 500 runs per case)

| Case | Median latency | p95 latency |
|---|---|---|
| 1 record | 15.1 ms | 56.2 ms |
| 10 records | 7.2 ms | 9.2 ms |
| 100 records | 8.6 ms | 15.0 ms |
| 362 records | 9.1 ms | 11.5 ms |

Model size is 0.49 MB, load time is 0.025 s, and batch throughput is about 38,000 records/sec.

## Limitations and Future Work

- **Patient ID as a feature:** `user_key` is currently included as a feature and dominates feature importance and SHAP. For unseen patients it carries no meaningful signal, so removing it (and fitting the scaler on training data only) is the next step.
- **Class imbalance:** the test set is about 80% positive, so accuracy and F1 look high even for trivial predictions. Future work: add a dummy-classifier baseline, report balanced accuracy, and use AUROC/AUPRC relative to class prevalence.
- **Validation:** cross-validation is currently row-wise. Patient-grouped CV (`GroupKFold`) would give a more honest estimate.
- **Small cohort:** 20 patients, with several having very few records. Conclusions, especially the personalization result, need more patients to confirm.
- **Tuning:** no systematic hyperparameter search was run.

## Getting Started

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install numpy pandas matplotlib seaborn scikit-learn xgboost shap joblib
```

1. Place the five `anonym_aamos00_*.csv` files in the project folder and update the file paths in the first data-loading cell (they currently point to `/content/`, as used in Google Colab).
2. Open `Asthma_project.ipynb` in Jupyter or Colab and run the cells in order.

## Repository Structure

```
├── Asthma_project.ipynb        # Full pipeline: preprocessing, modelling, personalization, SHAP, benchmarks
├── master_dataset.csv          # Merged patient-day dataset (generated by the notebook)
├── *.joblib                    # Saved models (generated by the notebook)
└── README.md
```

## Author

**[Your Name]** · [LinkedIn](https://linkedin.com/in/your-profile) · [Email](mailto:you@example.com)
