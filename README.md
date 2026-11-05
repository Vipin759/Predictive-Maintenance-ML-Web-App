# Machine Failure Prediction

A machine learning project that predicts whether a machine will fail from its operating conditions, with a SQL analysis and a small web app.

**Live demo:** ADD YOUR RENDER LINK HERE

## Overview

- **Data:** AI4I 2020 Predictive Maintenance Dataset (UCI Machine Learning Repository, S. Matzka, 2020): 10,000 synthetic records, 3.39% failures.
- **Inputs:** air temperature, process temperature, rotational speed, torque, tool wear, product quality (L, M, H).
- **Model:** Random Forest (200 trees) with class weighting to handle the imbalance.
- **Leakage avoided:** the failure-mode columns (TWF, HDF, PWF, OSF, RNF) and the ID columns are not used as inputs.
- **SQL analysis:** SQLite with GROUP BY, HAVING, CASE and subqueries.
- **Web app:** Flask form with input validation that returns a failure probability.

## Results (20% stratified test set, 2,000 records, 68 failures)

| Metric (failure class) | Value |
|---|---|
| Precision | 0.750 |
| Recall | 0.706 |
| F1 | 0.727 |

Accuracy is 98.2%, but always predicting "no failure" already scores 96.6%, so precision, recall and F1 are the metrics that matter here.

**Findings**
- Torque, rotational speed and tool wear are the most influential inputs.
- Failure rate by quality: L 3.92%, M 2.77%, H 2.09%.
- Failure rate jumps from about 2.3% to 15.4% once tool wear reaches 200 minutes.

## Run locally

```
pip install -r requirements.txt
python train_model.py     # trains and saves model.joblib (the app also does this automatically if the file is missing)
python app.py
```
Open http://127.0.0.1:5001

On Windows use `python` and `pip`; on macOS or Linux use `python3` and `pip3`.

## Files

| File | Purpose |
|---|---|
| `Machine_Failure_Prediction.ipynb` | Exploration, model, evaluation, SQL analysis |
| `train_model.py` | Trains the model and saves `model.joblib` |
| `app.py` | Flask web app |
| `ai4i2020.csv` | Dataset |
| `requirements.txt` | Python packages |

## Limitations

The data is synthetic, the model uses default settings without tuning, and the app is a demonstration rather than production software.
