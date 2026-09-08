# Churn AutoML demo (Kashef two-stage workflow)

Working demo of the `data-scientist` agent methodology: LazyPredict screening
(27 algorithms, ~13s) → FLAML Bayesian optimization (180s budget, 5-fold CV)
→ Streamlit dashboard. Dataset: `data/telco_customer_churn.csv` — IBM's public
**"Telco Customer Churn"** sample data (7,043 rows), pulled directly from IBM's
own public GitHub:
`https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`.
(Swapped 2026-09-08 from an earlier course-derived dataset of unconfirmed public
license, before this demo was made public.) Spec in `context.md`.

## Results (2026-09-08 run)

| Stage | Model | ROC-AUC (test) |
|---|---|---|
| Screening best | LogisticRegression (defaults) | 0.840 |
| FLAML optimized | XGBoost (136 est., 8 leaves, lr 0.033) | **0.849** |

Top features: OnlineSecurity, Contract, TechSupport, InternetService, tenure.

## Run it

```bash
py -3.11 -m venv venv
./venv/Scripts/python.exe -m pip install -r requirements.txt
./venv/Scripts/python.exe -m streamlit run dashboard.py --server.port 8511
```

To retrain from scratch instead of using the committed `models/`/`outputs/` artifacts,
also install `matplotlib`, `tqdm`, and `lazypredict` (`pip install --no-deps lazypredict`
— its own pins conflict with the pins above), then run `run_workflow.py` (~4 min).

Dashboard sections: Overview / Algorithm Leaderboard / Model Selection (funnel +
tuned hyperparameters) / Performance (ROC, confusion, threshold sweep) /
Insights / Live Predictor.
