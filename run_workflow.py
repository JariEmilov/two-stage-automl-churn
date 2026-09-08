"""Two-stage AutoML workflow (Kashef methodology): LazyPredict screen -> FLAML optimize.

Dataset: telco_customer_churn.csv — IBM's public "Telco Customer Churn" sample data
(7,043 rows, target `Churn`, ~27% churn), pulled directly from IBM's own public GitHub
(raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d). Swapped 2026-09-08 to
replace the earlier course-derived dataset of unconfirmed public/redistribution status.
Artifacts: outputs/*.csv, outputs/metrics.json, outputs/figures/*.png, models/*.joblib
"""
import json, time, warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
SEED = 42
ROOT = Path(__file__).parent
OUT = ROOT / "outputs"; FIG = OUT / "figures"; MODELS = ROOT / "models"
for d in (OUT, FIG, MODELS): d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- Part 1: prep + EDA
df = pd.read_csv(ROOT / "data" / "telco_customer_churn.csv")
df = df.drop(columns=["customerID"])
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
target = "Churn"
df[target] = (df[target] == "Yes").astype(int)
print(f"rows={len(df)} churn_rate={df[target].mean():.3f}")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
df[target].value_counts().plot.bar(ax=ax[0], title="Churn distribution")
df.groupby("Contract")[target].mean().sort_values().plot.barh(ax=ax[1], title="Churn rate by contract type")
fig.tight_layout(); fig.savefig(FIG / "eda_churn.png", dpi=120); plt.close(fig)

X = df.drop(columns=[target]); y = df[target]
cat_cols = X.select_dtypes(include="object").columns.tolist()
encoders = {}
for c in cat_cols:
    le = LabelEncoder(); X[c] = le.fit_transform(X[c]); encoders[c] = le
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

# ---------------------------------------------------------------- Part 2: LazyPredict screening
from lazypredict.Supervised import LazyClassifier
t0 = time.time()
clf = LazyClassifier(verbose=0, ignore_warnings=True, predictions=False, random_state=SEED)
leaderboard, _ = clf.fit(X_train, X_test, y_train, y_test)
screen_secs = time.time() - t0
leaderboard = leaderboard.sort_values("ROC AUC", ascending=False)
leaderboard.to_csv(OUT / "lazypredict_results.csv")
print(f"\nLazyPredict: {len(leaderboard)} models in {screen_secs:.1f}s. Top 10:")
print(leaderboard.head(10))

best_screen_name = leaderboard.index[0]
best_screen_auc = float(leaderboard.iloc[0]["ROC AUC"])

# ---------------------------------------------------------------- Part 3: FLAML optimization
# ponytail: estimator list fixed to the tree/boosting families that dominate churn
# screening; widen if a non-tree family ever cracks the leaderboard top 5.
from flaml import AutoML
automl = AutoML()
t0 = time.time()
automl.fit(X_train.values, y_train.values, task="classification", metric="roc_auc",
           time_budget=180, estimator_list=["lgbm", "xgboost", "rf", "extra_tree"],
           eval_method="cv", n_splits=5, seed=SEED, verbose=1)
opt_secs = time.time() - t0
proba = automl.predict_proba(X_test.values)[:, 1]
pred = (proba >= 0.5).astype(int)

metrics = {
    "screen_models": int(len(leaderboard)),
    "screen_seconds": round(screen_secs, 1),
    "screen_best_model": best_screen_name,
    "screen_best_roc_auc": round(best_screen_auc, 4),
    "flaml_seconds": round(opt_secs, 1),
    "flaml_best_estimator": automl.best_estimator,
    "flaml_best_config": automl.best_config,
    "flaml_cv_roc_auc": round(1 - automl.best_loss, 4),
    "test_roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
    "test_f1": round(float(f1_score(y_test, pred)), 4),
    "test_accuracy": round(float(accuracy_score(y_test, pred)), 4),
    "test_precision": round(float(precision_score(y_test, pred)), 4),
    "test_recall": round(float(recall_score(y_test, pred)), 4),
    "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
}
print(f"\nFLAML best: {automl.best_estimator} test ROC-AUC={metrics['test_roc_auc']}"
      f" (screening baseline {best_screen_auc:.4f})")

# threshold sweep (business tuning)
rows = []
for th in (0.3, 0.4, 0.5, 0.6, 0.7):
    p = (proba >= th).astype(int)
    rows.append({"threshold": th,
                 "precision": round(float(precision_score(y_test, p)), 4),
                 "recall": round(float(recall_score(y_test, p)), 4),
                 "f1": round(float(f1_score(y_test, p)), 4)})
pd.DataFrame(rows).to_csv(OUT / "threshold_sweep.csv", index=False)

# ROC curve + confusion matrix figures
fpr, tpr, _ = roc_curve(y_test, proba)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(fpr, tpr, label=f"AUC={metrics['test_roc_auc']}"); ax[0].plot([0, 1], [0, 1], "--")
ax[0].set_title("ROC curve (test)"); ax[0].legend()
cm = np.array(metrics["confusion_matrix"])
ax[1].imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax[1].text(j, i, cm[i, j], ha="center", va="center")
ax[1].set_title("Confusion matrix"); ax[1].set_xlabel("predicted"); ax[1].set_ylabel("actual")
fig.tight_layout(); fig.savefig(FIG / "roc_confusion.png", dpi=120); plt.close(fig)

# feature importance from the tuned model
est = automl.model.estimator
imp = pd.Series(est.feature_importances_, index=X.columns).sort_values(ascending=False)
imp.to_csv(OUT / "feature_importance.csv", header=["importance"])
print("\nTop 5 features:"); print(imp.head(5))

# ---------------------------------------------------------------- Part 4: package
joblib.dump(automl, MODELS / "churn_model.joblib")
joblib.dump(encoders, MODELS / "label_encoders.joblib")
joblib.dump(X.columns.tolist(), MODELS / "feature_names.joblib")
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")

assert metrics["test_roc_auc"] > 0.75, "model below minimum bar"
print("\nWORKFLOW COMPLETE — artifacts in models/ and outputs/")
