"""Streamlit dashboard for the two-stage AutoML churn workflow (Kashef methodology).

Sections: Overview / Algorithm Leaderboard / Model Selection / Performance /
Insights / Live Predictor. Reads artifacts produced by run_workflow.py.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_curve

ROOT = Path(__file__).parent
# dataviz reference palette (light mode)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_LIGHT = "#9dc3ee"
GOOD, WARN, BAD = "#008300", "#eda100", "#c5221f"
INK2 = "#52514e"

st.set_page_config(page_title="Churn AutoML — Two-Stage Workflow", layout="wide")


@st.cache_data
def load_artifacts():
    metrics = json.loads((ROOT / "outputs" / "metrics.json").read_text())
    lb = pd.read_csv(ROOT / "outputs" / "lazypredict_results.csv", index_col=0)
    sweep = pd.read_csv(ROOT / "outputs" / "threshold_sweep.csv")
    imp = pd.read_csv(ROOT / "outputs" / "feature_importance.csv", index_col=0)
    df = pd.read_csv(ROOT / "data" / "telco_customer_churn.csv")
    df = df.drop(columns=["customerID"])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return metrics, lb, sweep, imp, df


@st.cache_resource
def load_model():
    model = joblib.load(ROOT / "models" / "churn_model.joblib")
    encoders = joblib.load(ROOT / "models" / "label_encoders.joblib")
    features = joblib.load(ROOT / "models" / "feature_names.joblib")
    return model, encoders, features


metrics, lb, sweep, imp, df = load_artifacts()
model, encoders, features = load_model()

st.sidebar.title("Churn AutoML")
page = st.sidebar.radio("Section", [
    "Overview", "Algorithm Leaderboard", "Model Selection",
    "Performance", "Insights", "Live Predictor"])
st.sidebar.caption("Two-stage workflow: LazyPredict screening → FLAML optimization "
                   "(methodology: Mark Kashef, ML Mini Series) · dataset: IBM Telco "
                   "Customer Churn (public)")

BAR_LAYOUT = dict(plot_bgcolor="white", font=dict(color="#0b0b0b"),
                  margin=dict(l=10, r=10, t=40, b=10))

if page == "Overview":
    st.title("Customer Churn — Two-Stage AutoML")
    c = st.columns(5)
    c[0].metric("Customers", f"{len(df):,}")
    c[1].metric("Churn rate", f"{df['Churn'].mean():.0%}")
    c[2].metric("Models screened", metrics["screen_models"])
    c[3].metric("Final model", metrics["flaml_best_estimator"].upper())
    c[4].metric("Test ROC-AUC", f"{metrics['test_roc_auc']:.3f}",
                delta=f"+{(metrics['test_roc_auc'] - metrics['screen_best_roc_auc']):.3f} vs screening")
    st.markdown(f"""
**The workflow in one paragraph.** {metrics['screen_models']} algorithms were screened
with default configs in **{metrics['screen_seconds']}s** (LazyPredict). The best screened
model was **{metrics['screen_best_model']}** at ROC-AUC **{metrics['screen_best_roc_auc']:.3f}**.
The dominant families then went through Bayesian hyperparameter search for
**{metrics['flaml_seconds']:.0f}s** (FLAML, 5-fold CV), which selected **{metrics['flaml_best_estimator']}**
and lifted the held-out test ROC-AUC to **{metrics['test_roc_auc']:.3f}**.
""")
    churn_by_ct = df.groupby("Contract")["Churn"].mean().sort_values()
    fig = go.Figure(go.Bar(x=churn_by_ct.values, y=churn_by_ct.index, orientation="h",
                           marker_color=BLUE, text=[f"{v:.0%}" for v in churn_by_ct.values],
                           textposition="outside", width=0.55))
    fig.update_layout(title="Churn rate by contract type (the strongest driver)",
                      xaxis_tickformat=".0%", height=300, **BAR_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

elif page == "Algorithm Leaderboard":
    st.title("Every algorithm we tested")
    st.caption(f"LazyPredict screening: {metrics['screen_models']} models in "
               f"{metrics['screen_seconds']}s, ranked by ROC-AUC (primary metric — "
               f"{df['Churn'].mean():.0%} churn makes accuracy misleading).")
    ranked = lb.sort_values("ROC AUC", ascending=True)
    fig = go.Figure(go.Bar(x=ranked["ROC AUC"], y=ranked.index, orientation="h",
                           marker_color=BLUE, width=0.6))
    fig.update_layout(height=26 * len(ranked) + 80, title="ROC-AUC by model (default configs)",
                      xaxis_range=[0, 1], **BAR_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(lb.sort_values("ROC AUC", ascending=False), use_container_width=True)

elif page == "Model Selection":
    st.title("How the winner was selected")
    n = metrics["screen_models"]
    fig = go.Figure(go.Funnel(
        y=[f"Screened ({n} algorithms)", "Top 10 by ROC-AUC",
           "Tree/boosting families → FLAML", "Final optimized model"],
        x=[n, 10, 4, 1], textinfo="value",
        marker=dict(color=[BLUE_LIGHT, "#6ea3e0", "#4a8cda", BLUE])))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Winner: {metrics['flaml_best_estimator']}")
        st.markdown("Fine-tuned hyperparameters found by FLAML's Bayesian (CFO) search:")
        st.table(pd.DataFrame(metrics["flaml_best_config"].items(),
                              columns=["hyperparameter", "value"]))
    with col2:
        st.subheader("Screening → optimization lift")
        stages = ["Baseline 0.5", f"Screening best\n({metrics['screen_best_model']})",
                  f"FLAML optimized\n({metrics['flaml_best_estimator']})"]
        vals = [0.5, metrics["screen_best_roc_auc"], metrics["test_roc_auc"]]
        fig = go.Figure(go.Bar(x=stages, y=vals, marker_color=[BLUE_LIGHT, "#4a8cda", BLUE],
                               text=[f"{v:.3f}" for v in vals], textposition="outside", width=0.5))
        fig.update_layout(title="ROC-AUC by stage", yaxis_range=[0, 1], height=340, **BAR_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"CV ROC-AUC during search: **{metrics['flaml_cv_roc_auc']:.3f}** · "
                    f"held-out test: **{metrics['test_roc_auc']:.3f}**")

elif page == "Performance":
    st.title("Final model performance (held-out test set)")
    c = st.columns(5)
    for col, k in zip(c, ["test_roc_auc", "test_f1", "test_precision", "test_recall", "test_accuracy"]):
        col.metric(k.replace("test_", "").replace("_", " ").upper(), f"{metrics[k]:.3f}")

    col1, col2 = st.columns(2)
    with col1:
        # recompute ROC curve on the same stratified split (seed fixed in run_workflow)
        from sklearn.model_selection import train_test_split
        X = df.drop(columns=["Churn"]).copy()
        for cname, le in encoders.items():
            X[cname] = le.transform(X[cname])
        _, X_test, _, y_test = train_test_split(X, df["Churn"], test_size=0.2,
                                                random_state=42, stratify=df["Churn"])
        proba = model.predict_proba(X_test.values)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="model",
                                 line=dict(color=BLUE, width=2)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="chance",
                                 line=dict(color="#b3b2ac", width=2, dash="dash")))
        fig.update_layout(title=f"ROC curve — AUC {metrics['test_roc_auc']:.3f}",
                          xaxis_title="False positive rate", yaxis_title="True positive rate",
                          height=380, **BAR_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        cm = np.array(metrics["confusion_matrix"])
        fig = go.Figure(go.Heatmap(z=cm, x=["pred retained", "pred churned"],
                                   y=["actual retained", "actual churned"],
                                   colorscale=[[0, "#eaf1fb"], [1, BLUE]], showscale=False,
                                   text=cm, texttemplate="%{text}"))
        fig.update_layout(title="Confusion matrix", height=380, **BAR_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Decision threshold — a business dial, not a constant")
    st.caption("A missed churner (false negative) costs a customer; a false alarm costs a retention offer. Pick the threshold to match those costs.")
    fig = go.Figure()
    for col_name, color in [("precision", BLUE), ("recall", ORANGE), ("f1", AQUA)]:
        fig.add_trace(go.Scatter(x=sweep["threshold"], y=sweep[col_name], mode="lines+markers",
                                 name=col_name, line=dict(color=color, width=2),
                                 marker=dict(size=8)))
    fig.update_layout(xaxis_title="threshold", height=360,
                      legend=dict(orientation="h", y=1.08), **BAR_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(sweep, use_container_width=True, hide_index=True)

elif page == "Insights":
    st.title("What drives churn")
    top = imp.head(10).iloc[::-1]
    fig = go.Figure(go.Bar(x=top["importance"], y=top.index, orientation="h",
                           marker_color=BLUE, width=0.6))
    fig.update_layout(title="Feature importance (tuned model)", height=380, **BAR_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Driver read-outs → actions")
    ct = df.groupby("Contract")["Churn"].mean().sort_values(ascending=False)
    no_security = df[df["OnlineSecurity"] == "No"]["Churn"].mean()
    has_security = df[df["OnlineSecurity"] == "Yes"]["Churn"].mean()
    no_support = df[df["TechSupport"] == "No"]["Churn"].mean()
    has_support = df[df["TechSupport"] == "Yes"]["Churn"].mean()
    new_acct = df[df["tenure"] <= 6]["Churn"].mean()
    st.markdown(f"""
- **Contract type** — {ct.index[0]}: **{ct.iloc[0]:.0%}** churn vs {ct.index[-1]}: **{ct.iloc[-1]:.0%}** → incentivize longer commitments with discounts.
- **Online security** — no add-on: **{no_security:.0%}** churn vs with add-on: **{has_security:.0%}** → bundle security into the base plan.
- **Tech support** — no add-on: **{no_support:.0%}** churn vs with add-on: **{has_support:.0%}** → proactive support outreach for at-risk accounts.
- **Tenure** — first 6 months: **{new_acct:.0%}** churn → strengthen the early-tenure onboarding program.
""")

elif page == "Live Predictor":
    st.title("Score a customer")
    num_cols = [c for c in features if c not in encoders]
    with st.form("predict"):
        cols = st.columns(3)
        values = {}
        for i, f in enumerate(features):
            with cols[i % 3]:
                if f in encoders:
                    values[f] = st.selectbox(f, list(encoders[f].classes_))
                else:
                    med = float(df[f].median())
                    values[f] = st.number_input(f, value=round(med, 1))
        submitted = st.form_submit_button("Predict churn")
    if submitted:
        row = []
        for f in features:
            v = values[f]
            row.append(int(encoders[f].transform([v])[0]) if f in encoders else float(v))
        proba = float(model.predict_proba(np.array([row]))[:, 1][0])
        risk = ("Low risk", GOOD) if proba < 0.4 else ("Medium risk", WARN) if proba < 0.7 else ("High risk", BAD)
        c1, c2 = st.columns([1, 2])
        c1.metric("Churn probability", f"{proba:.1%}")
        c1.markdown(f"**{risk[0]}**")
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=proba * 100,
            number=dict(suffix="%"),
            gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=risk[1]),
                       steps=[dict(range=[0, 40], color="#e7f2e7"),
                              dict(range=[40, 70], color="#fdf3dd"),
                              dict(range=[70, 100], color="#fbe5e3")])))
        fig.update_layout(height=280, margin=dict(l=30, r=30, t=30, b=10))
        c2.plotly_chart(fig, use_container_width=True)
        action = {"Low risk": "Standard engagement; loyalty rewards.",
                  "Medium risk": "Automated email sequence + satisfaction survey + 10-15% offer.",
                  "High risk": "Personal retention call + exclusive discount + dedicated manager."}[risk[0]]
        st.markdown(f"**Recommended action:** {action}")
