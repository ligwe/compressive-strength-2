import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import differential_evolution

BASE_DIR = Path(__file__).parent
st.set_page_config(page_title="Concrete Strength AI Tool", page_icon="🏗️", layout="wide")

@st.cache_resource
def load_artifacts():
    models = joblib.load(BASE_DIR / "models.joblib")
    with open(BASE_DIR / "deployment_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    return models, config

models, config = load_artifacts()
RAW_FEATURES = config["raw_features"]
FEATURES = config["engineered_features"]
AGES = config["optimization_ages"]
BOUNDS = [tuple(pair) for pair in config["optimization_bounds"]]
FEASIBILITY = {key: tuple(value) for key, value in config["feasibility_ranges"].items()}


def build_engineered_row(cement, water, sand, granite, age):
    """Keep feature calculations aligned with the research notebook."""
    cement, water, sand, granite, age = map(float, [cement, water, sand, granite, age])
    if cement <= 0 or water <= 0 or granite <= 0 or sand < 0 or age <= 0:
        raise ValueError("Cement, water, granite and curing age must be positive; sand cannot be negative.")
    wc_ratio = water / cement
    return {
        "Cement": cement, "Water": water, "Sand": sand, "Granite": granite,
        "Curing Age": age, "WC_Ratio_Calculated": wc_ratio, "WC_Ratio": wc_ratio,
        "Total_Aggregate": sand + granite, "Fine_Coarse_Ratio": sand / granite,
        "Total_Listed_Mass": cement + water + sand + granite,
    }


def make_feature_frame(values):
    row = build_engineered_row(values["Cement"], values["Water"], values["Sand"], values["Granite"], values["Curing Age"])
    return pd.DataFrame([row])[FEATURES]


def candidate_frame(candidate):
    cement, sand, granite, wc_ratio = map(float, candidate)
    water = cement * wc_ratio
    rows = [build_engineered_row(cement, water, sand, granite, age) for age in AGES]
    return pd.DataFrame(rows)[FEATURES]


def constraint_penalty(candidate):
    cement, sand, granite, wc_ratio = map(float, candidate)
    if cement <= 0 or granite <= 0 or sand < 0 or wc_ratio <= 0:
        return 1e8
    water = cement * wc_ratio
    values = {
        "aggregate_to_cement": (sand + granite) / cement,
        "fine_coarse": sand / granite,
        "total_mass": cement + water + sand + granite,
        "wc_ratio": wc_ratio,
    }
    penalty = 0.0
    for name, value in values.items():
        lower, upper = FEASIBILITY[name]
        scale = max(upper - lower, 1e-8)
        if value < lower:
            penalty += (lower - value) / scale
        elif value > upper:
            penalty += (value - upper) / scale
    return penalty


def optimize_mix(model_name, maxiter=60, popsize=10):
    model = models[model_name]

    def objective(candidate):
        penalty = constraint_penalty(candidate)
        if penalty > 0:
            return 1e5 + 1e5 * penalty
        predictions = np.asarray(model.predict(candidate_frame(candidate)), dtype=float).ravel()
        return -float(np.mean(predictions))

    result = differential_evolution(
        objective, bounds=BOUNDS, seed=int(config["random_state"]), maxiter=maxiter,
        popsize=popsize, tol=0.01, polish=True, updating="immediate", workers=1
    )
    cement, sand, granite, wc_ratio = map(float, result.x)
    water = cement * wc_ratio
    predictions = np.asarray(model.predict(candidate_frame(result.x)), dtype=float).ravel()
    summary = {
        "Cement": cement, "Water": water, "Sand": sand, "Granite": granite,
        "Water/Cement Ratio": wc_ratio, "Sand/Granite Ratio": sand / granite,
        "Aggregate/Cement Ratio": (sand + granite) / cement,
        "Total Listed Mass": cement + water + sand + granite,
        "Mean predicted strength": float(np.mean(predictions)),
        "Minimum predicted strength across ages": float(np.min(predictions)),
        "Maximum predicted strength across ages": float(np.max(predictions)),
        "Feasibility penalty": float(constraint_penalty(result.x)),
        "Optimizer converged": bool(result.success), "Optimizer message": str(result.message),
    }
    age_results = pd.DataFrame({"Curing Age": AGES, "Predicted Compressive Strength": predictions})
    return summary, age_results


st.title("Concrete Strength Prediction and Mix Optimization")
st.caption("Research demonstration using the trained SVR, ANN and XGBoost pipelines from the accompanying notebook.")
st.warning("Research use only. Predictions and optimized candidates are not experimentally validated mix designs. Verify workability, durability, applicable design standards and laboratory strength before construction use.")
prediction_tab, optimization_tab, performance_tab, about_tab = st.tabs(["Strength prediction", "Mix optimization", "Model performance", "About this tool"])

with prediction_tab:
    st.subheader("Enter concrete mix proportions")
    defaults, ranges = config["physical_defaults"], config["physical_ranges"]
    with st.form("predict_form"):
        cols, inputs = st.columns(3), {}
        for i, feature in enumerate(RAW_FEATURES):
            info = ranges[feature]
            lo, hi, default = float(info["min"]), float(info["max"]), float(defaults[feature])
            with cols[i % 3]:
                inputs[feature] = st.number_input(
                    feature, min_value=lo, max_value=hi, value=min(max(default, lo), hi), format="%.4f",
                    help=f"Observed training range: {lo:.4f} to {hi:.4f}. Values outside this range are not accepted in this demo."
                )
        selected_model = st.selectbox("Prediction model", ["XGBoost", "SVR", "ANN", "Compare all models"])
        predict_clicked = st.form_submit_button("Predict compressive strength", type="primary")

    if predict_clicked:
        try:
            X_input = make_feature_frame(inputs)
            selected = list(models.keys()) if selected_model == "Compare all models" else [selected_model]
            results = [{"Model": name, "Predicted compressive strength": float(np.asarray(models[name].predict(X_input)).ravel()[0])} for name in selected]
            result_df = pd.DataFrame(results)
            metric_cols = st.columns(len(results))
            for col, row in zip(metric_cols, results):
                col.metric(row["Model"], f"{row['Predicted compressive strength']:.3f}")
            st.dataframe(result_df, use_container_width=True, hide_index=True)
            if len(result_df) > 1:
                st.bar_chart(result_df.set_index("Model"))
            st.subheader("Calculated engineered features")
            st.dataframe(X_input.T.rename(columns={0: "Value"}), use_container_width=True)
            st.caption(config["strength_unit_note"])
            out_of_range = [name for name, value in inputs.items() if value < ranges[name]["q05"] or value > ranges[name]["q95"]]
            if out_of_range:
                st.info("Caution: these inputs are outside the central 5th to 95th percentile training range: " + ", ".join(out_of_range))
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

with optimization_tab:
    st.subheader("Optimize one candidate mix across supported curing ages")
    st.write("The optimizer maximizes mean predicted strength across curing ages represented in the training partition, while penalizing compositions outside the notebook's empirical feasibility ranges.")
    opt_model = st.selectbox("Optimization model", ["XGBoost", "SVR", "ANN"], key="opt_model")
    maxiter = st.slider("Optimization effort", min_value=20, max_value=120, value=60, step=10, help="Higher values can improve the search but take longer.")
    if st.button("Optimize concrete mix", type="primary"):
        with st.spinner("Searching candidate mix proportions. This may take a little while..."):
            try:
                summary, by_age = optimize_mix(opt_model, maxiter=maxiter)
                st.success("Optimization search completed. Review the convergence and feasibility notes below.")
                st.dataframe(pd.DataFrame([summary]).T.rename(columns={0: "Value"}), use_container_width=True)
                st.subheader("Predicted strength across training supported curing ages")
                st.dataframe(by_age, use_container_width=True, hide_index=True)
                st.line_chart(by_age.set_index("Curing Age"))
                if summary["Feasibility penalty"] > 1e-8:
                    st.warning("The returned candidate falls outside one or more empirical composition ranges. Treat it as a failed feasibility check and do not use it as a recommended mix.")
                if not summary["Optimizer converged"]:
                    st.info("The optimizer did not report convergence. Consider a higher optimization effort and inspect the result carefully.")
            except Exception as exc:
                st.error(f"Optimization failed: {exc}")

with performance_tab:
    st.subheader("Held out test set performance")
    metrics = config.get("test_metrics", {})
    if metrics:
        metric_df = pd.DataFrame(metrics).T
        st.dataframe(metric_df, use_container_width=True)
        if "RMSE" in metric_df.columns:
            st.bar_chart(metric_df[["RMSE"]])
    else:
        st.info("Test metrics were not available in the exported configuration.")
    st.caption("These are the test metrics recorded in the notebook at export time. They are not recalculated by this web app.")

with about_tab:
    st.subheader("Methodology and limitations")
    st.markdown("""
    **Models:** Support Vector Regression, Artificial Neural Network implemented as scikit-learn MLPRegressor, and XGBoost.

    **Inputs:** Cement, water, sand, granite and curing age. The app calculates water to cement ratio, total aggregate, sand to granite ratio and total listed mass exactly as defined in the notebook.

    **Optimization:** Searches cement, sand, granite and water to cement ratio within training derived bounds. It maximizes average model predicted strength over the curing ages represented in the training partition and penalizes empirical composition range violations.

    **Limitations:** Model predictions are limited by the coverage and quality of the training dataset. This optimization objective does not itself guarantee a code compliant, durable, workable or economical concrete mix. Independent engineering review and laboratory validation are required.
    """)
