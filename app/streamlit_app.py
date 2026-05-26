from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st


RISK_FILE = Path("reports/maintenance_risk_scores.csv")
FEATURE_FILE = Path("data/processed/fd001_train_features.csv")

RUL_SHAP_IMAGE = Path("reports/figures/rul_shap_summary.png")
FAILURE_SHAP_IMAGE = Path("reports/figures/failure_shap_summary.png")


st.set_page_config(
    page_title="Gedis Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
)


@st.cache_data
def load_risk_data():
    if not RISK_FILE.exists():
        st.error(
            "Missing reports/maintenance_risk_scores.csv. "
            "Run src/models/predict_risk_score.py first."
        )
        st.stop()

    return pd.read_csv(RISK_FILE)


@st.cache_data
def load_feature_data():
    if not FEATURE_FILE.exists():
        st.error(
            "Missing data/processed/fd001_train_features.csv. "
            "Run src/features/build_features.py first."
        )
        st.stop()

    return pd.read_csv(FEATURE_FILE)


risk_df = load_risk_data()
feature_df = load_feature_data()


st.title("Gedis – Predictive Maintenance Demo")

st.markdown(
    """
    
    Goal is to explore how machine learning can support predictive maintenance:
    estimating Remaining Useful Life, detecting near-failure risk, identifying unusual sensor behaviour,
    and combining these outputs into a simple maintenance risk score.

    """
)


# Sidebar
st.sidebar.header("Engine selection")

unit_ids = sorted(risk_df["unit_number"].unique())
selected_unit = st.sidebar.selectbox("Select engine unit", unit_ids)

unit_risk_df = risk_df[risk_df["unit_number"] == selected_unit].copy()
unit_feature_df = feature_df[feature_df["unit_number"] == selected_unit].copy()

latest_state = unit_risk_df.sort_values("time_in_cycles").iloc[-1]

st.sidebar.metric("Latest cycle", int(latest_state["time_in_cycles"]))
st.sidebar.metric("Actual RUL", int(latest_state["RUL"]))
st.sidebar.metric("Predicted RUL", round(latest_state["predicted_RUL"], 1))
st.sidebar.metric(
    "Failure probability",
    f"{latest_state['failure_probability']:.1%}"
)
st.sidebar.metric(
    "Maintenance risk",
    f"{latest_state['maintenance_risk_score']:.1f}"
)
st.sidebar.write(f"Risk level: **{latest_state['risk_level']}**")


# Main KPIs
col1, col2, col3, col4 = st.columns(4)

col1.metric("Predicted RUL", f"{latest_state['predicted_RUL']:.1f} cycles")
col2.metric("Failure probability", f"{latest_state['failure_probability']:.1%}")
col3.metric("Anomaly score", f"{latest_state['anomaly_score']:.3f}")
col4.metric("Risk score", f"{latest_state['maintenance_risk_score']:.1f}")


# Interpretation
st.subheader("Simple maintenance interpretation")

risk_level = latest_state["risk_level"]

if risk_level == "Critical":
    st.error("Critical risk: the model sees strong signs of late-stage degradation.")
elif risk_level == "High":
    st.warning("High risk: the engine shows elevated failure probability and should be monitored closely.")
elif risk_level == "Medium":
    st.info("Medium risk: some degradation signals are present, but the situation is not yet critical.")
else:
    st.success("Low risk: the engine looks relatively healthy based on the available sensor data.")


# Risk trend
st.subheader("Risk trend over engine lifecycle")

fig_risk = px.line(
    unit_risk_df,
    x="time_in_cycles",
    y="maintenance_risk_score",
    title=f"Engine {selected_unit} maintenance risk score over time",
    labels={
        "time_in_cycles": "Time in cycles",
        "maintenance_risk_score": "Maintenance risk score"
    }
)

fig_risk.add_hline(y=25, line_dash="dot")
fig_risk.add_hline(y=50, line_dash="dot")
fig_risk.add_hline(y=75, line_dash="dot")

st.plotly_chart(fig_risk, use_container_width=True)


# RUL and failure probability
left, right = st.columns(2)

with left:
    fig_rul = px.line(
        unit_risk_df,
        x="time_in_cycles",
        y=["RUL", "predicted_RUL"],
        title="Actual vs predicted RUL",
        labels={
            "time_in_cycles": "Time in cycles",
            "value": "RUL",
            "variable": "Series"
        }
    )
    st.plotly_chart(fig_rul, use_container_width=True)

with right:
    fig_failure = px.line(
        unit_risk_df,
        x="time_in_cycles",
        y="failure_probability",
        title="Failure probability over time",
        labels={
            "time_in_cycles": "Time in cycles",
            "failure_probability": "Failure probability"
        }
    )
    st.plotly_chart(fig_failure, use_container_width=True)


# Sensor inspection
st.subheader("Sensor signal inspection")

sensor_cols = [
    col for col in unit_feature_df.columns
    if col.startswith("sensor_")
    and "_roll_" not in col
    and not col.endswith("_delta")
]

default_sensors = [
    sensor for sensor in ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_11"]
    if sensor in sensor_cols
]

selected_sensors = st.multiselect(
    "Select sensors to display",
    sensor_cols,
    default=default_sensors
)

if selected_sensors:
    sensor_plot_df = unit_feature_df[
        ["time_in_cycles"] + selected_sensors
    ].melt(
        id_vars="time_in_cycles",
        var_name="sensor",
        value_name="value"
    )

    fig_sensors = px.line(
        sensor_plot_df,
        x="time_in_cycles",
        y="value",
        color="sensor",
        title=f"Engine {selected_unit} sensor trends"
    )

    st.plotly_chart(fig_sensors, use_container_width=True)
else:
    st.info("Select at least one sensor to display.")


# Fleet overview
st.subheader("Fleet-level overview")

latest_per_unit = (
    risk_df.sort_values("time_in_cycles")
    .groupby("unit_number")
    .tail(1)
    .copy()
)

fig_fleet = px.histogram(
    latest_per_unit,
    x="risk_level",
    title="Latest risk level distribution across engines",
    category_orders={
        "risk_level": ["Low", "Medium", "High", "Critical"]
    }
)

st.plotly_chart(fig_fleet, use_container_width=True)

st.dataframe(
    latest_per_unit[
        [
            "unit_number",
            "time_in_cycles",
            "RUL",
            "predicted_RUL",
            "failure_probability",
            "maintenance_risk_score",
            "risk_level",
        ]
    ].sort_values("maintenance_risk_score", ascending=False),
    use_container_width=True
)


# Project limitations
st.subheader("Project limitations")

st.markdown(
    """
    This dashboard is based on a benchmark simulation dataset, not live factory data.

    Main limitations:
    - the dataset is cleaner than real industrial sensor data;
    - the failure modes are simplified;
    - maintenance cost, spare parts and downtime impact are not modelled;
    - the risk score is a custom heuristic, not an officially validated maintenance rule;
    - the model should be interpreted as a learning project, not as a deployed safety-critical system.

    The main purpose is to demonstrate the predictive maintenance workflow:
    data preparation, time-series feature engineering, supervised learning, anomaly detection,
    explainability and dashboarding.
    """
)


# Explainability
st.subheader("Model explainability")

tab1, tab2 = st.tabs(["RUL model SHAP", "Failure model SHAP"])

with tab1:
    if RUL_SHAP_IMAGE.exists():
        st.image(str(RUL_SHAP_IMAGE), caption="RUL model SHAP summary")
    else:
        st.info("Run src/models/explain_models.py to generate RUL SHAP summary.")

with tab2:
    if FAILURE_SHAP_IMAGE.exists():
        st.image(str(FAILURE_SHAP_IMAGE), caption="Failure model SHAP summary")
    else:
        st.info("Run src/models/explain_models.py to generate failure SHAP summary.")