import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

from src.recommendation import get_risk_level, get_recommendation
from src.iot_simulator import generate_sensor_data


st.set_page_config(
    page_title="AI Predictive Maintenance Dashboard",
    page_icon="🏭",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
.metric-card {
    background-color: #1E1E1E;
    padding: 20px;
    border-radius: 15px;
}
</style>
""", unsafe_allow_html=True)

st.title("🏭 AI-Powered Industrial IoT Predictive Maintenance Dashboard")
st.write("Predict machine failure using IoT sensor data and machine learning.")

model = joblib.load("models/predictive_model.pkl")
scaler = joblib.load("models/scaler.pkl")

df = pd.read_csv("data/predictive_maintenance.csv")

tab1, tab2, tab3, tab4 = st.tabs([
    "Live Prediction",
    "Dataset Analytics",
    "Model Performance",
    "IoT Simulation"
])

with tab1:
    st.header("🔍 Live Machine Failure Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        machine_type = st.selectbox("Machine Type", [0, 1, 2])
        air_temp = st.slider("Air Temperature [K]", 295.0, 310.0, 300.0)
        process_temp = st.slider("Process Temperature [K]", 305.0, 320.0, 310.0)

    with col2:
        rpm = st.slider("Rotational Speed [rpm]", 1000, 2000, 1500)
        torque = st.slider("Torque [Nm]", 20.0, 80.0, 40.0)
        tool_wear = st.slider("Tool Wear [min]", 0, 260, 100)

    with col3:
        twf = st.selectbox("Tool Wear Failure", [0, 1])
        hdf = st.selectbox("Heat Dissipation Failure", [0, 1])
        pwf = st.selectbox("Power Failure", [0, 1])
        osf = st.selectbox("Overstrain Failure", [0, 1])
        rnf = st.selectbox("Random Failure", [0, 1])

    input_data = {
        "Type": machine_type,
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rpm,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
        "TWF": twf,
        "HDF": hdf,
        "PWF": pwf,
        "OSF": osf,
        "RNF": rnf
    }

    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    risk_level, risk_message = get_risk_level(probability)
    recommendations = get_recommendation(input_data)

    if st.button("Predict Machine Health"):
        c1, c2, c3 = st.columns(3)

        c1.metric("Prediction", "Failure" if prediction == 1 else "No Failure")
        c2.metric("Failure Probability", f"{probability * 100:.2f}%")
        c3.metric("Risk Level", risk_level)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={"text": "Failure Risk %"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "red"},
                "steps": [
                    {"range": [0, 25], "color": "green"},
                    {"range": [25, 50], "color": "yellow"},
                    {"range": [50, 75], "color": "orange"},
                    {"range": [75, 100], "color": "red"}
                ]
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Maintenance Recommendations")
        for rec in recommendations:
            st.warning(rec)

with tab2:
    st.header("📊 Dataset Analytics")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", df.shape[0])
    c2.metric("Total Features", df.shape[1])
    c3.metric("Failures", int(df["Machine failure"].sum()))
    c4.metric("No Failures", int((df["Machine failure"] == 0).sum()))

    fig1 = px.histogram(df, x="Machine failure", color="Machine failure",
                        title="Failure vs No Failure Distribution")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.scatter(
        df,
        x="Torque [Nm]",
        y="Rotational speed [rpm]",
        color="Machine failure",
        title="Torque vs Rotational Speed"
    )
    st.plotly_chart(fig2, use_container_width=True)

    numeric_df = df.select_dtypes(include=["int64", "float64"])
    corr = numeric_df.corr()

    fig3 = px.imshow(
        corr,
        text_auto=True,
        title="Sensor Correlation Heatmap",
        color_continuous_scale="RdBu_r"
    )
    st.plotly_chart(fig3, use_container_width=True)

with tab3:
    st.header("📈 Model Performance")

    comparison = pd.read_csv("outputs/model_comparison.csv")
    st.dataframe(comparison)

    fig4 = px.bar(
        comparison,
        x="Model",
        y=["Accuracy", "Precision", "Recall", "F1 Score"],
        barmode="group",
        title="Model Comparison"
    )
    st.plotly_chart(fig4, use_container_width=True)

    st.image("outputs/confusion_matrix.png", caption="Confusion Matrix")

with tab4:
    st.header("📡 Virtual IoT Sensor Simulation")

    if st.button("Generate Live Sensor Reading"):
        sensor_data = generate_sensor_data()
        st.json(sensor_data)

        sensor_df = pd.DataFrame([sensor_data])
        sensor_scaled = scaler.transform(sensor_df)

        sim_pred = model.predict(sensor_scaled)[0]
        sim_prob = model.predict_proba(sensor_scaled)[0][1]

        risk_level, risk_message = get_risk_level(sim_prob)

        st.metric("Simulated Prediction", "Failure" if sim_pred == 1 else "No Failure")
        st.metric("Failure Probability", f"{sim_prob * 100:.2f}%")
        st.metric("Risk Level", risk_level)