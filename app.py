import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import plotly.express as px

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Renewable Energy Prediction System",
    page_icon="⚡",
    layout="wide"
)

st.title("🔮 Predict Renewable Energy Output")

# --------------------------------------------------
# Load Model
# --------------------------------------------------
@st.cache_resource
def load_model():
    model_path = "model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("data/renewable_energy_dataset.csv")
        data.columns = data.columns.str.strip().str.lower()
        return data
    except:
        return None

data = load_data()

# --------------------------------------------------
# Sidebar Inputs
# --------------------------------------------------
st.sidebar.header("Input Parameters")

temperature = st.sidebar.number_input(
    "Temperature (°C)", -20.0, 60.0, 25.0
)

humidity = st.sidebar.number_input(
    "Humidity (%)", 0.0, 100.0, 50.0
)

wind_speed = st.sidebar.number_input(
    "Wind Speed (m/s)", 0.0, 50.0, 5.0
)

solar_radiation = st.sidebar.number_input(
    "Solar Radiation (W/m²)", 0.0, 1500.0, 500.0
)

# --------------------------------------------------
# Prediction Section
# --------------------------------------------------
st.subheader("⚡ Energy Output Prediction")

if st.button("⚡ Predict Energy Output"):

    if model is None:
        st.error("Model file not found.")
    else:

        input_data = np.array([[temperature, humidity, wind_speed, solar_radiation]])

        prediction = model.predict(input_data)

        st.success(f"⚡ Predicted Energy Output: {prediction[0]:.2f} kWh")

# --------------------------------------------------
# Dashboard Section
# --------------------------------------------------
st.markdown("---")
st.header("📊 Renewable Energy Dashboard")

if data is None:
    st.error("Dataset not found.")
else:

    # ------------------------------
    # KPI Metrics
    # ------------------------------
    st.subheader("Key Performance Indicators")

    col1, col2, col3 = st.columns(3)

    if "energy_output" in data.columns:

        col1.metric(
            "Average Energy Output",
            f"{data['energy_output'].mean():.2f} kWh"
        )

        col2.metric(
            "Maximum Energy Output",
            f"{data['energy_output'].max():.2f} kWh"
        )

        col3.metric(
            "Minimum Energy Output",
            f"{data['energy_output'].min():.2f} kWh"
        )

    # ------------------------------
    # Charts
    # ------------------------------
    st.subheader("Energy Analysis Charts")

    col1, col2 = st.columns(2)

    if "solar_radiation" in data.columns:

        fig1 = px.scatter(
            data,
            x="solar_radiation",
            y="energy_output",
            title="Solar Radiation vs Energy Output"
        )

        col1.plotly_chart(fig1, use_container_width=True)

    if "wind_speed" in data.columns:

        fig2 = px.scatter(
            data,
            x="wind_speed",
            y="energy_output",
            title="Wind Speed vs Energy Output"
        )

        col2.plotly_chart(fig2, use_container_width=True)

    # ------------------------------
    # Histogram
    # ------------------------------
    st.subheader("Energy Output Distribution")

    if "energy_output" in data.columns:

        fig3 = px.histogram(
            data,
            x="energy_output",
            nbins=30,
            title="Distribution of Energy Output"
        )

        st.plotly_chart(fig3, use_container_width=True)

    # ------------------------------
    # Correlation Heatmap
    # ------------------------------
    st.subheader("Feature Correlation Heatmap")

    corr = data.corr(numeric_only=True)

    fig4 = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="Viridis"
    )

    st.plotly_chart(fig4, use_container_width=True)

    # ------------------------------
    # Dataset Viewer
    # ------------------------------
    st.subheader("Dataset Preview")

    st.dataframe(data.head(10))

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("Built with Streamlit 🚀 | Renewable Energy Prediction System")