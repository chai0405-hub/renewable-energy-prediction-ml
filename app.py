import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Renewable Energy Prediction System",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Renewable Energy Output Prediction")
st.write("Enter environmental parameters to predict renewable energy output.")

# --------------------------------------------------
# Load Model
# --------------------------------------------------
@st.cache_resource
def load_model():
    model_path = "model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        return None

model = load_model()

# --------------------------------------------------
# Sidebar Inputs
# --------------------------------------------------
st.sidebar.header("Input Parameters")

temperature = st.sidebar.number_input(
    "Temperature (°C)", min_value=-20.0, max_value=60.0, value=25.0
)

humidity = st.sidebar.number_input(
    "Humidity (%)", min_value=0.0, max_value=100.0, value=50.0
)

wind_speed = st.sidebar.number_input(
    "Wind Speed (m/s)", min_value=0.0, max_value=50.0, value=5.0
)

solar_radiation = st.sidebar.number_input(
    "Solar Radiation (W/m²)", min_value=0.0, max_value=1500.0, value=500.0
)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("Predict Energy Output"):

    if model is None:
        st.error("⚠ Model file (model.pkl) not found.")
    else:
        input_data = np.array([[temperature, humidity, wind_speed, solar_radiation]])
        prediction = model.predict(input_data)

        st.success(f"🔋 Predicted Energy Output: {prediction[0]:.2f} kWh")

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
st.markdown("---")
st.header("📊 Renewable Energy Data Dashboard")

data_path = "data/renewable_energy_dataset.csv"

if os.path.exists(data_path):

    data = pd.read_csv(data_path)

    # Clean column names
    data.columns = data.columns.str.strip().str.lower()

    # Detect energy output column safely
    if "energy_output" in data.columns:
        energy_col = "energy_output"
    elif "energy output" in data.columns:
        energy_col = "energy output"
    else:
        energy_col = data.columns[-1]

    # --------------------------------------------------
    # KPI Metrics
    # --------------------------------------------------
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Average Energy Output",
        round(data[energy_col].mean(), 2)
    )

    col2.metric(
        "Maximum Energy Output",
        round(data[energy_col].max(), 2)
    )

    col3.metric(
        "Minimum Energy Output",
        round(data[energy_col].min(), 2)
    )

    # --------------------------------------------------
    # Scatter Chart
    # --------------------------------------------------
    st.subheader("⚡ Solar Radiation vs Energy Output")

    if "solar_radiation" in data.columns:

        fig = px.scatter(
            data,
            x="solar_radiation",
            y=energy_col,
            title="Solar Radiation vs Energy Output",
            color=energy_col
        )

        st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------
    # Histogram
    # --------------------------------------------------
    st.subheader("📈 Energy Output Distribution")

    fig2 = px.histogram(
        data,
        x=energy_col,
        nbins=30,
        title="Energy Output Distribution"
    )

    st.plotly_chart(fig2, use_container_width=True)

    # --------------------------------------------------
    # Correlation Heatmap
    # --------------------------------------------------
    st.subheader("🔥 Feature Correlation Heatmap")

    corr = data.corr(numeric_only=True)

    fig3, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)

    st.pyplot(fig3)

else:
    st.warning("Dataset not found. Please upload renewable_energy_dataset.csv inside the data folder.")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("Built with Streamlit 🚀")