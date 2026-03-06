import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Renewable Energy Prediction System",
    page_icon="⚡",
    layout="centered"
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
        st.error("⚠ Model file (model.pkl) not found. Please upload it to the repository.")
    else:
        input_data = np.array([[temperature, humidity, wind_speed, solar_radiation]])
        prediction = model.predict(input_data)

        st.success(f"🔋 Predicted Energy Output: {prediction[0]:.2f} kWh")

# --------------------------------------------------
# Dashboard Section
# --------------------------------------------------
st.markdown("---")
st.header("📊 Renewable Energy Data Dashboard")

# Load dataset
data = pd.read_csv("renewable_energy_dataset.csv")

# Show dataset
st.subheader("Dataset Preview")
st.dataframe(data.head())

# Show statistics
st.subheader("Dataset Statistics")
st.write(data.describe())

# Temperature vs Energy Output
st.subheader("Temperature vs Energy Output")
st.line_chart(data.set_index("Temperature")["Energy Output"])

# Humidity vs Energy Output
st.subheader("Humidity vs Energy Output")
st.line_chart(data.set_index("Humidity")["Energy Output"])

# Wind Speed vs Energy Output
st.subheader("Wind Speed vs Energy Output")
st.line_chart(data.set_index("Wind Speed")["Energy Output"])

# Solar Radiation vs Energy Output
st.subheader("Solar Radiation vs Energy Output")
st.line_chart(data.set_index("Solar Radiation")["Energy Output"])

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("Built with Streamlit 🚀")