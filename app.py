import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="Renewable Energy Prediction System",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Renewable Energy Output Prediction System")

# ---------------------------------
# Load Dataset
# ---------------------------------
data = pd.read_csv("data/renewable_energy_dataset.csv")

# ---------------------------------
# Load ML Model
# ---------------------------------
model = joblib.load("model.pkl")

# ---------------------------------
# Sidebar Inputs
# ---------------------------------
st.sidebar.header("Enter Environmental Parameters")

temperature = st.sidebar.slider("Temperature (°C)", 0.0, 50.0, 25.0)
solar_irradiance = st.sidebar.slider("Solar Irradiance", 0.0, 1000.0, 500.0)
wind_speed = st.sidebar.slider("Wind Speed (m/s)", 0.0, 25.0, 10.0)
rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 50.0, 5.0)

# ---------------------------------
# Prediction
# ---------------------------------
input_data = np.array([[temperature, solar_irradiance, wind_speed, rainfall]])

prediction = model.predict(input_data)

st.subheader("🔮 Predicted Renewable Energy Output")

st.success(f"⚡ Predicted Energy Output: {prediction[0]:.2f} kWh")

# ---------------------------------
# Dashboard Section
# ---------------------------------
st.header("📊 Renewable Energy Dashboard")

# Metrics
col1, col2, col3 = st.columns(3)

col1.metric("Average Energy Output", round(data["energy_output"].mean(), 2))
col2.metric("Maximum Energy Output", round(data["energy_output"].max(), 2))
col3.metric("Minimum Energy Output", round(data["energy_output"].min(), 2))

# ---------------------------------
# Dataset Preview
# ---------------------------------
if st.checkbox("Show Dataset"):
    st.write(data.head())

# ---------------------------------
# Energy Distribution
# ---------------------------------
st.subheader("⚡ Energy Output Distribution")

fig1 = px.histogram(
    data,
    x="energy_output",
    nbins=30,
    title="Distribution of Energy Output"
)

st.plotly_chart(fig1, use_container_width=True)

# ---------------------------------
# Solar Irradiance vs Energy
# ---------------------------------
st.subheader("☀ Solar Irradiance vs Energy Output")

fig2 = px.scatter(
    data,
    x="solar_irradiance",
    y="energy_output",
    color="temperature",
    title="Solar Irradiance vs Energy Output"
)

st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------
# Wind Speed vs Energy
# ---------------------------------
st.subheader("🌬 Wind Speed vs Energy Output")

fig3 = px.scatter(
    data,
    x="wind_speed",
    y="energy_output",
    color="rainfall",
    title="Wind Speed vs Energy Output"
)

st.plotly_chart(fig3, use_container_width=True)

# ---------------------------------
# Correlation Heatmap
# ---------------------------------
st.subheader("🔥 Feature Correlation Heatmap")

fig, ax = plt.subplots()

sns.heatmap(
    data.corr(),
    annot=True,
    cmap="coolwarm",
    ax=ax
)

st.pyplot(fig)

# ---------------------------------
# Energy Trend
# ---------------------------------
st.subheader("📈 Energy Output Trend")

fig4 = px.line(
    data,
    y="energy_output",
    title="Energy Output Trend"
)

st.plotly_chart(fig4, use_container_width=True)

# ---------------------------------
# Footer
# ---------------------------------
st.markdown("---")
st.markdown("Developed using Machine Learning & Streamlit")