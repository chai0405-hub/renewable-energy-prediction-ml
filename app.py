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

st.title("⚡ Renewable Energy Output Prediction Dashboard")
st.write("Predict renewable energy generation using environmental parameters.")

# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_model():
    if os.path.exists("model.pkl"):
        return joblib.load("model.pkl")
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

st.sidebar.header("⚙ Input Parameters")

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

st.header("🔮 Energy Output Prediction")

if st.button("Predict Energy Output"):

    if model is None:

        st.error("⚠ model.pkl not found in project folder.")

    else:

        input_data = np.array([[temperature, humidity, wind_speed, solar_radiation]])

        prediction = model.predict(input_data)

        st.success(
            f"🔋 Predicted Energy Output: {prediction[0]:.2f} kWh"
        )

# --------------------------------------------------
# Dashboard Section
# --------------------------------------------------

st.markdown("---")
st.header("📊 Energy Analytics Dashboard")

if data is None:

    st.error("⚠ Dataset not found. Please upload renewable_energy_dataset.csv")

else:

    # ---------------- KPI Metrics ----------------

    if "energy_output" in data.columns:

        st.subheader("📈 Key Metrics")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Average Output",
            f"{data['energy_output'].mean():.2f} kWh"
        )

        col2.metric(
            "Maximum Output",
            f"{data['energy_output'].max():.2f} kWh"
        )

        col3.metric(
            "Minimum Output",
            f"{data['energy_output'].min():.2f} kWh"
        )

    # ---------------- Solar Radiation Chart ----------------

    if "solar_radiation" in data.columns and "energy_output" in data.columns:

        st.subheader("☀ Solar Radiation vs Energy Output")

        fig1 = px.scatter(
            data,
            x="solar_radiation",
            y="energy_output",
            title="Solar Radiation Impact on Energy Production"
        )

        st.plotly_chart(fig1, use_container_width=True)

    # ---------------- Wind Speed Chart ----------------

    if "wind_speed" in data.columns and "energy_output" in data.columns:

        st.subheader("💨 Wind Speed vs Energy Output")

        fig2 = px.scatter(
            data,
            x="wind_speed",
            y="energy_output",
            title="Wind Speed Impact on Energy Production"
        )

        st.plotly_chart(fig2, use_container_width=True)

    # ---------------- Histogram ----------------

    if "energy_output" in data.columns:

        st.subheader("📊 Energy Output Distribution")

        fig3 = px.histogram(
            data,
            x="energy_output",
            nbins=30
        )

        st.plotly_chart(fig3, use_container_width=True)

    # ---------------- Correlation Heatmap ----------------

    st.subheader("🔥 Correlation Heatmap")

    corr = data.corr(numeric_only=True)

    fig4 = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="viridis"
    )

    st.plotly_chart(fig4, use_container_width=True)

    # ---------------- Dataset Preview ----------------

    st.subheader("📂 Dataset Preview")

    st.dataframe(data.head(10))

    # ---------------- Download Dataset ----------------

    st.download_button(
        label="📥 Download Dataset",
        data=data.to_csv(index=False),
        file_name="renewable_energy_dataset.csv",
        mime="text/csv"
    )

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown("---")
st.caption("⚡ Renewable Energy Prediction System | Built with Streamlit")