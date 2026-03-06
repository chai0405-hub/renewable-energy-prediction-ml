import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import plotly.express as px
from sklearn.metrics import r2_score, mean_squared_error

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Renewable Energy Prediction Dashboard",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Renewable Energy Output Prediction System")
st.markdown("Machine Learning powered dashboard for predicting renewable energy production.")

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
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_model():
    if os.path.exists("model.pkl"):
        return joblib.load("model.pkl")
    return None

model = load_model()

# --------------------------------------------------
# Tabs Layout
# --------------------------------------------------

tab1, tab2, tab3 = st.tabs([
    "🔮 Energy Prediction",
    "📊 Energy Analytics Dashboard",
    "📂 Dataset Explorer"
])

# ==================================================
# TAB 1 : PREDICTION
# ==================================================

with tab1:

    st.header("🔮 Predict Renewable Energy Output")

    st.sidebar.header("Input Parameters")

    temperature = st.sidebar.slider(
        "Temperature (°C)", -20.0, 60.0, 25.0
    )

    humidity = st.sidebar.slider(
        "Humidity (%)", 0.0, 100.0, 50.0
    )

    wind_speed = st.sidebar.slider(
        "Wind Speed (m/s)", 0.0, 50.0, 5.0
    )

    solar_radiation = st.sidebar.slider(
        "Solar Radiation (W/m²)", 0.0, 1500.0, 500.0
    )

    if st.button("⚡ Predict Energy Output"):

        if model is None:

            st.error("Model file not found. Please upload model.pkl")

        else:

            input_data = np.array([[temperature, humidity, wind_speed, solar_radiation]])

            prediction = model.predict(input_data)

            st.success(
                f"🔋 Predicted Energy Output: {prediction[0]:.2f} kWh"
            )

# ==================================================
# TAB 2 : DASHBOARD
# ==================================================

with tab2:

    if data is None:
        st.error("Dataset not found.")
    else:

        st.header("📊 Energy Analytics Dashboard")

        # ---------------- KPIs ----------------

        st.subheader("📈 Key Performance Indicators")

        col1, col2, col3, col4 = st.columns(4)

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

        col4.metric(
            "Total Energy Generated",
            f"{data['energy_output'].sum():.2f} kWh"
        )

        st.markdown("---")

        # ---------------- Solar vs Energy ----------------

        st.subheader("☀ Solar Radiation vs Energy Output")

        fig1 = px.scatter(
            data,
            x="solar_radiation",
            y="energy_output",
            color="temperature",
            size="wind_speed",
            title="Solar Radiation Impact on Energy Production"
        )

        st.plotly_chart(fig1, use_container_width=True)

        # ---------------- Wind vs Energy ----------------

        st.subheader("💨 Wind Speed vs Energy Output")

        fig2 = px.scatter(
            data,
            x="wind_speed",
            y="energy_output",
            color="humidity",
            title="Wind Speed Impact on Energy Production"
        )

        st.plotly_chart(fig2, use_container_width=True)

        # ---------------- Energy Distribution ----------------

        st.subheader("📊 Energy Output Distribution")

        fig3 = px.histogram(
            data,
            x="energy_output",
            nbins=30,
            title="Distribution of Energy Output"
        )

        st.plotly_chart(fig3, use_container_width=True)

        # ---------------- Correlation Heatmap ----------------

        st.subheader("🔥 Feature Correlation Heatmap")

        corr = data.corr(numeric_only=True)

        fig4 = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="viridis"
        )

        st.plotly_chart(fig4, use_container_width=True)

        # ---------------- Model Performance ----------------

        st.subheader("📈 Model Performance")

        if model is not None:

            X = data[["temperature","humidity","wind_speed","solar_radiation"]]
            y = data["energy_output"]

            y_pred = model.predict(X)

            r2 = r2_score(y, y_pred)
            rmse = mean_squared_error(y, y_pred, squared=False)

            col1, col2 = st.columns(2)

            col1.metric("R² Score", f"{r2:.2f}")
            col2.metric("RMSE", f"{rmse:.2f}")

        # ---------------- Energy Trend ----------------

        st.subheader("📈 Energy Production Trend")

        fig5 = px.line(
            data,
            y="energy_output",
            title="Energy Output Trend"
        )

        st.plotly_chart(fig5, use_container_width=True)

# ==================================================
# TAB 3 : DATASET
# ==================================================

with tab3:

    if data is None:
        st.error("Dataset not found.")
    else:

        st.header("📂 Dataset Explorer")

        st.subheader("Dataset Preview")
        st.dataframe(data.head(20))

        st.subheader("Dataset Statistics")
        st.write(data.describe())

        st.subheader("Dataset Columns")
        st.write(list(data.columns))

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
st.caption("🚀 Built with Streamlit | Renewable Energy Prediction System")