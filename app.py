import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

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
    return pd.read_csv("data/renewable_energy_dataset.csv")

data = load_data()

# Clean column names
data.columns = data.columns.str.strip()

# --------------------------------------------------
# Load ML Model
# --------------------------------------------------

@st.cache_resource
def load_model():
    if os.path.exists("model.pkl"):
        return joblib.load("model.pkl")
    else:
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

    st.header("📊 Energy Analytics Dashboard")

    # ---------------- KPIs ----------------

    st.subheader("📈 Key Performance Indicators")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Average Output",
        f"{data['Energy_Output'].mean():.2f} kWh"
    )

    col2.metric(
        "Maximum Output",
        f"{data['Energy_Output'].max():.2f} kWh"
    )

    col3.metric(
        "Minimum Output",
        f"{data['Energy_Output'].min():.2f} kWh"
    )

    col4.metric(
        "Total Energy Generated",
        f"{data['Energy_Output'].sum():.2f} kWh"
    )

    st.markdown("---")

    # ---------------- Charts ----------------

    st.subheader("📊 Energy Distribution")

    fig = plt.figure()

    plt.hist(data["Energy_Output"], bins=20)

    plt.xlabel("Energy Output (kWh)")
    plt.ylabel("Frequency")

    st.pyplot(fig)

    # ---------------- Solar vs Energy ----------------

    st.subheader("☀ Solar Radiation vs Energy Output")

    fig = plt.figure()

    plt.scatter(
        data["Solar_Radiation"],
        data["Energy_Output"]
    )

    plt.xlabel("Solar Radiation")
    plt.ylabel("Energy Output")

    st.pyplot(fig)

    # ---------------- Wind vs Energy ----------------

    st.subheader("💨 Wind Speed vs Energy Output")

    fig = plt.figure()

    plt.scatter(
        data["Wind_Speed"],
        data["Energy_Output"]
    )

    plt.xlabel("Wind Speed")
    plt.ylabel("Energy Output")

    st.pyplot(fig)

    # ---------------- Correlation Heatmap ----------------

    st.subheader("🔥 Feature Correlation Heatmap")

    fig = plt.figure()

    sns.heatmap(
        data.corr(),
        annot=True
    )

    st.pyplot(fig)

    # ---------------- Feature Importance ----------------

    if model is not None and hasattr(model, "feature_importances_"):

        st.subheader("⚡ Feature Importance")

        features = [
            "Temperature",
            "Humidity",
            "Wind_Speed",
            "Solar_Radiation"
        ]

        importance = pd.DataFrame({
            "Feature": features,
            "Importance": model.feature_importances_
        })

        st.bar_chart(
            importance.set_index("Feature")
        )

# ==================================================
# TAB 3 : DATASET
# ==================================================

with tab3:

    st.header("📂 Dataset Explorer")

    st.subheader("Dataset Preview")

    st.dataframe(data)

    st.subheader("Dataset Statistics")

    st.write(data.describe())

    st.subheader("Dataset Columns")

    st.write(data.columns)

    # Download button

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

st.caption("🚀 Built with Streamlit | Machine Learning Renewable Energy Prediction System")