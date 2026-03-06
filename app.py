# ==========================================================
# Renewable Energy Prediction System
# Fully Deployment-Ready Version
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================================
# Page Config
# ==========================================================

st.set_page_config(
    page_title="Renewable Energy Prediction System",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Renewable Energy Output Prediction System")
st.markdown("Machine Learning Dashboard for Renewable Energy Analysis")

# ==========================================================
# Sidebar
# ==========================================================

st.sidebar.header("⚙ System Settings")

page = st.sidebar.radio(
    "Select Module",
    [
        "Energy Prediction",
        "Region Analysis",
        "Country Analysis",
        "Dataset Dashboard",
        "Correlation Heatmap"
    ]
)

# ==========================================================
# Paths
# ==========================================================

DATA_DIR = "data"
DATA_FILE = f"{DATA_DIR}/renewable_energy_dataset.csv"
MODEL_FILE = "model.pkl"

# ==========================================================
# Dataset Loader
# ==========================================================

@st.cache_data
def load_data():

    os.makedirs(DATA_DIR, exist_ok=True)

    required_columns = [
        "Temperature",
        "Rainfall",
        "Humidity",
        "WindSpeed",
        "SolarRadiation",
        "EnergyOutput"
    ]

    if os.path.exists(DATA_FILE):
        data = pd.read_csv(DATA_FILE)
    else:
        data = None

    # regenerate dataset if missing or incorrect
    if data is None or not all(col in data.columns for col in required_columns):

        np.random.seed(42)

        regions = ["Asia","Europe","Africa","North America","South America","Australia"]
        countries = ["India","USA","Germany","Brazil","China","Australia"]

        data = pd.DataFrame({

            "Temperature": np.random.uniform(10,40,300),
            "Rainfall": np.random.uniform(50,250,300),
            "Humidity": np.random.uniform(30,90,300),
            "WindSpeed": np.random.uniform(1,12,300),
            "SolarRadiation": np.random.uniform(300,900,300),
            "Region": np.random.choice(regions,300),
            "Country": np.random.choice(countries,300)

        })

        data["EnergyOutput"] = (
            data["SolarRadiation"]*0.6
            + data["WindSpeed"]*15
            - data["Humidity"]*0.4
            + np.random.normal(0,10,300)
        )

        data.to_csv(DATA_FILE, index=False)

    return data


data = load_data()

# ==========================================================
# Model Loader
# ==========================================================

features = [
    "Temperature",
    "Rainfall",
    "Humidity",
    "WindSpeed",
    "SolarRadiation"
]

target = "EnergyOutput"

@st.cache_resource
def load_model():

    if os.path.exists(MODEL_FILE):

        model = joblib.load(MODEL_FILE)

    else:

        from sklearn.ensemble import RandomForestRegressor

        if not all(col in data.columns for col in features + [target]):
            st.error("Dataset missing required ML columns.")
            st.write("Current columns:", list(data.columns))
            st.stop()

        X = data[features]
        y = data[target]

        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )

        model.fit(X,y)

        joblib.dump(model,MODEL_FILE)

    return model


model = load_model()

# ==========================================================
# ENERGY PREDICTION
# ==========================================================

if page == "Energy Prediction":

    st.header("🔮 Predict Renewable Energy Output")

    col1,col2 = st.columns(2)

    with col1:
        temperature = st.slider("Temperature (°C)",0,50,25)
        rainfall = st.slider("Rainfall (mm)",0,300,150)
        humidity = st.slider("Humidity (%)",0,100,60)

    with col2:
        wind_speed = st.slider("Wind Speed (m/s)",0,20,5)
        solar_radiation = st.slider("Solar Radiation (W/m²)",0,1000,600)

    if st.button("Predict Energy Output"):

        input_df = pd.DataFrame({
            "Temperature":[temperature],
            "Rainfall":[rainfall],
            "Humidity":[humidity],
            "WindSpeed":[wind_speed],
            "SolarRadiation":[solar_radiation]
        })

        prediction = model.predict(input_df)[0]

        st.success(f"⚡ Predicted Renewable Energy Output: {prediction:.2f} kWh")

# ==========================================================
# REGION ANALYSIS
# ==========================================================

elif page == "Region Analysis":

    st.header("🌍 Region-wise Energy Production")

    region_energy = data.groupby("Region")[target].mean().reset_index()

    fig = px.bar(
        region_energy,
        x="Region",
        y=target,
        color="Region",
        title="Average Energy Output by Region"
    )

    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# COUNTRY ANALYSIS
# ==========================================================

elif page == "Country Analysis":

    st.header("🌎 Country-wise Energy Production")

    country_energy = data.groupby("Country")[target].mean().reset_index()

    fig = px.bar(
        country_energy,
        x="Country",
        y=target,
        color="Country",
        title="Average Energy Output by Country"
    )

    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# DATASET DASHBOARD
# ==========================================================

elif page == "Dataset Dashboard":

    st.header("📊 Dataset Dashboard")

    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    st.subheader("Statistics")
    st.write(data.describe())

    col1,col2 = st.columns(2)

    with col1:
        fig = px.histogram(data,x="Temperature",nbins=30,title="Temperature Distribution")
        st.plotly_chart(fig)

    with col2:
        fig = px.histogram(data,x="SolarRadiation",nbins=30,title="Solar Radiation Distribution")
        st.plotly_chart(fig)

    st.subheader("Wind Speed vs Energy Output")

    fig = px.scatter(
        data,
        x="WindSpeed",
        y="EnergyOutput",
        color="Region",
        size="SolarRadiation",
        hover_data=["Country"]
    )

    st.plotly_chart(fig)

# ==========================================================
# CORRELATION HEATMAP
# ==========================================================

elif page == "Correlation Heatmap":

    st.header("🔥 Feature Correlation Heatmap")

    corr = data.corr(numeric_only=True)

    fig,ax = plt.subplots(figsize=(10,6))

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax
    )

    st.pyplot(fig)

# ==========================================================
# Footer
# ==========================================================

st.markdown("---")

st.markdown(
"""
### Project Information

This dashboard predicts renewable energy output using environmental factors.

Features:

• Energy prediction  
• Region analysis  
• Country analysis  
• Dataset dashboard  
• Correlation heatmap  

Built with **Streamlit + Machine Learning**.
"""
)