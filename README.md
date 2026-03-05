# ⚡ Renewable Energy Output Prediction System

A Machine Learning web application built using **Streamlit** that predicts renewable energy output based on environmental conditions such as temperature, humidity, wind speed, and solar radiation.

---

## 🚀 Project Overview

This project uses a trained Machine Learning model to estimate renewable energy generation based on environmental input parameters.  
The application provides an interactive web interface built with Streamlit.

---

## 📊 Input Features

The model takes the following inputs:

- 🌡 Temperature (°C)
- 💧 Humidity (%)
- 🌬 Wind Speed (m/s)
- ☀ Solar Radiation (W/m²)

---

## 🎯 Output

- 🔋 Predicted Energy Output (kWh)

---

## 🛠 Technologies Used

- Python
- Streamlit
- Scikit-learn
- NumPy
- Pandas
- Joblib

---

## 📂 Project Structure

```
renewable-energy-prediction-ml/
│
├── app.py                          # Streamlit application
├── model.pkl                       # Trained ML model
├── renewable_energy_dataset.csv    # Dataset used for training
├── requirements.txt                # Project dependencies
└── README.md                       # Project documentation
```

---

## ▶️ How to Run Locally

1. Clone the repository:

```
git clone <your-repository-link>
cd renewable-energy-prediction-ml
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Run the Streamlit app:

```
streamlit run app.py
```

---

## 🌐 Deployment

This application can be deployed using:

- Streamlit Cloud
- Render
- Railway

---

## 📌 Future Improvements

- Use real-world renewable energy datasets
- Improve model accuracy with advanced algorithms
- Add data visualization dashboard
- Deploy with live dataset updates

---

## 👨‍💻 Author

Developed as part of a Machine Learning project.

---

⭐ If you like this project, consider giving it a star!