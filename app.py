from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return "Renewable Energy Prediction API is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = np.array([[
        data["carbon_emission"],
        data["energy_output"],
        data["renewability_index"]
    ]])
    prediction = model.predict(features)
    return jsonify({"adoption_prediction": int(prediction[0])})

@app.route("/ui")
def ui():
    return render_template("index.html", result=None)

@app.route("/predict_form", methods=["POST"])
def predict_form():
    ce = float(request.form["carbon_emission"])
    eo = float(request.form["energy_output"])
    ri = float(request.form["renewability_index"])

    features = np.array([[ce, eo, ri]])
    prediction = model.predict(features)[0]

    result = "Adopted" if prediction == 1 else "Not Adopted"
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)