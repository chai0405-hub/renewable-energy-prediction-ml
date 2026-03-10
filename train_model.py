# train_model.py

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------------------------------
# 1. Load Dataset
# ---------------------------------------
data = pd.read_csv("renewable_energy_dataset.csv")

print("Dataset Loaded Successfully")
print(data.head())

# ---------------------------------------
# 2. Define Features and Target
# ---------------------------------------
X = data.drop("energy_output", axis=1)
y = data["energy_output"]

# ---------------------------------------
# 3. Train Test Split
# ---------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# ---------------------------------------
# 4. Create Model
# ---------------------------------------
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    random_state=42
)

# ---------------------------------------
# 5. Train Model
# ---------------------------------------
model.fit(X_train, y_train)

print("Model Training Completed")

# ---------------------------------------
# 6. Predictions
# ---------------------------------------
predictions = model.predict(X_test)

# ---------------------------------------
# 7. Evaluate Model
# ---------------------------------------
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Model Performance")
print("MAE:", mae)
print("R2 Score:", r2)

# ---------------------------------------
# 8. Save Model
# ---------------------------------------
joblib.dump(model, "model.pkl")

print("Model saved successfully as model.pkl")