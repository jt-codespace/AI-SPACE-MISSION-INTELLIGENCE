import pandas as pd
import joblib

# Load model
model = joblib.load(
    "model/mission_model.pkl"
)

# Load encoders
encoders = joblib.load(
    "model/encoders.pkl"
)

# Example values
company = "SpaceX"
rocket_status = "Active"
country = "USA"
price = 50

# Encode inputs
company_encoded = (
    encoders["Company"]
    .transform([company])[0]
)

rocket_encoded = (
    encoders["RocketStatus"]
    .transform([rocket_status])[0]
)

country_encoded = (
    encoders["Country"]
    .transform([country])[0]
)

sample = pd.DataFrame({
    "Company": [company_encoded],
    "RocketStatus": [rocket_encoded],
    "Country": [country_encoded],
    "Price": [price]
})

prediction = model.predict(
    sample
)

probability = model.predict_proba(
    sample
)

status = (
    encoders["MissionStatus"]
    .inverse_transform(prediction)
)

print(
    "\nPredicted Mission Status:",
    status[0]
)

print(
    f"Confidence: {max(probability[0])*100:.2f}%"
)