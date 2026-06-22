import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Create model folder
os.makedirs("model", exist_ok=True)

# Load cleaned data
df = pd.read_csv("data/cleaned_space_missions.csv")

# Extract country
df["Country"] = (
    df["Location"]
    .str.split(",")
    .str[-1]
    .str.strip()
)

# Convert mission status into binary classes
df["MissionStatus"] = df["MissionStatus"].replace({
    "Partial Failure": "Failure",
    "Prelaunch Failure": "Failure"
})

# Keep required columns
ml_df = df[
    [
        "Company",
        "RocketStatus",
        "Country",
        "Price",
        "MissionStatus"
    ]
].copy()

# Fill missing prices
ml_df["Price"] = ml_df["Price"].fillna(
    ml_df["Price"].median()
)

# Save encoders
encoders = {}

for col in [
    "Company",
    "RocketStatus",
    "Country",
    "MissionStatus"
]:
    encoder = LabelEncoder()

    ml_df[col] = encoder.fit_transform(
        ml_df[col]
    )

    encoders[col] = encoder

# Save encoders
joblib.dump(
    encoders,
    "model/encoders.pkl"
)

# Save ML-ready dataset
ml_df.to_csv(
    "data/ml_ready_data.csv",
    index=False
)

print("ML-ready dataset created successfully!")

print("\nMission Status Classes:")
print(
    encoders["MissionStatus"].classes_
)