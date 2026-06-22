import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

os.makedirs(
    "model",
    exist_ok=True
)

# Load dataset
df = pd.read_csv(
    "data/ml_ready_data.csv"
)

# Features
X = df[
    [
        "Company",
        "RocketStatus",
        "Country",
        "Price"
    ]
]

# Target
y = df["MissionStatus"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Random Forest Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train
model.fit(
    X_train,
    y_train
)

# Predict
predictions = model.predict(
    X_test
)

# Accuracy
accuracy = accuracy_score(
    y_test,
    predictions
)

print(
    f"\nAccuracy: {accuracy*100:.2f}%"
)

# Save model
joblib.dump(
    model,
    "model/mission_model.pkl"
)

print(
    "\nModel saved successfully!"
)