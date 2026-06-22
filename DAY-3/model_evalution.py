import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# Load model
model = joblib.load(
    "model/mission_model.pkl"
)

# Load data
df = pd.read_csv(
    "data/ml_ready_data.csv"
)

X = df[
    [
        "Company",
        "RocketStatus",
        "Country",
        "Price"
    ]
]

y = df["MissionStatus"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

predictions = model.predict(
    X_test
)

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        predictions
    )
)

cm = confusion_matrix(
    y_test,
    predictions
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.title(
    "Mission Status Confusion Matrix"
)

plt.savefig(
    "images/confusion_matrix.png"
)

plt.show()

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

plt.figure(figsize=(8,5))

plt.bar(
    importance["Feature"],
    importance["Importance"]
)

plt.title(
    "Feature Importance"
)

plt.tight_layout()

plt.savefig(
    "images/feature_importance.png"
)

plt.show()