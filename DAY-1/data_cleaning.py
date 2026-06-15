import pandas as pd

df = pd.read_csv("data/space_missions.csv", encoding="latin1")

# Remove duplicates
df.drop_duplicates(inplace=True)

# Remove rows with missing MissionStatus
df = df.dropna(subset=["MissionStatus"])

# Convert Price column to numeric
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

# Fill missing Price values with median
df["Price"] = df["Price"].fillna(df["Price"].median())

print("Cleaned Dataset Shape:")
print(df.shape)

# Save cleaned dataset
df.to_csv("data/cleaned_space_missions.csv", index=False)

print("Cleaned dataset saved!")