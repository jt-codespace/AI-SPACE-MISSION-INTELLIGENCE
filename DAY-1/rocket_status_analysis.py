import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/cleaned_space_missions.csv")

rocket_counts = df["RocketStatus"].value_counts()

plt.figure(figsize=(8,5))

rocket_counts.plot(kind="bar")

plt.title("Rocket Status Distribution")

plt.xlabel("Rocket Status")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig("images/rocket_status.png")

plt.show()