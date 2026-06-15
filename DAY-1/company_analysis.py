import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv("data/cleaned_space_missions.csv" , encoding="latin1")

top_companies = df["Company"].value_counts().head(10)

plt.figure(figsize=(10,5))
top_companies.plot(kind="bar")

plt.title("Top 10 Launch Companies")
plt.xlabel("Company")
plt.ylabel("Number of Launches")

plt.tight_layout()

os.makedirs("images", exist_ok=True)
plt.savefig("images/top_companies.png")

plt.show()