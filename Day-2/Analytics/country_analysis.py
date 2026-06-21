import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/cleaned_space_missions.csv")

df["Country"] = (
    df["Location"]
    .str.split(",")
    .str[-1]
    .str.strip()
)
Top_Countries = (
    df["Country"]
    .value_counts()
    .head(10)
)

plt.figure(figsize=(10,5))

Top_Countries.plot(kind='bar')

plt.title("Top 10 countries by launches")
plt.xlabel("countries")
plt.ylabel("launch count")

plt.tight_layout()

plt.savefig("images/country_analysis.png")

plt.show()