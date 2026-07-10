import pandas as pd
import matplotlib.pylab as plt

df = pd.read_csv("BASE_DIR/data/cleaned_space_missions.csv")

df["Date"] = pd.to_datetime(
    df["Date"],
    format="%Y-%m-%d",
    errors="coerce"
)     

df["Year"] = df["Date"].dt.year

launches_per_year = df["Year"].value_counts().sort_index()

plt.figure(figsize=(10,5))

plt.plot(
    launches_per_year.index,
    launches_per_year.values,
    marker="o"
)
plt.title("Space Launches Per Year")
plt.tight_layout()
plt.xlabel("Year")
plt.ylabel("Number Of Launches")

plt.grid(True)

plt.savefig("images/launch_trend.png")

plt.show()
