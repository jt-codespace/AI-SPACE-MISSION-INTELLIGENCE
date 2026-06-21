import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/cleaned_space_missions.csv")

cost_df =df[df["Price"] > 0]

plt.figure(figsize=(10,5))

plt.hist(
    cost_df["Price"],
    bins=20
)
plt.title("Mission Cost Distribution")
plt.xlabel("Cost")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig("images/cost_distributon.png")
plt.show()