import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/cleaned_space_missions.csv")

mission_counts = df["MissionStatus"].value_counts()

plt.figure(figsize=(7,7))

plt.pie(
    mission_counts,
    labels=mission_counts.index,
    autopct="%1.1f%%"
)

plt.title("Mission Success vs Failure")

plt.savefig("images/mission_status.png")

plt.show()