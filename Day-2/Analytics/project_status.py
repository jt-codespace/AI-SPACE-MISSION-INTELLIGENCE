import pandas as pd

df = pd.read_csv("data/cleaned_space_missions.csv")

total_missions = len(df)

successful = len(
    df[
        df["MissionStatus"]
        == "Success"
    ]
)
failed  = total_missions - successful

success_rate = (successful/total_missions)*100

print(f"Total Missions: {total_missions}")
print(f"Successful: {successful}")
print(f"failed: {failed}")
print(f"Success_Rate: {success_rate}")