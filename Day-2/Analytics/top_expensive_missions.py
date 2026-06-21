import pandas as pd

df = pd.read_csv("data/cleaned_space_missions.csv")

top10 = (
    df.sort_values(by="Price",ascending=False).head(10)
)
print(
    top10[
        [
            "Company",
            "Mission",
            "Price"
        ]
    ]
)
