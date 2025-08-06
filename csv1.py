import pandas as pd

df = pd.read_csv("test.csv")
grouped = df.groupby("Gloss")["Video file"].apply(list).reset_index()
print(grouped)
grouped.to_csv("grouped.csv", index=False)
resultats = df[df["Gloss"] == "HELLO"]["Video file"].unique()
print(resultats)