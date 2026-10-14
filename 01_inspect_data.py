import pandas as pd

df = pd.read_csv("premier-league-matches.csv")

print(df.shape)
print(df.head())
print(df.dtypes)
print(df.isnull().sum())