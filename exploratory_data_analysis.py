import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('premier-league-matches.csv')

# first look
print(df.shape)           # how many rows and columns?
print(df.head())          # what does it actually look like?
print(df.dtypes)          # what type is each column?
print(df.isnull().sum())  # any missing values?

df = pd.read_csv('premier-league-matches.csv')

# decode FTR into readable labels first
df['result'] = df['FTR'].map({'H': 'Home Win', 'A': 'Away Win', 'D': 'Draw'})

print(df['result'].value_counts())
print(df['result'].value_counts(normalize=True).round(3))

# for numeric features
print(df.describe())

# encode result so we can run the correlation
from sklearn.preprocessing import LabelEncoder
le_temp = LabelEncoder()
df['result_encoded'] = le_temp.fit_transform(df['result'])

# quick correlation check
print(df.corr(numeric_only=True)['result_encoded'].sort_values())

import matplotlib.pyplot as plt

df['HomeGoals'].hist(bins=10)
plt.title('Distribution of Home Goals')
plt.show()