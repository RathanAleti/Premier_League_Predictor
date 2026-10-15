import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('premier-league-matches.csv')

# STEP 1: Encode text as numbers
team_encoder = LabelEncoder()
team_encoder.fit(pd.concat([df['Home'], df['Away']]))   # one shared list of teams
df['Home_encoded'] = team_encoder.transform(df['Home'])
df['Away_encoded'] = team_encoder.transform(df['Away'])

result_encoder = LabelEncoder()
df['result_encoded'] = result_encoder.fit_transform(df['FTR'])
print("Result codes:", dict(zip(result_encoder.classes_, result_encoder.transform(result_encoder.classes_))))

features = ['Home_encoded', 'Away_encoded', 'Season_End_Year', 'Wk']
X = df[features]
y = df['result_encoded']

# STEP 2: Split 80% train / 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Train rows:", len(X_train), "| Test rows:", len(X_test))

# STEP 3: Train
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# STEP 4: Predict
predictions = model.predict(X_test)
print("First 10 predictions:", predictions[:10])