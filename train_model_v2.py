import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('premier-league-matches.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)   # oldest to newest
df['match_id'] = df.index

# 1. Make one row per TEAM per match (each match becomes two rows)
home = pd.DataFrame({
    'match_id': df['match_id'], 'Date': df['Date'], 'team': df['Home'],
    'scored': df['HomeGoals'], 'conceded': df['AwayGoals'],
    'points': df['FTR'].map({'H': 3, 'D': 1, 'A': 0}),
})
away = pd.DataFrame({
    'match_id': df['match_id'], 'Date': df['Date'], 'team': df['Away'],
    'scored': df['AwayGoals'], 'conceded': df['HomeGoals'],
    'points': df['FTR'].map({'A': 3, 'D': 1, 'H': 0}),
})
long = pd.concat([home, away]).sort_values(['team', 'Date', 'match_id'])

# 2. Rolling form over the last 5 matches, using ONLY earlier matches
N = 5
for col in ['scored', 'conceded', 'points']:
    long[col + '_avg'] = long.groupby('team')[col].transform(
        lambda s: s.shift(1).rolling(N, min_periods=N).mean()
    )

# 3. Attach each team's form back to the original matches
form = long[['match_id', 'team', 'scored_avg', 'conceded_avg', 'points_avg']]

home_form = form.rename(columns={
    'team': 'Home', 'scored_avg': 'home_scored', 'conceded_avg': 'home_conceded', 'points_avg': 'home_points'})
away_form = form.rename(columns={
    'team': 'Away', 'scored_avg': 'away_scored', 'conceded_avg': 'away_conceded', 'points_avg': 'away_points'})

df = df.merge(home_form, on=['match_id', 'Home']).merge(away_form, on=['match_id', 'Away'])
df = df.dropna().sort_values('Date').reset_index(drop=True)   # drop each team's first 5 matches
df['points_diff'] = df['home_points'] - df['away_points']

features = ['home_scored', 'home_conceded', 'home_points',
            'away_scored', 'away_conceded', 'away_points', 'points_diff']
X = df[features]
y = df['FTR']

# 4. Split by date: first 80% of matches train, last 20% test
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]
print("Train rows:", len(X_train), "| Test rows:", len(X_test))

# 5. Train and evaluate
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("\nBaseline (share of each result in the test set):")
print(y_test.value_counts(normalize=True).round(3))
print("\nAccuracy:", round(accuracy_score(y_test, predictions), 3))
print(classification_report(y_test, predictions, zero_division=0))