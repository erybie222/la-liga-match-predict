import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('LaLiga_Matches.csv')

# print(dataset.shape)
# print(dataset.columns)
# print(dataset.head())

dataset['Date'] = pd.to_datetime(dataset['Date'], dayfirst=True)
dataset = dataset.sort_values('Date')

#N is number of last matches
def home_form_stats(date: pd.Timestamp, N: int, team: str, df: pd.DataFrame) -> dict:
    past_matches = df[
        ((df['HomeTeam'] == team) | (df['AwayTeam'] == team)) &
        (df['Date'] < date)
    ].sort_values('Date', ascending=False).head(N)

    if past_matches.empty:
        return {
            "form_points": 0,
            "avg_goals_for": 0.0,
            "avg_goals_against": 0.0,
            "win_ratio": 0.0,
            "avg_goal_diff": 0.0
        }

    points = 0
    goals_for = []
    goals_against = []
    wins = 0

    for _, row in past_matches.iterrows():
        if row['HomeTeam'] == team:
            gf = row['FTHG']
            ga = row['FTAG']
            result = row['FTR']
            if result == 'H':
                points += 3
                wins += 1
            elif result == 'D':
                points += 1
        else:
            gf = row['FTAG']
            ga = row['FTHG']
            result = row['FTR']
            if result == 'A':
                points += 3
                wins += 1
            elif result == 'D':
                points += 1

        goals_for.append(gf)
        goals_against.append(ga)

    n = len(past_matches)
    return {
        "form_points": points,
        "avg_goals_for": sum(goals_for) / n,
        "avg_goals_against": sum(goals_against) / n,
        "win_ratio": wins / n,
        "avg_goal_diff": (sum(goals_for) - sum(goals_against)) / n
    }


def away_form_stats(date: pd.Timestamp, N: int, team: str, df: pd.DataFrame) -> dict:
    past_matches = df[
        ((df['HomeTeam'] == team) | (df['AwayTeam'] == team)) &
        (df['Date'] < date)
    ].sort_values('Date', ascending=False).head(N)

    if past_matches.empty:
        return {
            "form_points": 0,
            "avg_goals_for": 0.0,
            "avg_goals_against": 0.0,
            "win_ratio": 0.0,
            "avg_goal_diff": 0.0
        }

    points = 0
    goals_for = []
    goals_against = []
    wins = 0

    for _, row in past_matches.iterrows():
        if row['AwayTeam'] == team:
            gf = row['FTAG']
            ga = row['FTHG']
            result = row['FTR']
            if result == 'A':
                points += 3
                wins += 1
            elif result == 'D':
                points += 1
        else:
            gf = row['FTHG']
            ga = row['FTAG']
            result = row['FTR']
            if result == 'H':
                points += 3
                wins += 1
            elif result == 'D':
                points += 1

        goals_for.append(gf)
        goals_against.append(ga)

    n = len(past_matches)
    return {
        "form_points": points,
        "avg_goals_for": sum(goals_for) / n,
        "avg_goals_against": sum(goals_against) / n,
        "win_ratio": wins / n,
        "avg_goal_diff": (sum(goals_for) - sum(goals_against)) / n
    }
