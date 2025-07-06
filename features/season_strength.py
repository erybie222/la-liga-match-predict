import pandas as pd


def get_team_season_strength(team: str, season: str, date: pd.Timestamp, df: pd.DataFrame) -> float:
    # wybierz mecze tej drużyny w danym sezonie przed podaną datą
    past_matches = df[
        ((df['HomeTeam'] == team) | (df['AwayTeam'] == team)) &
        (df['Season'] == season) &
        (df['Date'] < date)
    ]
    
    if past_matches.empty:
        return 0.0

    points = 0
    for _, row in past_matches.iterrows():
        if row['HomeTeam'] == team:
            if row['FTR'] == 'H':
                points += 3
            elif row['FTR'] == 'D':
                points += 1
        elif row['AwayTeam'] == team:
            if row['FTR'] == 'A':
                points += 3
            elif row['FTR'] == 'D':
                points += 1

    return points / len(past_matches)
