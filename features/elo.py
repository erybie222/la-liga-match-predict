import pandas as pd


def compute_elo_ratings(df: pd.DataFrame, k: float = 20, start_rating: float = 1500) -> pd.DataFrame:
    team_elos = {}
    home_elo_list = []
    away_elo_list = []

    for _, row in df.iterrows():
        home = row['HomeTeam']
        away = row['AwayTeam']
        result = row['FTR']

        # Initialize if team not seen before
        if home not in team_elos:
            team_elos[home] = start_rating
        if away not in team_elos:
            team_elos[away] = start_rating

        r_home = team_elos[home]
        r_away = team_elos[away]

        # Expected score
        e_home = 1 / (1 + 10 ** ((r_away - r_home) / 400))
        e_away = 1 - e_home

        # Actual result
        if result == 'H':
            s_home, s_away = 1, 0
        elif result == 'A':
            s_home, s_away = 0, 1
        else:
            s_home, s_away = 0.5, 0.5

        # Update ratings
        team_elos[home] = r_home + k * (s_home - e_home)
        team_elos[away] = r_away + k * (s_away - e_away)

        # Append pre-match ratings
        home_elo_list.append(r_home)
        away_elo_list.append(r_away)

    df = df.copy()
    df['home_elo'] = home_elo_list
    df['away_elo'] = away_elo_list
    return df
