import os
import pandas as pd
from features.elo import compute_elo_ratings

def get_current_elo_ranking() -> pd.DataFrame:
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'LaLiga_Matches.csv')

    df = pd.read_csv(csv_path)
    df = df.sort_values('Date')
    df = compute_elo_ratings(df)

    latest_elos = {}

    for team in set(df['HomeTeam']).union(df['AwayTeam']):
        home_matches = df[df['HomeTeam'] == team]
        away_matches = df[df['AwayTeam'] == team]

        if not home_matches.empty:
            home_elo = home_matches.iloc[-1]['home_elo']
        else:
            home_elo = None

        if not away_matches.empty:
            away_elo = away_matches.iloc[-1]['away_elo']
        else:
            away_elo = None

        values = [v for v in [home_elo, away_elo] if v is not None]
        latest_elos[team] = sum(values) / len(values) if values else 1500  # fallback na 1500

    # <-- UWAGA: return musi być POZA pętlą
    ranking_df = pd.DataFrame(list(latest_elos.items()), columns=['Team', 'ELO'])
    ranking_df = ranking_df.sort_values('ELO', ascending=False).reset_index(drop=True)
    return ranking_df
