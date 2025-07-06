import pandas as pd


def get_h2h_stats(home_team: str, away_team: str, date: pd.Timestamp, df: pd.DataFrame) -> dict:
    h2h_matches = df[
        (((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)) |
         ((df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team))) &
        (df['Date'] < date)
    ]

    if h2h_matches.empty:
        return {
            'h2h_home_win_rate': 0.0,
            'h2h_draw_rate': 0.0,
            'h2h_goal_diff_avg': 0.0,
            'h2h_matches_count': 0
        }

    home_wins = 0
    draws = 0
    goal_diffs = []

    for _, row in h2h_matches.iterrows():
        if row['HomeTeam'] == home_team:
            # normal mecz: home vs away
            if row['FTR'] == 'H':
                home_wins += 1
            elif row['FTR'] == 'D':
                draws += 1
            diff = row['FTHG'] - row['FTAG']
        else:
            # odwrotny mecz: away vs home
            if row['FTR'] == 'A':
                home_wins += 1
            elif row['FTR'] == 'D':
                draws += 1
            diff = row['FTAG'] - row['FTHG']  # zamiana, bo gospodarze są teraz jako goście

        goal_diffs.append(diff)

    total = len(h2h_matches)
    return {
        'h2h_home_win_rate': home_wins / total,
        'h2h_draw_rate': draws / total,
        'h2h_goal_diff_avg': sum(goal_diffs) / total,
        'h2h_matches_count': total
    }


