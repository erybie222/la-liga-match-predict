import os
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from features.form_stats import home_form_stats, away_form_stats
from features.season_strength import get_team_season_strength
from features.h2h import get_h2h_stats
from features.elo import compute_elo_ratings
from sklearn.model_selection import train_test_split


def prepare_dataset(N=5):
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'LaLiga_Matches.csv')
    dataset = pd.read_csv(csv_path, parse_dates=['Date'])
    dataset['Date'] = pd.to_datetime(dataset['Date'], dayfirst=True)
    dataset = dataset.sort_values('Date')

    dataset = compute_elo_ratings(dataset)

    home_stats_list = []
    away_stats_list = []
    home_strength_list = []
    away_strength_list = []
    h2h_stats_list = []

    for idx, row in dataset.iterrows():
        date = row['Date']
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        season = row['Season']

        home_stats = home_form_stats(date, N, home_team, dataset)
        away_stats = away_form_stats(date, N, away_team, dataset)
        home_strength = get_team_season_strength(home_team, season, date, dataset)
        away_strength = get_team_season_strength(away_team, season, date, dataset)
        h2h_stats = get_h2h_stats(home_team, away_team, date, dataset)

        home_stats_list.append(home_stats)
        away_stats_list.append(away_stats)
        home_strength_list.append(home_strength)
        away_strength_list.append(away_strength)
        h2h_stats_list.append(h2h_stats)

    home_df = pd.DataFrame(home_stats_list).add_prefix('home_')
    away_df = pd.DataFrame(away_stats_list).add_prefix('away_')
    h2h_df = pd.DataFrame(h2h_stats_list)

    dataset_features = pd.concat([
        dataset.reset_index(drop=True),
        home_df,
        away_df,
        h2h_df
    ], axis=1)

    dataset_features['home_strength'] = home_strength_list
    dataset_features['away_strength'] = away_strength_list
    dataset_features['form_diff'] = dataset_features['home_form_points'] - dataset_features['away_form_points']
    dataset_features['goal_diff_diff'] = dataset_features['home_avg_goal_diff'] - dataset_features['away_avg_goal_diff']
    dataset_features['win_ratio_diff'] = dataset_features['home_win_ratio'] - dataset_features['away_win_ratio']

    le_ftr = LabelEncoder()
    dataset_features['FTR_encoded'] = le_ftr.fit_transform(dataset_features['FTR'])

    feature_columns = [
        'home_strength', 'away_strength',
        'home_avg_goals_for', 'home_avg_goals_against',
        'home_avg_goal_diff',
        'away_avg_goals_for', 'away_avg_goals_against',
        'away_avg_goal_diff',
        'goal_diff_diff',
        'h2h_home_win_rate', 'h2h_draw_rate', 'h2h_goal_diff_avg', 'h2h_matches_count',
        'home_elo', 'away_elo'
    ]

    X = dataset_features[feature_columns]
    y = dataset_features['FTR_encoded']

    return X, y, le_ftr



def get_preprocessed_data():
    X, y, le_ftr = prepare_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    return X, y, X_train, X_test, y_train, y_test, le_ftr


def prepare_match_features(home_team: str, away_team: str) -> pd.DataFrame:
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'LaLiga_Matches.csv')
    df = pd.read_csv(csv_path, parse_dates=['Date'])
    df = compute_elo_ratings(df)
    last_date = df['Date'].max()
    last_season = df[df['Date'] == last_date]['Season'].values[0]

    home_stats = home_form_stats(last_date, 5, home_team, df)
    away_stats = away_form_stats(last_date, 5, away_team, df)
    home_strength = get_team_season_strength(home_team, last_season, last_date, df)
    away_strength = get_team_season_strength(away_team, last_season, last_date, df)
    h2h_stats = get_h2h_stats(home_team, away_team, last_date, df)

    latest_home_elo = df[df['HomeTeam'] == home_team]['home_elo'].iloc[-1] if not df[df['HomeTeam'] == home_team].empty else 1500
    latest_away_elo = df[df['AwayTeam'] == away_team]['away_elo'].iloc[-1] if not df[df['AwayTeam'] == away_team].empty else 1500

    features = {
        'home_strength': home_strength,
        'away_strength': away_strength,
        'home_avg_goals_for': home_stats['avg_goals_for'],
        'home_avg_goals_against': home_stats['avg_goals_against'],
        'home_avg_goal_diff': home_stats['avg_goal_diff'],
        'away_avg_goals_for': away_stats['avg_goals_for'],
        'away_avg_goals_against': away_stats['avg_goals_against'],
        'away_avg_goal_diff': away_stats['avg_goal_diff'],
        'goal_diff_diff': home_stats['avg_goal_diff'] - away_stats['avg_goal_diff'],
        'h2h_home_win_rate': h2h_stats['h2h_home_win_rate'],
        'h2h_draw_rate': h2h_stats['h2h_draw_rate'],
        'h2h_goal_diff_avg': h2h_stats['h2h_goal_diff_avg'],
        'h2h_matches_count': h2h_stats['h2h_matches_count'],
        'home_elo': latest_home_elo,
        'away_elo': latest_away_elo
    }

    return pd.DataFrame([features])
