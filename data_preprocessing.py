import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from functions import home_form_stats, away_form_stats


dataset = pd.read_csv('LaLiga_Matches.csv')

# print(dataset.shape)
# print(dataset.columns)
# print(dataset.head())

dataset['Date'] = pd.to_datetime(dataset['Date'], dayfirst=True)
dataset = dataset.sort_values('Date')

home_stats_list = []
away_stats_list = []

N= 5

for idx, row in dataset.iterrows():
    date = row['Date']
    home_team = row['HomeTeam']
    away_team = row['AwayTeam']

    home_stats = home_form_stats(date, N , home_team, dataset)
    away_stats = away_form_stats(date, N , away_team,dataset )

    home_stats_list.append(home_stats)
    away_stats_list.append(away_stats)

home_stats_df = pd.DataFrame(home_stats_list).add_prefix('home_')
away_stats_df = pd.DataFrame(away_stats_list).add_prefix('away_')

dataset_features = pd.concat([dataset.reset_index(drop=True), home_stats_df, away_stats_df], axis=1)

print(dataset_features.head(10))

