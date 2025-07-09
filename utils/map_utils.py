import folium
import pandas as pd
from utils.elo_utils import get_current_elo_ranking
import os
def generate_elo_map():
    df_elo = get_current_elo_ranking()
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    df_locations = pd.read_csv(os.path.join(base_path, 'data', 'stadium_locations.csv'))

    print("ELO teams:", set(df_elo['Team']))
    print("Location teams:", set(df_locations['Team']))


    df = pd.merge(df_elo, df_locations, on="Team")

    print("Common:", set(df_elo['Team']) & set(df_locations['Team']))
    print("Missing in locations:", set(df_elo['Team']) - set(df_locations['Team']))
    print("Missing in elo:", set(df_locations['Team']) - set(df_elo['Team']))


    m = folium.Map(location=[40.0, -4.0], zoom_start=6)

    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']], 
            radius=row['ELO'] / 100,
            popup=f"{row['Team']} – ELO: {int(row['ELO'])}",
            color='blue',
            fill=True,
            fill_opacity=0.7
        ).add_to(m)

    return m
