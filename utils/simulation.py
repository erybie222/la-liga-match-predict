import os
import pandas as pd
from collections import defaultdict
from utils.preprocessing import prepare_match_features


def simulate_season(season: str, df_matches: pd.DataFrame, model, label_encoder) -> pd.DataFrame:
    season_matches = df_matches[df_matches['Season'] == season].copy()
    season_matches = season_matches.sort_values('Date')
    points_progression = [] 
    table = defaultdict(lambda: {
        'Team': '',
        'MP': 0, 'W': 0, 'D': 0, 'L': 0, 'GF': 0, 'GA': 0, 'Pts': 0
    })

    for _, match in season_matches.iterrows():
        home = match['HomeTeam']
        away = match['AwayTeam']


        try:
            features = prepare_match_features(home, away)
            if features.empty:
                continue

            pred_encoded = model.predict(features)[0]
            pred_label = label_encoder.inverse_transform([pred_encoded])[0] 

             

            points_progression.append({
                'Date': match['Date'],
                'Team': home,
                'Points': table[home]['Pts']
            })
            points_progression.append({
                'Date': match['Date'],
                'Team': away,
                'Points': table[away]['Pts']
            })


            table[home]['Team'] = home
            table[away]['Team'] = away
            table[home]['MP'] += 1
            table[away]['MP'] += 1

        
            if pred_label == 'H':
                table[home]['W'] += 1
                table[away]['L'] += 1
                table[home]['Pts'] += 3
                table[home]['GF'] += 2
                table[home]['GA'] += 1
                table[away]['GF'] += 1
                table[away]['GA'] += 2
            elif pred_label == 'A':
                table[away]['W'] += 1
                table[home]['L'] += 1
                table[away]['Pts'] += 3
                table[away]['GF'] += 2
                table[away]['GA'] += 1
                table[home]['GF'] += 1
                table[home]['GA'] += 2
            else:  
                table[home]['D'] += 1
                table[away]['D'] += 1
                table[home]['Pts'] += 1
                table[away]['Pts'] += 1
                table[home]['GF'] += 1
                table[home]['GA'] += 1
                table[away]['GF'] += 1
                table[away]['GA'] += 1

        except Exception as e:
            print(f"Błąd dla meczu {home} vs {away}: {e}")

    table_df = pd.DataFrame(table.values())
    table_df['GD'] = table_df['GF'] - table_df['GA']
    table_df = table_df.sort_values(by=['Pts', 'GD', 'GF'], ascending=False).reset_index(drop=True)
    table_df.index += 1  

   

    output_path = os.path.join("simulations", f"{season}_simulation.csv")
    os.makedirs("simulations", exist_ok=True)
    table_df.to_csv(output_path, index=False)


    return table_df, pd.DataFrame(points_progression)
