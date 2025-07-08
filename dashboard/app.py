import csv
from datetime import datetime
import sys
import os

from matplotlib import pyplot as plt
import shap

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.simulation import simulate_season
from features.elo import compute_elo_ratings
import streamlit as st
import pandas as pd
import joblib
from utils.preprocessing import prepare_dataset, prepare_match_features, get_preprocessed_data
from utils.elo_utils import get_current_elo_ranking
from models.compare_models import compare_models
from models.evaluate import evaluate_model
from utils.shap_explainer import get_shap_explanation

model = joblib.load('../models/best_model_tuned_smote.pkl')
label_encoder = joblib.load('../models/label_encoder_smote.pkl')

df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'LaLiga_Matches.csv'), parse_dates=['Date'])
teams = sorted(set(df['HomeTeam']) | set(df["AwayTeam"]))



view = st.sidebar.radio("📊 Choose view:", [
    "🧠 Match prediction",
    "📈 ELO ranking",
    "🏆 Season Simulation",
    "📌 Team Overview",
    "🔬 Model Comparison",
    "📐 Model Evaluation",
    "🔎 Explain Prediction",
    "📊 Hyperparameter Tuning Results"
])

if view == "📈 ELO ranking":
    st.title("📈 ELO ranking of La Liga teams")
    df_ranking = get_current_elo_ranking()
    st.dataframe(df_ranking, use_container_width=True)

    selected_team = st.selectbox("📍 Choos team to analyze the change of ELO in timeline", df_ranking['Team'])

    df_all = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'LaLiga_Matches.csv'), parse_dates=['Date'])
    df_all = df_all.sort_values('Date')
    df_all = compute_elo_ratings(df_all)

    elo_values = []
    dates = []

    for _, row in df_all.iterrows():
        if row['HomeTeam'] == selected_team:
            elo_values.append(row['home_elo'])
            dates.append(row['Date'])
        elif row['AwayTeam'] == selected_team:
            elo_values.append(row['away_elo'])
            dates.append(row['Date'])

    df_plot = pd.DataFrame({'Date': pd.to_datetime(dates, dayfirst=True), 'ELO': elo_values})
    df_plot = df_plot.sort_values('Date')  
    st.line_chart(df_plot.set_index('Date'))



elif view == "🏆 Season Simulation":
    selected_season = st.selectbox("📅 Choose season", df["Season"].unique()[::-1])
    if st.button("⚔️ Simulate full season") or "df_progress" in st.session_state:
        if "df_progress" not in st.session_state:
            table, df_progress = simulate_season(selected_season, df, model, label_encoder)
            st.session_state.df_progress = df_progress
            st.session_state.table = table
        else:
            df_progress = st.session_state.df_progress
            table = st.session_state.table

            st.subheader(f"📊 Predicted Final Standings – {selected_season}")
            st.dataframe(table)

        df_progress['Date'] = pd.to_datetime(df_progress['Date'], dayfirst=True)
        df_progress = df_progress.sort_values('Date')

        leader_points = df_progress.groupby('Date')['Points'].max().reset_index(name='LeaderPoints')

        df_progress = df_progress.merge(leader_points, on='Date')
        df_progress['BehindLeader'] = df_progress['LeaderPoints'] - df_progress['Points']

        selected_team = st.selectbox("📍 Choose team to track vs leader", sorted(df_progress['Team'].unique()))
        team_df = df_progress[df_progress['Team'] == selected_team]

        st.line_chart(team_df.set_index('Date')['BehindLeader'])



elif view == "🧠 Match prediction":

    st.title("⚽ La Liga Match Predictor")

    history_path = os.path.join(os.path.dirname(__file__), '..', 'history', 'predictions.csv')
    if os.path.exists(history_path):
        df_history = pd.read_csv(history_path)
        st.dataframe(df_history.tail(5).iloc[::-1],use_container_width=True)
    else:
        st.info("No predictions yet.")


    home = st.selectbox("🏠 Choose home team", teams)
    away = st.selectbox("🚗 Choose away team", teams)

    if home == away:
        st.warning("Teams must be different!")
    else:
        if st.button("🔮 Predict result"):
            try:
                features = prepare_match_features(home, away)
                st.subheader("📋 Input features: ")
                st.write(features.T)

                pred = model.predict(features)[0]
                proba = model.predict_proba(features)[0]

                pred_label = label_encoder.inverse_transform([pred])[0]
                
                st.success(f"**Predicted result: {pred_label}**")

                st.subheader("📊 Probabilities:")
                for i, p in enumerate(proba):
                    label = label_encoder.inverse_transform([i])[0]
                    st.write(f"{label}: {round(p*100, 2)}%")

                history_path = os.path.join(os.path.dirname(__file__), '..', 'history', 'predictions.csv')
                os.makedirs(os.path.dirname(history_path), exist_ok=True)

                with open(history_path, mode='a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    if os.stat(history_path).st_size == 0:
                        writer.writerow(["timestamp", "home_team", "away_team", "prediction", "proba_H", "proba_D", "proba_A"])

                    writer.writerow([
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        home,
                        away,
                        pred_label,
                        round(proba[0], 3),
                        round(proba[1], 3),
                        round(proba[2], 3)
                    ])
            except Exception as e:
                st.error(f"Error: {str(e)}")

elif view == "📌 Team Overview":
    st.title("📌 Team Overview")

    selected_team = st.selectbox("🔎 Choose team to analyze", teams)
    selected_season = st.selectbox("📅 Choose season", sorted(df["Season"].unique(), reverse=True))

    team_matches = df[
        ((df["HomeTeam"] == selected_team) | (df["AwayTeam"] == selected_team)) &
        (df["Season"] == selected_season)
    ].sort_values("Date")

    recent_matches = team_matches.tail(5)
    results = []
    for _, row in recent_matches.iterrows():
        if row["HomeTeam"] == selected_team:
            if row["FTHG"] > row["FTAG"]:
                results.append("W")
            elif row["FTHG"] < row["FTAG"]:
                results.append("L")
            else:
                results.append("D")
        else:
            if row["FTAG"] > row["FTHG"]:
                results.append("W")
            elif row["FTAG"] < row["FTHG"]:
                results.append("L")
            else:
                results.append("D")

    goals_for = []
    goals_against = []
    points_over_time = []
    cumulative_points = 0
    dates = []

    for _, row in team_matches.iterrows():
        if row["HomeTeam"] == selected_team:
            gf, ga = row["FTHG"], row["FTAG"]
        else:
            gf, ga = row["FTAG"], row["FTHG"]

        goals_for.append(gf)
        goals_against.append(ga)

        if gf > ga:
            cumulative_points += 3
        elif gf == ga:
            cumulative_points += 1

        points_over_time.append(cumulative_points)
        dates.append(row["Date"])

    st.subheader(f"📅 Match history – {selected_team} ({selected_season})")
    st.dataframe(team_matches[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]])

    st.subheader("📈 Recent form (last 5 matches)")
    st.write(" ➤ " + " - ".join(results[::-1]))  

    st.subheader("⚽ Average goals per match")
    st.write(f"**Scored:** {round(sum(goals_for) / len(goals_for), 2)} | **Conceded:** {round(sum(goals_against) / len(goals_against), 2)}")

    st.subheader("📈 Cumulative Points Over Time")
    st.line_chart(pd.DataFrame({'Points': points_over_time}, index=pd.to_datetime(dates, dayfirst=True)))

elif view == "🔬 Model Comparison":
    st.title("🔬 Model Performance Comparison")

    X, y, _ = prepare_dataset()

    df_results = compare_models(X, y)

    st.dataframe(df_results.style.background_gradient(cmap='Blues'), use_container_width=True)
    st.bar_chart(df_results.set_index("Model")[["Accuracy", "F1 Score"]])
 
elif view == "📐 Model Evaluation":
    st.title("📐 Model Evaluation")

    

    _, _, _, X_test, _, y_test, le = get_preprocessed_data()

    evaluate_model(model, X_test, y_test, le)

elif view == "🔎 Explain Prediction":
    st.title("🔎 Explain Prediction")

    home = st.selectbox("🏠 Home team", teams, key="explain_home")
    away = st.selectbox("🚗 Away team", teams, key="explain_away")

    if home == away:
        st.warning("Choose two different teams.")
    elif st.button("🔍 Explain prediction"):
        try:
            features = prepare_match_features(home, away)
            pred_proba = model.predict_proba(features)[0]
            pred_class_index = pred_proba.argmax()
            pred_class_label = label_encoder.inverse_transform([pred_class_index])[0]

            st.success(f"🔮 Predicted result: {pred_class_label} ({round(pred_proba[pred_class_index]*100, 2)}%)")


            shap_values, explainer = get_shap_explanation(model, features)

            st.subheader(f"🔍 SHAP Waterfall for predicted class ({pred_class_label})")
            shap_single_class = shap_values[0, :, pred_class_index]
            fig1 = plt.figure()
            shap.plots.waterfall(shap_single_class, show=False)
            st.pyplot(fig1)

            st.subheader("📊 SHAP Bar Plot (single prediction)")
            fig2 = plt.figure()
            shap.plots.bar(shap_values[0, :, pred_class_index], max_display=10, show=False)
            st.pyplot(fig2)
        except Exception as e:
            st.error(f"Error: {str(e)}")
elif view == "📊 Hyperparameter Tuning Results":
    st.title("📊 Hyperparameter Tuning Results")

    try:
        df_tuning = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'results', 'tuning_results.csv'))

        metric = st.selectbox("📈 Choose metric to sort by:", ["mean_test_score", "rank_test_score", "mean_fit_time"])
        df_show = df_tuning.sort_values(metric, ascending=(metric != "rank_test_score"))

        st.dataframe(df_show[[
            "params", "mean_test_score", "rank_test_score", "mean_fit_time"
        ]].style.background_gradient(cmap='Greens'), use_container_width=True)

        st.subheader("📊 Best 10 configurations (mean_test_score)")
        st.bar_chart(df_show.nlargest(10, "mean_test_score").set_index("rank_test_score")["mean_test_score"])

    except Exception as e:
        st.error(f"❌ Could not load tuning results: {str(e)}")

