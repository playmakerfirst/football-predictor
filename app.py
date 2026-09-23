import streamlit as st
import requests
import pandas as pd
import time
from scipy.stats import poisson

st.set_page_config(page_title="توقع المباريات", layout="centered")

API_KEY = st.secrets["API_KEY"]
headers = {"X-Auth-Token": API_KEY}

competitions = {
    "PL": "الدوري الإنجليزي",
    "PD": "الدوري الإسباني",
    "BL1": "الدوري الألماني",
    "SA": "الدوري الإيطالي",
    "FL1": "الدوري الفرنسي",
    "DED": "الدوري الهولندي",
    "PPL": "الدوري البرتغالي"
}

@st.cache_data(ttl=3600)
def load_matches():
    all_matches = []
    for code, name in competitions.items():
        response = requests.get(
            f"https://api.football-data.org/v4/competitions/{code}/matches?status=FINISHED",
            headers=headers
        )
        comp_data = response.json()
        if "matches" in comp_data:
            for match in comp_data["matches"]:
                all_matches.append({
                    "الدوري": name,
                    "التاريخ": match["utcDate"][:10],
                    "الفريق_المضيف": match["homeTeam"]["name"],
                    "الفريق_الضيف": match["awayTeam"]["name"],
                    "أهداف_المضيف": match["score"]["fullTime"]["home"],
                    "أهداف_الضيف": match["score"]["fullTime"]["away"]
                })
        time.sleep(7)
    return pd.DataFrame(all_matches)

def calculate_points(row, team_name):
    if row["الفريق_المضيف"] == team_name:
        if row["أهداف_المضيف"] > row["أهداف_الضيف"]:
            return 3
        elif row["أهداف_المضيف"] == row["أهداف_الضيف"]:
            return 1
        else:
            return 0
    elif row["الفريق_الضيف"] == team_name:
        if row["أهداف_الضيف"] > row["أهداف_المضيف"]:
            return 3
        elif row["أهداف_الضيف"] == row["أهداف_المضيف"]:
            return 1
        else:
            return 0
    return None

def get_team_form(team_name, matches_df, last_n=5):
    team_matches = matches_df[
        (matches_df["الفريق_المضيف"] == team_name) |
        (matches_df["الفريق_الضيف"] == team_name)
    ].copy()
    team_matches["نقاط"] = team_matches.apply(lambda row: calculate_points(row, team_name), axis=1)
    return team_matches.tail(last_n)["نقاط"].sum()

def get_team_stats_weighted(team_name, matches_df, recent_n=8):
    home_matches = matches_df[matches_df["الفريق_المضيف"] == team_name].tail(recent_n)
    away_matches = matches_df[matches_df["الفريق_الضيف"] == team_name].tail(recent_n)
    return (
        home_matches["أهداف_المضيف"].mean(),
        home_matches["أهداف_الضيف"].mean(),
        away_matches["أهداف_الضيف"].mean(),
        away_matches["أهداف_المضيف"].mean()
    )

def predict_match_weighted(home_team, away_team, matches_df, max_goals=5, recent_n=8):
    league_avg_home = matches_df["أهداف_المضيف"].mean()
    league_avg_away = matches_df["أهداف_الضيف"].mean()
    home_scored, home_conceded, _, _ = get_team_stats_weighted(home_team, matches_df, recent_n)
    _, _, away_scored, away_conceded = get_team_stats_weighted(away_team, matches_df, recent_n)

    home_attack_strength = home_scored / league_avg_home
    away_defense_strength = away_conceded / league_avg_away
    home_expected_goals = home_attack_strength * away_defense_strength * league_avg_home

    away_attack_strength = away_scored / league_avg_away
    home_defense_strength = home_conceded / league_avg_home
    away_expected_goals = away_attack_strength * home_defense_strength * league_avg_away

    home_win, draw, away_win = 0, 0, 0
    for i in range(max_goals):
        for j in range(max_goals):
            p = poisson.pmf(i, home_expected_goals) * poisson.pmf(j, away_expected_goals)
            if i > j:
                home_win += p
            elif i == j:
                draw += p
            else:
                away_win += p
    return home_win, draw, away_win, home_expected_goals, away_expected_goals

def head_to_head(home_team, away_team, matches_df):
    h2h_matches = matches_df[
        ((matches_df["الفريق_المضيف"] == home_team) & (matches_df["الفريق_الضيف"] == away_team)) |
        ((matches_df["الفريق_المضيف"] == away_team) & (matches_df["الفريق_الضيف"] == home_team))
    ]
    if h2h_matches.empty:
        return None
    home_wins, away_wins, draws = 0, 0, 0
    for _, row in h2h_matches.iterrows():
        if row["أهداف_المضيف"] > row["أهداف_الضيف"]:
            if row["الفريق_المضيف"] == home_team:
                home_wins += 1
            else:
                away_wins += 1
        elif row["أهداف_المضيف"] < row["أهداف_الضيف"]:
            if row["الفريق_الضيف"] == home_team:
                home_wins += 1
            else:
                away_wins += 1
        else:
            draws += 1
    return len(h2h_matches), home_wins, away_wins, draws

st.title("⚽ توقع نتائج المباريات")
st.caption("يعتمد على الفورم الأخير، المواجهات المباشرة، ومعدل الأهداف")

with st.spinner("جاري تحميل بيانات المباريات..."):
    all_matches_df = load_matches()

league = st.selectbox("اختر الدوري", list(competitions.values()))

league_teams = sorted(set(
    all_matches_df[all_matches_df["الدوري"] == league]["الفريق_المضيف"]
).union(set(
    all_matches_df[all_matches_df["الدوري"] == league]["الفريق_الضيف"]
)))

col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("الفريق المضيف", league_teams, key="home")
with col2:
    away_team = st.selectbox("الفريق الضيف", league_teams, key="away")

if st.button("توقع المباراة", type="primary"):
    if home_team == away_team:
        st.warning("اختر فريقين مختلفين")
    else:
        home_form = get_team_form(home_team, all_matches_df)
        away_form = get_team_form(away_team, all_matches_df)

        h2h = head_to_head(home_team, away_team, all_matches_df)

        home_win, draw, away_win, home_goals, away_goals = predict_match_weighted(
            home_team, away_team, all_matches_df
        )

        st.subheader(f"{home_team} ضد {away_team}")

        c1, c2 = st.columns(2)
        c1.metric(f"فورم {home_team}", f"{home_form} نقطة")
        c2.metric(f"فورم {away_team}", f"{away_form} نقطة")

        st.write("### المواجهات المباشرة")
        if h2h:
            total, hw, aw, d = h2h
            st.write(f"عدد المواجهات: {total} | فوز {home_team}: {hw} | فوز {away_team}: {aw} | تعادل: {d}")
        else:
            st.write("لا توجد مواجهات سابقة بالبيانات المتاحة")

        st.write("### توقع النتيجة")
        c1, c2, c3 = st.columns(3)
        c1.metric(f"فوز {home_team}", f"{round(home_win*100,1)}%")
        c2.metric("تعادل", f"{round(draw*100,1)}%")
        c3.metric(f"فوز {away_team}", f"{round(away_win*100,1)}%")

        st.info(f"متوقع عدد الأهداف: {round(home_goals,1)} - {round(away_goals,1)}")
