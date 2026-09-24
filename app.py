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
    "PPL": "الدوري البرتغالي",
    "CL": "دوري أبطال أوروبا",
    "EC": "كأس الأمم الأوروبية",
    "ELC": "التشامبيونشيب الإنجليزي",
    "BSA": "الدوري البرازيلي"
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
                    "كود_الدوري": code,
                    "التاريخ": match["utcDate"][:10],
                    "الفريق_المضيف": match["homeTeam"]["name"],
                    "الفريق_الضيف": match["awayTeam"]["name"],
                    "أهداف_المضيف": match["score"]["fullTime"]["home"],
                    "أهداف_الضيف": match["score"]["fullTime"]["away"]
                })
        time.sleep(7)
    return pd.DataFrame(all_matches)

@st.cache_data(ttl=3600)
def load_upcoming():
    upcoming = []
    for code, name in competitions.items():
        response = requests.get(
            f"https://api.football-data.org/v4/competitions/{code}/matches?status=SCHEDULED",
            headers=headers
        )
        comp_data = response.json()
        if "matches" in comp_data:
            for match in comp_data["matches"]:
                upcoming.append({
                    "الدوري": name,
                    "كود_الدوري": code,
                    "التاريخ": match["utcDate"][:10],
                    "الفريق_المضيف": match["homeTeam"]["name"],
                    "الفريق_الضيف": match["awayTeam"]["name"]
                })
        time.sleep(7)
    return pd.DataFrame(upcoming)

@st.cache_data(ttl=3600)
def load_standings():
    standings_dict = {}
    for code in competitions.keys():
        try:
            response = requests.get(
                f"https://api.football-data.org/v4/competitions/{code}/standings",
                headers=headers
            )
            comp_data = response.json()
            table = {}
            if "standings" in comp_data:
                for entry in comp_data["standings"][0]["table"]:
                    table[entry["team"]["name"]] = {
                        "position": entry["position"],
                        "points": entry["points"]
                    }
            standings_dict[code] = table
        except:
            standings_dict[code] = {}
        time.sleep(7)
    return standings_dict

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

def predict_match_advanced(home_team, away_team, matches_df, standings_dict, code, max_goals=6, recent_n=8):
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

    best_prob = 0
    likely_home_goals, likely_away_goals = 0, 0
    home_win, draw, away_win = 0, 0, 0
    btts_yes = 0
    over_2_5 = 0

    for i in range(max_goals):
        for j in range(max_goals):
            p = poisson.pmf(i, home_expected_goals) * poisson.pmf(j, away_expected_goals)
            if p > best_prob:
                best_prob = p
                likely_home_goals, likely_away_goals = i, j
            if i > j:
                home_win += p
            elif i == j:
                draw += p
            else:
                away_win += p
            if i > 0 and j > 0:
                btts_yes += p
            if (i + j) > 2.5:
                over_2_5 += p

    first_half_home = round(likely_home_goals * 0.45)
    first_half_away = round(likely_away_goals * 0.45)
    second_half_home = likely_home_goals - first_half_home
    second_half_away = likely_away_goals - first_half_away

    table = standings_dict.get(code, {})
    home_position = table.get(home_team, {}).get("position", "غير متوفر")
    away_position = table.get(away_team, {}).get("position", "غير متوفر")

    return {
        "home_win": home_win, "draw": draw, "away_win": away_win,
        "final_score": (likely_home_goals, likely_away_goals),
        "first_half": (first_half_home, first_half_away),
        "second_half": (second_half_home, second_half_away),
        "btts_yes": btts_yes,
        "over_2_5": over_2_5,
        "home_position": home_position,
        "away_position": away_position
    }

st.title("⚽ توقع نتائج المباريات")
st.caption("يعتمد على الترتيب، الفورم، المواجهات المباشرة، ومعدل الأهداف")

with st.spinner("جاري تحميل البيانات..."):
    all_matches_df = load_matches()
    upcoming_df = load_upcoming()
    standings_dict = load_standings()

league = st.selectbox("اختر الدوري", list(competitions.values()))
code = [k for k, v in competitions.items() if v == league][0]

league_upcoming = upcoming_df[upcoming_df["الدوري"] == league].sort_values("التاريخ")

if league_upcoming.empty:
    st.warning("لا توجد مباريات قادمة مجدولة حالياً بهذا الدوري")
else:
    match_labels = [
        f"{row['التاريخ']} | {row['الفريق_المضيف']} ضد {row['الفريق_الضيف']}"
        for _, row in league_upcoming.iterrows()
    ]
    selected_label = st.selectbox("اختر المباراة", match_labels)
    selected_index = match_labels.index(selected_label)
    selected_match = league_upcoming.iloc[selected_index]
    home_team = selected_match["الفريق_المضيف"]
    away_team = selected_match["الفريق_الضيف"]

    if st.button("تحليل المباراة", type="primary"):
        result = predict_match_advanced(home_team, away_team, all_matches_df, standings_dict, code)

        st.subheader(f"{home_team} ضد {away_team}")

        c1, c2 = st.columns(2)
        c1.metric(f"ترتيب {home_team}", result["home_position"])
        c2.metric(f"ترتيب {away_team}", result["away_position"])

        home_form = get_team_form(home_team, all_matches_df)
        away_form = get_team_form(away_team, all_matches_df)
        c1, c2 = st.columns(2)
        c1.metric(f"فورم {home_team}", f"{home_form} نقطة")
        c2.metric(f"فورم {away_team}", f"{away_form} نقطة")

        h2h = head_to_head(home_team, away_team, all_matches_df)
        st.write("### المواجهات المباشرة")
        if h2h:
            total, hw, aw, d = h2h
            st.write(f"عدد المواجهات: {total} | فوز {home_team}: {hw} | فوز {away_team}: {aw} | تعادل: {d}")
        else:
            st.write("لا توجد مواجهات سابقة بالبيانات المتاحة")

        st.write("### احتمالية النتيجة")
        c1, c2, c3 = st.columns(3)
        c1.metric(f"فوز {home_team}", f"{round(result['home_win']*100,1)}%")
        c2.metric("تعادل", f"{round(result['draw']*100,1)}%")
        c3.metric(f"فوز {away_team}", f"{round(result['away_win']*100,1)}%")

        fh, fa = result["first_half"]
        sh, sa = result["second_half"]
        full_h, full_a = result["final_score"]

        st.write("### النتيجة الأقرب توقعاً")
        st.success(f"النتيجة النهائية المتوقعة: {full_h} - {full_a}")
        st.write(f"نهاية الشوط الأول المتوقعة: {fh} - {fa}")
        st.write(f"الشوط الثاني وحده المتوقع: {sh} - {sa}")

        st.write("### مؤشرات إضافية")
        c1, c2 = st.columns(2)
        c1.metric("احتمالية تسجيل الفريقين", f"{round(result['btts_yes']*100,1)}%")
        c2.metric("احتمالية أكثر من 2.5 هدف بالمجموع", f"{round(result['over_2_5']*100,1)}%")

        st.caption("تنبيه: هذي توقعات إحصائية تقريبية ولا تضمن النتيجة الفعلية للمباراة")
