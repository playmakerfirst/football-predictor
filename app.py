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
        time.sleep(7)
    return standings_dict
