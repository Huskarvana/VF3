
import streamlit as st
import requests
import pandas as pd
from datetime import datetime
from transformers import pipeline

# --- CONFIGURATION ---
st.set_page_config(page_title="Veille DS Automobiles", layout="wide")
st.title("🚗 Agent de Veille – DS Automobiles (APIs multiples)")

API_KEY_NEWSDATA = st.secrets["API_KEY_NEWSDATA"]
MEDIASTACK_API_KEY = st.secrets["MEDIASTACK_API_KEY"]

@st.cache_resource
def get_sentiment_pipeline():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

sentiment_analyzer = get_sentiment_pipeline()

# Filtres
langue = st.selectbox("Filtrer par langue", ["fr", "en", "es", "tous"])
mot_cle = st.text_input("🔍 Recherche par mot-clé (optionnel)", "")
nb_articles = st.slider("Nombre d'articles par source", 5, 30, 25)

# Collecte
def fetch_newsdata(query, lang, max_results=10):
    url = "https://newsdata.io/api/1/news"
    params = {
        "apikey": API_KEY_NEWSDATA,
        "q": query,
        "language": "" if lang == "tous" else lang,
        "page": 1
    }
    try:
        r = requests.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        return [{
            "date": item.get("pubDate", ""),
            "titre": item.get("title", ""),
            "contenu": item.get("description", ""),
            "source": item.get("source_id", ""),
            "lien": item.get("link", "")
        } for item in data.get("results", [])[:max_results]]
    except Exception as e:
        st.error(f"Erreur NewsData: {e}")
        return []

# Analyse
def analyser_article(row):
    try:
        sentiment = sentiment_analyzer(row['contenu'][:512])[0]['label']
    except:
        sentiment = "neutral"
    résumé = row['contenu'][:200] + "..."
    return pd.Series({'résumé': résumé, 'ton': sentiment.upper()})

# Lancer la veille
if st.button("🔍 Lancer la veille"):
    query = "DS Automobiles"
    if mot_cle:
        query += " " + mot_cle

    articles = fetch_newsdata(query, langue, nb_articles)
    df = pd.DataFrame(articles)

    if not df.empty:
        df[["résumé", "ton"]] = df.apply(analyser_article, axis=1)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values(by="date", ascending=False)
        st.dataframe(df[["date", "titre", "ton", "résumé", "source", "lien"]])
    else:
        st.warning("Aucun article trouvé.")
