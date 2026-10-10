import os
import json
import streamlit as st
from sqlalchemy import create_engine, text

# Safely load DB_URL across Streamlit Cloud, local secrets, env vars, or local SQLite fallback
DB_URL = None

try:
    if "DATABASE_URL" in st.secrets:
        DB_URL = st.secrets["DATABASE_URL"]
except Exception:
    pass

if not DB_URL:
    DB_URL = os.environ.get("DATABASE_URL", "sqlite:///youtube_vault.db")

# SQLAlchemy expects 'postgresql://' instead of 'postgres://'
if DB_URL.startswith("postgres://"):
    DB_URL = DB_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DB_URL, pool_pre_ping=True)

def init_db():
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS trained_videos (
                video_id TEXT PRIMARY KEY,
                title TEXT,
                features TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS daily_recommendations (
                id SERIAL PRIMARY KEY,
                date TEXT,
                video_id TEXT,
                title TEXT,
                channel TEXT,
                thumbnail TEXT,
                url TEXT
            );
        """))

init_db()

def get_watched_history_features():
    """Returns a list of feature dictionaries or arrays for trained videos."""
    with engine.connect() as conn:
        result = conn.execute(text("SELECT video_id, title, features FROM trained_videos"))
        rows = result.fetchall()
        
    history = []
    for row in rows:
        features_data = None
        if row[2]:
            try:
                features_data = json.loads(row[2])
            except Exception:
                features_data = row[2]
        history.append({
            "video_id": row[0],
            "title": row[1],
            "features": features_data
        })
    return history

def save_trained_video(video_id, title, features):
    """Saves or updates a trained seed video."""
    features_json = json.dumps(features) if isinstance(features, (list, dict)) else str(features)
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO trained_videos (video_id, title, features)
                VALUES (:video_id, :title, :features)
                ON CONFLICT (video_id) DO UPDATE 
                SET title = EXCLUDED.title, features = EXCLUDED.features;
            """),
            {"video_id": video_id, "title": title, "features": features_json}
        )

def get_daily_recommendations(date_str):
    """Gets recommendations for a specific date."""
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT video_id, title, channel, thumbnail, url FROM daily_recommendations WHERE date = :date"),
            {"date": date_str}
        )
        rows = result.fetchall()
    return [
        {"video_id": r[0], "title": r[1], "channel": r[2], "thumbnail": r[3], "url": r[4]}
        for r in rows
    ]

def save_daily_recommendations(date_str, videos):
    """Saves daily recommendations."""
    with engine.begin() as conn:
        for v in videos:
            conn.execute(
                text("""
                    INSERT INTO daily_recommendations (date, video_id, title, channel, thumbnail, url)
                    VALUES (:date, :video_id, :title, :channel, :thumbnail, :url);
                """),
                {
                    "date": date_str,
                    "video_id": v.get("video_id"),
                    "title": v.get("title"),
                    "channel": v.get("channel"),
                    "thumbnail": v.get("thumbnail"),
                    "url": v.get("url")
                }
            )