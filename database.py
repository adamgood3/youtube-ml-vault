import os
import streamlit as st
from sqlalchemy import create_engine, text

DB_URL = st.secrets.get("DATABASE_URL") or os.environ.get("DATABASE_URL", "")

if DB_URL.startswith("postgres://"):
    DB_URL = DB_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DB_URL, pool_pre_ping=True)

def init_db():
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS trained_videos (
                video_id TEXT PRIMARY KEY,
                title TEXT,
                channel TEXT,
                feature_text TEXT
            );
        """))

def log_watched_video(video_id, title, channel, feature_text):
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO trained_videos (video_id, title, channel, feature_text)
            VALUES (:video_id, :title, :channel, :feature_text)
            ON CONFLICT (video_id) DO UPDATE SET
                title = EXCLUDED.title,
                channel = EXCLUDED.channel,
                feature_text = EXCLUDED.feature_text;
        """), {
            "video_id": video_id,
            "title": title,
            "channel": channel,
            "feature_text": feature_text
        })

def get_watched_history_features():
    with engine.begin() as conn:
        result = conn.execute(text("SELECT feature_text FROM trained_videos;")).fetchall()
        return [row[0] for row in result if row[0]]
