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
            ALTER TABLE trained_videos ADD COLUMN IF NOT EXISTS title TEXT;
            ALTER TABLE trained_videos ADD COLUMN IF NOT EXISTS channel TEXT;
            ALTER TABLE trained_videos ADD COLUMN IF NOT EXISTS feature_text TEXT;

            CREATE TABLE IF NOT EXISTS active_recommendations (
                id SERIAL PRIMARY KEY,
                video_id TEXT UNIQUE,
                title TEXT,
                channel TEXT,
                url TEXT,
                similarity_score FLOAT,
                source TEXT
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

def get_watched_history():
    """Returns all watched records with IDs and features."""
    with engine.begin() as conn:
        result = conn.execute(text("SELECT video_id, title, channel, feature_text FROM trained_videos;")).fetchall()
        return [
            {"video_id": row[0], "title": row[1], "channel": row[2], "feature_text": row[3]}
            for row in result
        ]

def get_watched_history_features():
    with engine.begin() as conn:
        result = conn.execute(text("SELECT feature_text FROM trained_videos;")).fetchall()
        return [row[0] for row in result if row[0]]

def save_active_recommendations(videos):
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE active_recommendations;"))
        for v in videos:
            conn.execute(text("""
                INSERT INTO active_recommendations (video_id, title, channel, url, similarity_score, source)
                VALUES (:video_id, :title, :channel, :url, :similarity_score, :source)
                ON CONFLICT (video_id) DO NOTHING;
            """), {
                "video_id": v["video_id"],
                "title": v["title"],
                "channel": v["channel"],
                "url": v["url"],
                "similarity_score": v.get("similarity_score", 0.0),
                "source": v.get("source", "Discovery")
            })

def get_active_recommendations():
    with engine.begin() as conn:
        result = conn.execute(text("""
            SELECT video_id, title, channel, url, similarity_score, source 
            FROM active_recommendations 
            ORDER BY similarity_score DESC;
        """)).fetchall()
        return [
            {
                "video_id": row[0],
                "title": row[1],
                "channel": row[2],
                "url": row[3],
                "similarity_score": row[4],
                "source": row[5]
            }
            for row in result
        ]
