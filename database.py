import os
import streamlit as st
from sqlalchemy import create_engine, text

# Use Supabase URL from Streamlit Secrets, fallback to local SQLite for offline dev
DB_URL = st.secrets.get("DATABASE_URL", os.environ.get("DATABASE_URL", "sqlite:///youtube_vault.db"))

# Fix postgres prefix for SQLAlchemy if needed
if DB_URL.startswith("postgres://"):
    DB_URL = DB_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DB_URL, pool_pre_ping=True)

def init_db():
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS trained_videos (
                video_id TEXT PRIMARY KEY,
                title TEXT,
                embedding TEXT,
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