import sqlite3
from datetime import date

DB_NAME = "youtube_vault.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # 1. Training history (from your manual seeds)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS watched_history (
            video_id TEXT PRIMARY KEY,
            title TEXT,
            channel TEXT,
            feature_text TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # 2. History of all videos ever recommended to prevent duplicates
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS presented_videos (
            video_id TEXT PRIMARY KEY,
            presented_date DATE
        )
    """)
    
    # 3. Active Daily Feed (the locked 5 videos for today)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_feed (
            video_id TEXT PRIMARY KEY,
            title TEXT,
            channel TEXT,
            feature_text TEXT,
            url TEXT,
            similarity_score REAL,
            source TEXT,
            feed_date DATE
        )
    """)
    
    conn.commit()
    conn.close()

def log_watched_video(video_id, title, channel, feature_text):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR IGNORE INTO watched_history (video_id, title, channel, feature_text)
        VALUES (?, ?, ?, ?)
    """, (video_id, title, channel, feature_text))
    conn.commit()
    conn.close()

def get_watched_history_features():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT feature_text FROM watched_history")
    rows = cursor.fetchall()
    conn.close()
    return [r[0] for r in rows]

def get_presented_video_ids():
    """Returns set of all video IDs presented in the last 30 days to enforce zero duplicates."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT video_id FROM presented_videos")
    rows = cursor.fetchall()
    conn.close()
    return {r[0] for r in rows}

def save_daily_feed(videos):
    """Replaces today's feed with a fresh set of 5 videos and marks them as presented."""
    today = str(date.today())
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Clear previous active feed
    cursor.execute("DELETE FROM daily_feed")
    
    for v in videos:
        cursor.execute("""
            INSERT OR REPLACE INTO daily_feed 
            (video_id, title, channel, feature_text, url, similarity_score, source, feed_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            v["video_id"], v["title"], v["channel"], v["feature_text"],
            v["url"], v["similarity_score"], v.get("source", "Pool"), today
        ))
        
        # Log to permanent blacklist to avoid seeing it again
        cursor.execute("""
            INSERT OR IGNORE INTO presented_videos (video_id, presented_date)
            VALUES (?, ?)
        """, (v["video_id"], today))
        
    conn.commit()
    conn.close()

def get_today_feed():
    """Retrieves the stored feed for today."""
    today = str(date.today())
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT video_id, title, channel, feature_text, url, similarity_score, source, feed_date
        FROM daily_feed WHERE feed_date = ?
    """, (today,))
    rows = cursor.fetchall()
    conn.close()
    
    feed = []
    for r in rows:
        feed.append({
            "video_id": r[0],
            "title": r[1],
            "channel": r[2],
            "feature_text": r[3],
            "url": r[4],
            "similarity_score": r[5],
            "source": r[6],
            "feed_date": r[7]
        })
    return feed