"""
daily_refresh.py: Standalone headless script to fetch, rank, and store today's 5 videos.
"""
from datetime import date
import database
import ml_engine

def run_daily_job(limit=5):
    print(f"[{date.today()}] Starting daily recommendation job...")
    database.init_db()
    
    history = database.get_watched_history_features()
    if not history:
        print("[ABORT] No profile training seeds found in database. Train seeds first!")
        return []
    
    seen_ids = database.get_presented_video_ids()
    print(f"[INFO] Currently blacklisting {len(seen_ids)} previously seen videos.")
    
    # 1. Fetch Subscription & Discovery candidates
    sub_videos = ml_engine.fetch_subscribed_channel_videos(max_channels=12, max_videos_per_channel=3)
    keywords = ml_engine.extract_interest_keywords(history, top_n=2)
    disc_videos = ml_engine.fetch_discovery_videos(keywords, max_results=8) if keywords else []
    
    # 2. Purge already-seen videos from candidate pool
    combined_pool = [v for v in (sub_videos + disc_videos) if v["video_id"] not in seen_ids]
    print(f"[INFO] Candidate pool after duplicate exclusion: {len(combined_pool)} videos.")
    
    # 3. Rank & diversify
    ranked = ml_engine.rank_and_diversify_videos(history, combined_pool, max_per_channel=1)
    top_daily = ranked[:limit]
    
    if top_daily:
        database.save_daily_feed(top_daily)
        print(f"[SUCCESS] Stored {len(top_daily)} fresh videos for today ({date.today()}).")
    else:
        print("[WARNING] Could not find enough fresh candidates.")
        
    return top_daily

if __name__ == "__main__":
    run_daily_job()