import os
import database
import ml_engine

def run_daily_refresh():
    print("[CRON] Starting daily 6:00 AM feed generation...")
    database.init_db()

    history_records = database.get_watched_history()
    if not history_records:
        print("[CRON] No watched seeds found in database. Exiting.")
        return

    watched_ids = {v["video_id"] for v in history_records}
    history_features = [v["feature_text"] for v in history_records]

    # 1. Option 2: Creator Uploads (RSS)
    creator_channels = list({v.get("channel_id") for v in history_records if v.get("channel_id")})
    creator_pool = ml_engine.fetch_creator_uploads(creator_channels, watched_ids, max_per_channel=3)

    # 2. Option 1: Up Next / Related Graph
    related_pool = ml_engine.fetch_related_videos(history_records, watched_ids, max_per_seed=4)

    # 3. Clean and rank
    raw_candidates = creator_pool + related_pool
    clean_candidates = ml_engine.verify_and_purge_shorts(raw_candidates)
    
    if not clean_candidates:
        print("[CRON] No valid candidates passed the duration filter.")
        return

    ranked = ml_engine.rank_and_diversify_videos(history_features, clean_candidates, max_per_channel=1)
    
    # 4. Save top 5 into Supabase
    database.save_active_recommendations(ranked[:5])
    print(f"[CRON] Successfully refreshed and saved {len(ranked[:5])} fresh videos to Supabase!")

if __name__ == "__main__":
    run_daily_refresh()
