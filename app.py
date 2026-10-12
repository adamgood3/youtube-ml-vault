import streamlit as st
import database
import ml_engine

# Collapsed sidebar by default for seamless mobile consumption
st.set_page_config(
    page_title="YouTube ML Vault",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    div[data-testid="stVerticalBlock"] > div[data-testid="stBlock"] {
        border-radius: 12px;
    }
    .ml-badge {
        background-color: #2e3440;
        color: #88c0d0;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 8px;
    }
    .source-tag {
        background-color: #3b4252;
        color: #eceff4;
        padding: 4px 10px;
        border-radius: 16px;
        font-size: 0.8rem;
        margin-left: 8px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📺 YouTube ML Vault")
database.init_db()

# --- Sidebar Controls (Desktop Model Training) ---
with st.sidebar:
    st.header("⚙️ Training & Seeds")
    
    with st.form(key="seed_form", clear_on_submit=True):
        raw_links = st.text_area(
            "Paste YouTube video URLs (one per line):",
            placeholder="https://www.youtube.com/watch?v=...\nhttps://www.youtube.com/watch?v=...",
            height=130
        )
        submit_seed = st.form_submit_button("Train Profile from Seeds", use_container_width=True)

    if submit_seed:
        lines = [line.strip() for line in raw_links.split("\n") if line.strip()]
        extracted_ids = [ml_engine.extract_video_id(line) for line in lines if ml_engine.extract_video_id(line)]
        
        if not extracted_ids:
            st.warning("Please paste at least one valid YouTube URL.")
        else:
            with st.spinner("Extracting metadata & updating vector space..."):
                seeds = ml_engine.fetch_seed_videos_by_ids(extracted_ids)
                for s in seeds:
                    database.log_watched_video(s["video_id"], s["title"], s["channel"], s["feature_text"])
                st.success(f"Trained {len(seeds)} video(s) into database!")

    history_records = database.get_watched_history()
    st.caption(f"📊 Trained profile videos: **{len(history_records)}**")
    st.divider()

    if st.button("🔄 Generate Hybrid Feed (Option 1 + 2)", use_container_width=True):
        with st.spinner("Harvesting creator uploads + related video network..."):
            watched_ids = {v["video_id"] for v in history_records}
            history_features = [v["feature_text"] for v in history_records]

            # 1. Option 2: Creator Uploads (RSS)
            creator_channels = list({v.get("channel_id") for v in history_records if v.get("channel_id")})
            creator_pool = ml_engine.fetch_creator_uploads(creator_channels, watched_ids, max_per_channel=3)

            # 2. Option 1: Up Next / Related Graph
            related_pool = ml_engine.fetch_related_videos(history_records, watched_ids, max_per_seed=4)

            # 3. Combine & purge Shorts
            raw_candidates = creator_pool + related_pool
            clean_candidates = ml_engine.verify_and_purge_shorts(raw_candidates)

            # 4. Nearest-Neighbor Semantic Ranking
            ranked = ml_engine.rank_and_diversify_videos(history_features, clean_candidates, max_per_channel=1)
            database.save_active_recommendations(ranked[:5])
            st.success("Fresh hybrid feed saved!")
            st.rerun()

# --- Feed Display ---
history_records = database.get_watched_history()

if not history_records:
    st.info("👋 Welcome! Open the sidebar on your laptop to paste your first seed videos.")
else:
    active_videos = database.get_active_recommendations()

    # Initial auto-generation if feed is empty
    if not active_videos:
        with st.spinner("Bootstrapping initial hybrid recommendations..."):
            watched_ids = {v["video_id"] for v in history_records}
            history_features = [v["feature_text"] for v in history_records]

            creator_channels = list({v.get("channel_id") for v in history_records if v.get("channel_id")})
            creator_pool = ml_engine.fetch_creator_uploads(creator_channels, watched_ids, max_per_channel=3)
            related_pool = ml_engine.fetch_related_videos(history_records, watched_ids, max_per_seed=4)

            raw_candidates = creator_pool + related_pool
            clean_candidates = ml_engine.verify_and_purge_shorts(raw_candidates)
            ranked = ml_engine.rank_and_diversify_videos(history_features, clean_candidates, max_per_channel=1)
            active_videos = ranked[:5]
            database.save_active_recommendations(active_videos)

    st.subheader("Your 5 Ranked Recommendations")
    
    if not active_videos:
        st.warning("No candidate videos available. Click 'Generate Hybrid Feed' in the sidebar.")
    else:
        for idx, video in enumerate(active_videos):
            with st.container(border=True):
                col_video, col_info = st.columns([1.2, 1])
                with col_video:
                    st.video(video["url"])
                with col_info:
                    score = video.get("similarity_score", 0.0)
                    source = video.get("source", "Hybrid Network")
                    st.markdown(
                        f"<span class='ml-badge'>🎯 Vector Match: {score * 100:.1f}%</span>"
                        f"<span class='source-tag'>{source}</span>",
                        unsafe_allow_html=True
                    )
                    st.markdown(f"#### {video['title']}")
                    st.write(f"**Channel:** {video['channel']}")
                    
                    if st.button("Mark Watched & Reinforce", key=f"feed_btn_{video['video_id']}_{idx}"):
                        feat_text = f"{video['title']} {video['channel']}"
                        database.log_watched_video(video["video_id"], video["title"], video["channel"], feat_text)
                        st.success("Trained model with this video!")
