import streamlit as st
import database
import ml_engine

# Start collapsed so mobile opens directly into video feed
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
            with st.spinner("Embedding and training seeds..."):
                seeds = ml_engine.fetch_seed_videos_by_ids(extracted_ids)
                for s in seeds:
                    database.log_watched_video(s["video_id"], s["title"], s["channel"], s["feature_text"])
                st.success(f"Added {len(seeds)} video(s) to training profile!")
                st.session_state.pop("ranked_feed", None)  # Refresh feed on new seeds

    history = database.get_watched_history_features()
    st.caption(f"📊 Trained profile videos: **{len(history)}**")
    st.divider()

    if st.button("🔄 Refresh Recommendations", use_container_width=True):
        st.session_state.pop("ranked_feed", None)
        st.rerun()

# --- Automatic Feed Generation ---
history = database.get_watched_history_features()

if not history:
    st.info("👋 Welcome! Open the sidebar (arrow at top left) on your computer to add your first YouTube video seeds.")
else:
    # Auto-fetch if not already in session state
    if "ranked_feed" not in st.session_state:
        with st.spinner("Assembling your personalized feed..."):
            keywords = ml_engine.extract_interest_keywords(history, top_n=3)
            st.session_state["extracted_keywords"] = keywords
            candidates = ml_engine.fetch_discovery_videos(keywords, max_results=6)
            ranked = ml_engine.rank_and_diversify_videos(history, candidates, max_per_channel=1)
            st.session_state["ranked_feed"] = ranked

    # Display Top 5 Videos
    videos = st.session_state.get("ranked_feed", [])[:5]

    if not videos:
        st.warning("No candidate videos matched. Click 'Refresh Recommendations' in the sidebar.")
    else:
        st.subheader("Your 5 Ranked Recommendations")
        for idx, video in enumerate(videos):
            with st.container(border=True):
                col_video, col_info = st.columns([1.2, 1])
                with col_video:
                    st.video(video["url"])
                with col_info:
                    score = video.get("similarity_score", 0.0)
                    source = video.get("source", "Discovery")
                    st.markdown(
                        f"<span class='ml-badge'>🎯 Vector Match: {score * 100:.1f}%</span>"
                        f"<span class='source-tag'>{source}</span>",
                        unsafe_allow_html=True
                    )
                    st.markdown(f"#### {video['title']}")
                    st.write(f"**Channel:** {video['channel']}")
                    
                    if st.button("Mark Watched & Reinforce", key=f"feed_btn_{video['video_id']}_{idx}"):
                        database.log_watched_video(video["video_id"], video["title"], video["channel"], video["feature_text"])
                        st.success("Reinforced ML profile!")
                        st.rerun()
