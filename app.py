import streamlit as st
import database
import ml_engine

st.set_page_config(page_title="YouTube ML Vault", page_icon="📺", layout="wide")

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
st.caption("Personalized Feed Engine (Semantic Centroid + Discovery Ranking)")

database.init_db()

# --- Sidebar Controls ---
st.sidebar.header("Profile Seeds & Settings")

# Form with clear_on_submit clears the text box automatically
with st.sidebar.form(key="seed_form", clear_on_submit=True):
    raw_links = st.text_area(
        "Paste YouTube video URLs (one per line):",
        placeholder="https://www.youtube.com/watch?v=...\nhttps://www.youtube.com/watch?v=...",
        height=140
    )
    submit_seed = st.form_submit_button("1. Train Profile from Seeds", use_container_width=True)

if submit_seed:
    lines = [line.strip() for line in raw_links.split("\n") if line.strip()]
    extracted_ids = [ml_engine.extract_video_id(line) for line in lines if ml_engine.extract_video_id(line)]
    
    if not extracted_ids:
        st.sidebar.warning("Please paste at least one valid YouTube video URL.")
    else:
        with st.spinner("Extracting video metadata & encoding vector profile..."):
            seeds = ml_engine.fetch_seed_videos_by_ids(extracted_ids)
            for s in seeds:
                database.log_watched_video(s["video_id"], s["title"], s["channel"], s["feature_text"])
            st.sidebar.success(f"Added {len(seeds)} videos to training set!")

# Display current count of training videos stored in database
current_history = database.get_watched_history_features()
st.sidebar.caption(f"📊 Total videos currently trained in ML profile: **{len(current_history)}**")

st.sidebar.divider()

video_limit = st.sidebar.slider("Recommended Videos", min_value=3, max_value=10, value=5)

if st.sidebar.button("2. Fetch & Rank Feed", type="primary", use_container_width=True):
    with st.spinner("Harvesting candidates and computing semantic distance..."):
        history = database.get_watched_history_features()
        if not history:
            st.warning("Please train your profile seeds in Step 1 first!")
        else:
            # 1. Fetch from subscriptions
            sub_videos = ml_engine.fetch_subscribed_channel_videos(max_channels=10, max_videos_per_channel=3)
            
            # 2. Extract profile topics directly from your seeds
            keywords = ml_engine.extract_interest_keywords(history, top_n=2)
            st.session_state["extracted_keywords"] = keywords
            
            # 3. Pull discovery videos
            disc_videos = ml_engine.fetch_discovery_videos(keywords, max_results=6)
            
            # 4. Rank combined candidate pool with max 1 video per channel
            combined = sub_videos + disc_videos
            ranked = ml_engine.rank_and_diversify_videos(history, combined, max_per_channel=1)
            st.session_state["ranked_feed"] = ranked

# --- Main Dashboard ---
if "extracted_keywords" in st.session_state and st.session_state["extracted_keywords"]:
    st.info(f"**Discovered Seed Topics:** {', '.join(st.session_state['extracted_keywords'])}")

if "ranked_feed" in st.session_state:
    videos = st.session_state["ranked_feed"][:video_limit]
    st.subheader(f"Top {len(videos)} Ranked Recommendations")

    for idx, video in enumerate(videos):
        with st.container(border=True):
            col_video, col_info = st.columns([1.2, 1])
            with col_video:
                st.video(video["url"])
            with col_info:
                score = video.get("similarity_score", 0.0)
                source = video.get("source", "Pool")
                st.markdown(f"<span class='ml-badge'>🎯 Vector Match: {score * 100:.1f}%</span><span class='source-tag'>{source}</span>", unsafe_allow_html=True)
                st.markdown(f"#### {video['title']}")
                st.write(f"**Channel:** {video['channel']}")
                
                if st.button("Mark Watched & Reinforce", key=f"btn_{video['video_id']}_{idx}"):
                    database.log_watched_video(video["video_id"], video["title"], video["channel"], video["feature_text"])
                    st.success("Trained model with this video!")