import os
import pickle
import re
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import isodate
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
SCOPES = ["https://www.googleapis.com/auth/youtube.readonly"]

# Minimum duration cutoff in seconds (180s = 3 minutes strictly blocks all Shorts and teaser clips)
MIN_DURATION_SECONDS = 180

def get_authenticated_service():
    creds = None
    if os.path.exists("token.pickle"):
        with open("token.pickle", "rb") as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists("client_secret.json"):
                raise FileNotFoundError("Missing client_secret.json!")
            flow = InstalledAppFlow.from_client_secrets_file("client_secret.json", SCOPES)
            creds = flow.run_local_server(port=0)

        with open("token.pickle", "wb") as token:
            pickle.dump(creds, token)

    return build("youtube", "v3", credentials=creds)

def extract_video_id(url_or_id):
    """Extracts YouTube 11-character video ID from varied URL formats."""
    url_or_id = url_or_id.strip()
    match = re.search(r"(?:v=|\/|shorts\/)([0-9A-Za-z_-]{11})", url_or_id)
    return match.group(1) if match else url_or_id if len(url_or_id) == 11 else None

def filter_out_shorts(video_records):
    """
    Batches video IDs into youtube.videos().list(part='contentDetails')
    and strictly discards any video with duration under MIN_DURATION_SECONDS.
    """
    if not video_records:
        return []

    youtube = get_authenticated_service()
    video_ids = [v["video_id"] for v in video_records]
    long_form_ids = set()

    # Query YouTube in batches of 50 (max allowed per API call)
    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i:i+50]
        try:
            req = youtube.videos().list(part="contentDetails", id=",".join(chunk))
            res = req.execute()

            for item in res.get("items", []):
                duration_iso = item["contentDetails"]["duration"]
                duration_sec = isodate.parse_duration(duration_iso).total_seconds()
                if duration_sec >= MIN_DURATION_SECONDS:
                    long_form_ids.add(item["id"])
        except Exception as err:
            print(f"[ERROR] Duration filtering failed: {err}")

    return [v for v in video_records if v["video_id"] in long_form_ids]

def fetch_seed_videos_by_ids(video_ids):
    """Fetches full metadata and embeddings for user-provided sample/seed videos."""
    if not video_ids:
        return []
    try:
        youtube = get_authenticated_service()
        req = youtube.videos().list(part="snippet,contentDetails", id=",".join(video_ids))
        res = req.execute()

        seed_videos = []
        for item in res.get("items", []):
            duration = isodate.parse_duration(item["contentDetails"]["duration"]).total_seconds()
            if duration >= MIN_DURATION_SECONDS:
                snip = item["snippet"]
                seed_videos.append({
                    "video_id": item["id"],
                    "title": snip["title"],
                    "channel": snip["channelTitle"],
                    "feature_text": f"{snip['title']} {snip['channelTitle']} {snip.get('description', '')}"
                })
        return seed_videos
    except Exception as e:
        print(f"[ERROR] Fetch seed videos failed: {e}")
        return []

def extract_interest_keywords(history_texts, top_n=2):
    """Extracts high-signal keyphrases directly from your seeded long-form videos."""
    if not history_texts:
        return []
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=100)
    matrix = vec.fit_transform(history_texts)
    scores = np.asarray(matrix.sum(axis=0)).ravel()
    top_indices = scores.argsort()[::-1][:top_n]
    return [vec.get_feature_names_out()[i] for i in top_indices]

def fetch_discovery_videos(keywords, max_results=8):
    """Pulls non-subscription candidates and verifies duration to purge Shorts."""
    if not keywords:
        return []
    try:
        youtube = get_authenticated_service()
        raw_discovery = []
        for kw in keywords:
            res = youtube.search().list(
                part="snippet",
                q=kw,
                type="video",
                videoDuration="medium",
                order="relevance",
                maxResults=max_results
            ).execute()

            for item in res.get("items", []):
                snippet = item["snippet"]
                v_id = item["id"]["videoId"]
                raw_discovery.append({
                    "video_id": v_id,
                    "title": snippet["title"],
                    "channel": snippet["channelTitle"],
                    "feature_text": f"{snippet['title']} {snippet['channelTitle']} {snippet.get('description', '')}",
                    "url": f"https://www.youtube.com/watch?v={v_id}",
                    "source": f"Discovery: '{kw}'"
                })

        # Apply strict duration filter
        return filter_out_shorts(raw_discovery)
    except Exception as e:
        print(f"[ERROR] Discovery fetch failed: {e}")
        return []

def fetch_subscribed_channel_videos(max_channels=10, max_videos_per_channel=3):
    """Pulls uploads across subscribed channels and verifies duration to purge Shorts."""
    try:
        youtube = get_authenticated_service()
        subs = youtube.subscriptions().list(part="snippet", mine=True, maxResults=max_channels).execute().get("items", [])
        if not subs:
            return []

        channel_ids = [s["snippet"]["resourceId"]["channelId"] for s in subs]
        channels_res = youtube.channels().list(part="contentDetails,snippet", id=",".join(channel_ids)).execute()

        raw_candidates = []
        for ch in channels_res.get("items", []):
            uploads_id = ch["contentDetails"]["relatedPlaylists"]["uploads"]
            ch_title = ch["snippet"]["title"]
            playlist_res = youtube.playlistItems().list(part="snippet", playlistId=uploads_id, maxResults=max_videos_per_channel).execute()

            for item in playlist_res.get("items", []):
                snippet = item["snippet"]
                v_id = snippet["resourceId"]["videoId"]
                raw_candidates.append({
                    "video_id": v_id,
                    "title": snippet["title"],
                    "channel": ch_title,
                    "feature_text": f"{snippet['title']} {ch_title} {snippet.get('description', '')}",
                    "url": f"https://www.youtube.com/watch?v={v_id}",
                    "source": "Subscribed Channel"
                })

        # Apply strict duration filter
        return filter_out_shorts(raw_candidates)
    except Exception as e:
        print(f"[ERROR] Subscription fetch failed: {e}")
        return []

def rank_and_diversify_videos(seed_features, candidate_videos, max_per_channel=1):
    """Computes cosine similarity against the seed centroid and caps channel repetition."""
    if not candidate_videos or not seed_features:
        return candidate_videos

    seed_embeddings = embedding_model.encode(seed_features)
    candidate_embeddings = embedding_model.encode([v["feature_text"] for v in candidate_videos])

    seed_centroid = np.mean(seed_embeddings, axis=0).reshape(1, -1)
    scores = cosine_similarity(seed_centroid, candidate_embeddings)[0]

    for idx, video in enumerate(candidate_videos):
        video["similarity_score"] = float(scores[idx])

    ranked = sorted(candidate_videos, key=lambda x: x["similarity_score"], reverse=True)

    diversified = []
    channel_counts = {}
    for video in ranked:
        ch = video["channel"]
        if channel_counts.get(ch, 0) < max_per_channel:
            diversified.append(video)
            channel_counts[ch] = channel_counts.get(ch, 0) + 1

    return diversified