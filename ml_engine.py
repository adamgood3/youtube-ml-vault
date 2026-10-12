import os
import re
import json
import urllib.request
import urllib.parse
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import yt_dlp

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Strictly filter out Shorts and short clips (minimum 3 minutes = 180s)
MIN_DURATION_SECONDS = 180

def extract_video_id(url_or_id):
    url_or_id = url_or_id.strip()
    match = re.search(r"(?:v=|\/|shorts\/)([0-9A-Za-z_-]{11})", url_or_id)
    return match.group(1) if match else url_or_id if len(url_or_id) == 11 else None

def get_video_info(video_id):
    """Uses yt-dlp to inspect exact duration and metadata without API keys."""
    ydl_opts = {
        'skip_download': True,
        'extract_flat': False,
        'quiet': True,
        'no_warnings': True
    }
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return {
                "title": info.get("title", ""),
                "channel": info.get("uploader", "YouTube Creator"),
                "duration": info.get("duration", 0),
                "description": (info.get("description") or "")[:400]
            }
    except Exception as e:
        print(f"[WARN] Failed extracting info for {video_id}: {e}")
        return None

def fetch_seed_videos_by_ids(video_ids):
    """Fetches seed video details, ensuring proper title and channel info."""
    if not video_ids:
        return []

    seed_videos = []
    for v_id in video_ids:
        info = get_video_info(v_id)
        if info:
            title = info["title"]
            channel = info["channel"]
            seed_videos.append({
                "video_id": v_id,
                "title": title,
                "channel": channel,
                "feature_text": f"{title} {channel} {info['description']}"
            })
    return seed_videos

def fetch_discovery_videos(watched_videos, max_results=12):
    """
    Searches YouTube for long-form candidates using seed channel and title themes,
    strictly discarding any video under 3 minutes or matching Shorts format.
    """
    if not watched_videos:
        return []

    # Use channels and core titles as high-signal search anchors
    search_queries = []
    for v in watched_videos[-4:]:  # focus on latest additions
        search_queries.append(f"{v['channel']} full video")
        clean_title = re.sub(r"[^a-zA-Z0-9\s]", "", v['title'])
        search_queries.append(" ".join(clean_title.split()[:5]))

    watched_ids = {v["video_id"] for v in watched_videos}
    candidates = []
    seen_ids = set()

    ydl_search_opts = {
        'skip_download': True,
        'extract_flat': 'in_playlist',
        'quiet': True,
        'no_warnings': True
    }

    with yt_dlp.YoutubeDL(ydl_search_opts) as ydl:
        for q in list(set(search_queries))[:4]:
            try:
                # Search YouTube directly for top results
                results = ydl.extract_info(f"ytsearch8:{q}", download=False)
                entries = results.get("entries", []) if results else []

                for entry in entries:
                    if not entry:
                        continue
                    v_id = entry.get("id")
                    if not v_id or v_id in watched_ids or v_id in seen_ids:
                        continue
                    
                    seen_ids.add(v_id)
                    
                    # Verify metadata and strict duration filter
                    detail = get_video_info(v_id)
                    if not detail:
                        continue
                    
                    # STRICT SHORTS CUTOFF: Discard anything under 3 minutes
                    if detail["duration"] < MIN_DURATION_SECONDS:
                        continue

                    candidates.append({
                        "video_id": v_id,
                        "title": detail["title"],
                        "channel": detail["channel"],
                        "feature_text": f"{detail['title']} {detail['channel']} {detail['description']}",
                        "url": f"https://www.youtube.com/watch?v={v_id}",
                        "source": f"Related to: {q}"
                    })

                    if len(candidates) >= max_results:
                        break
            except Exception as e:
                print(f"[WARN] Discovery search error for '{q}': {e}")

    return candidates

def rank_and_diversify_videos(seed_features, candidate_videos, max_per_channel=1):
    """Computes cosine similarity against the user's semantic centroid."""
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
