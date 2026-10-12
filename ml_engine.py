import os
import re
import json
import urllib.request
import urllib.parse
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_video_id(url_or_id):
    """Extracts YouTube 11-character video ID from varied URL formats."""
    url_or_id = url_or_id.strip()
    match = re.search(r"(?:v=|\/|shorts\/)([0-9A-Za-z_-]{11})", url_or_id)
    return match.group(1) if match else url_or_id if len(url_or_id) == 11 else None

def fetch_seed_videos_by_ids(video_ids):
    """Fetches public title and channel metadata via YouTube oEmbed."""
    if not video_ids:
        return []

    seed_videos = []
    for v_id in video_ids:
        try:
            url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={v_id}&format=json"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                title = data.get("title", "")
                channel = data.get("author_name", "")
                seed_videos.append({
                    "video_id": v_id,
                    "title": title,
                    "channel": channel,
                    "feature_text": f"{title} {channel}"
                })
        except Exception as e:
            print(f"[WARN] Failed fetching metadata for {v_id}: {e}")
    return seed_videos

def extract_interest_keywords(history_texts, top_n=3):
    """Extracts high-signal keyphrases directly from your seeded videos."""
    if not history_texts:
        return []
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=100)
    matrix = vec.fit_transform(history_texts)
    scores = np.asarray(matrix.sum(axis=0)).ravel()
    top_indices = scores.argsort()[::-1][:top_n]
    return [vec.get_feature_names_out()[i] for i in top_indices]

def fetch_discovery_videos(keywords, max_results=8):
    """Pulls candidate videos matching extracted seed topics with metadata."""
    if not keywords:
        return []

    candidates = []
    seen_ids = set()

    for kw in keywords:
        try:
            search_query = urllib.parse.quote(kw)
            search_url = f"https://www.youtube.com/results?search_query={search_query}"
            req = urllib.request.Request(search_url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
            with urllib.request.urlopen(req, timeout=6) as response:
                html = response.read().decode()
                # Extract distinct video IDs
                found_ids = re.findall(r"/watch\?v=([0-9A-Za-z_-]{11})", html)
                
                count = 0
                for v_id in found_ids:
                    if v_id in seen_ids:
                        continue
                    seen_ids.add(v_id)

                    # Fetch oEmbed to get real channel name and title
                    try:
                        oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={v_id}&format=json"
                        oembed_req = urllib.request.Request(oembed_url, headers={'User-Agent': 'Mozilla/5.0'})
                        with urllib.request.urlopen(oembed_req, timeout=4) as oresp:
                            odata = json.loads(oresp.read().decode())
                            title = odata.get("title", f"Video {v_id}")
                            channel = odata.get("author_name", "YouTube Creator")
                            
                            candidates.append({
                                "video_id": v_id,
                                "title": title,
                                "channel": channel,
                                "feature_text": f"{title} {channel}",
                                "url": f"https://www.youtube.com/watch?v={v_id}",
                                "source": f"Topic: '{kw}'"
                            })
                            count += 1
                            if count >= max_results:
                                break
                    except Exception:
                        continue
        except Exception as e:
            print(f"[WARN] Search query failed for '{kw}': {e}")

    return candidates

def fetch_subscribed_channel_videos(max_channels=10, max_videos_per_channel=3):
    return []

def rank_and_diversify_videos(seed_features, candidate_videos, max_per_channel=2):
    """Computes cosine similarity against the seed centroid and limits channel clustering."""
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
