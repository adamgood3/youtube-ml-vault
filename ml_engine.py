import os
import re
import json
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import yt_dlp

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
MIN_DURATION_SECONDS = 180  # Strict 3-minute cutoff to eliminate Shorts

def extract_video_id(url_or_id):
    """Extracts YouTube 11-character video ID from varied URL formats."""
    url_or_id = url_or_id.strip()
    match = re.search(r"(?:v=|\/|shorts\/)([0-9A-Za-z_-]{11})", url_or_id)
    return match.group(1) if match else url_or_id if len(url_or_id) == 11 else None

def get_video_metadata(video_id):
    """Extracts metadata and channel ID using keyless oEmbed and page inspection."""
    title, channel, channel_id = f"Video {video_id}", "YouTube Creator", None
    try:
        oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
        req = urllib.request.Request(oembed_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            title = data.get("title", title)
            channel = data.get("author_name", channel)
            author_url = data.get("author_url", "")
            if "/channel/" in author_url:
                channel_id = author_url.split("/channel/")[-1].strip()
    except Exception as e:
        print(f"[WARN] oEmbed failed for {video_id}: {e}")

    # Fallback to scrape channel_id if missing
    if not channel_id:
        try:
            v_url = f"https://www.youtube.com/watch?v={video_id}"
            req = urllib.request.Request(v_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as resp:
                html = resp.read().decode('utf-8', errors='ignore')
                m = re.search(r'"channelId":"(UC[a-zA-Z0-9_-]{22})"', html)
                if m:
                    channel_id = m.group(1)
        except Exception:
            pass

    return {
        "video_id": video_id,
        "title": title,
        "channel": channel,
        "channel_id": channel_id,
        "feature_text": f"{title} {channel}"
    }

def fetch_seed_videos_by_ids(video_ids):
    """Ingests user seed videos with title, channel name, and channel ID."""
    records = []
    for v_id in video_ids:
        data = get_video_metadata(v_id)
        if data:
            records.append(data)
    return records

def fetch_creator_uploads(channel_ids, watched_ids, max_per_channel=3):
    """Option 2: Pulls fresh long-form uploads directly from seeded creators via public RSS feeds."""
    candidates = []
    seen = set()

    for ch_id in channel_ids:
        if not ch_id:
            continue
        try:
            feed_url = f"https://www.youtube.com/feeds/videos.xml?channel_id={ch_id}"
            req = urllib.request.Request(feed_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as resp:
                tree = ET.fromstring(resp.read().decode())
                ns = {'atom': 'http://www.w3.org/2005/Atom', 'yt': 'http://www.youtube.com/xml/schemas/2015'}
                
                count = 0
                for entry in tree.findall('atom:entry', ns):
                    v_id_elem = entry.find('yt:videoId', ns)
                    title_elem = entry.find('atom:title', ns)
                    author_elem = entry.find('atom:author/atom:name', ns)

                    if v_id_elem is None or title_elem is None:
                        continue

                    v_id = v_id_elem.text
                    title = title_elem.text
                    channel = author_elem.text if author_elem is not None else "Creator"

                    if v_id in watched_ids or v_id in seen:
                        continue
                    
                    seen.add(v_id)
                    candidates.append({
                        "video_id": v_id,
                        "title": title,
                        "channel": channel,
                        "feature_text": f"{title} {channel}",
                        "url": f"https://www.youtube.com/watch?v={v_id}",
                        "source": f"Creator: {channel}"
                    })
                    count += 1
                    if count >= max_per_channel:
                        break
        except Exception as e:
            print(f"[WARN] Failed fetching RSS for channel {ch_id}: {e}")

    return candidates

def fetch_related_videos(seed_records, watched_ids, max_per_seed=3):
    """Option 1: Scrapes genuine Up Next / related videos from your seed watch pages."""
    candidates = []
    seen = set()

    for seed in seed_records[-4:]:  # focus on the 4 most recent seeds
        v_id = seed["video_id"]
        try:
            url = f"https://www.youtube.com/watch?v={v_id}"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
            with urllib.request.urlopen(req, timeout=6) as resp:
                html = resp.read().decode('utf-8', errors='ignore')
                
                # Match recommended video IDs linked on the page
                found_ids = re.findall(r'"videoId":"([0-9A-Za-z_-]{11})"', html)
                
                count = 0
                for rel_id in found_ids:
                    if rel_id == v_id or rel_id in watched_ids or rel_id in seen:
                        continue
                    
                    seen.add(rel_id)
                    meta = get_video_metadata(rel_id)
                    candidates.append({
                        "video_id": rel_id,
                        "title": meta["title"],
                        "channel": meta["channel"],
                        "feature_text": meta["feature_text"],
                        "url": f"https://www.youtube.com/watch?v={rel_id}",
                        "source": f"Related to: {seed['title'][:25]}..."
                    })
                    count += 1
                    if count >= max_per_seed:
                        break
        except Exception as e:
            print(f"[WARN] Related video fetch failed for {v_id}: {e}")

    return candidates

def verify_and_purge_shorts(candidates):
    """Verifies duration using yt-dlp to strictly eliminate all Shorts."""
    clean_candidates = []
    ydl_opts = {'skip_download': True, 'quiet': True, 'no_warnings': True}

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for c in candidates:
            try:
                info = ydl.extract_info(c["url"], download=False)
                duration = info.get("duration", 0)
                if duration >= MIN_DURATION_SECONDS:
                    clean_candidates.append(c)
            except Exception:
                # If extraction fails, keep candidate if title does not contain 'shorts' or '#'
                if "#shorts" not in c["title"].lower() and "shorts" not in c["url"].lower():
                    clean_candidates.append(c)

    return clean_candidates

def rank_and_diversify_videos(seed_features, candidate_videos, max_per_channel=1):
    """
    Nearest-Neighbor Vector Scoring: Scores candidates against individual seeds 
    rather than a mashed average centroid, preventing topic dilution.
    """
    if not candidate_videos or not seed_features:
        return candidate_videos

    seed_embeddings = embedding_model.encode(seed_features)
    candidate_embeddings = embedding_model.encode([v["feature_text"] for v in candidate_videos])

    # Calculate similarity matrix (candidates x seeds)
    similarity_matrix = cosine_similarity(candidate_embeddings, seed_embeddings)
    
    # Assign each candidate its best match across any of your specific seeds
    for idx, video in enumerate(candidate_videos):
        video["similarity_score"] = float(np.max(similarity_matrix[idx]))

    ranked = sorted(candidate_videos, key=lambda x: x["similarity_score"], reverse=True)

    diversified = []
    channel_counts = {}
    for video in ranked:
        ch = video["channel"]
        if channel_counts.get(ch, 0) < max_per_channel:
            diversified.append(video)
            channel_counts[ch] = channel_counts.get(ch, 0) + 1

    return diversified
