"""
Fetches comments from a YouTube video using the YouTube Data API v3,
and extracts the video ID from common URL formats.
"""
import os
import re
from typing import List, Dict

import requests

YOUTUBE_API_URL = "https://www.googleapis.com/youtube/v3/commentThreads"


def extract_video_id(url_or_id: str) -> str:
    """
    Accepts a full YouTube URL (watch, youtu.be, shorts) or a bare video ID
    and returns just the 11-character video ID.
    """
    url_or_id = url_or_id.strip()

    # Already looks like a bare video ID (11 chars, no slashes/dots)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url_or_id):
        return url_or_id

    patterns = [
        r"(?:v=|/videos/|embed/|shorts/)([A-Za-z0-9_-]{11})",
        r"youtu\.be/([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)

    raise ValueError(f"Could not extract a video ID from: {url_or_id}")


def fetch_comments(video_id: str, max_comments: int = 50) -> List[Dict]:
    """
    Fetches top-level comments for a video via the YouTube Data API.
    Requires the YOUTUBE_API_KEY environment variable to be set.
    """
    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "YOUTUBE_API_KEY environment variable is not set. "
            "Create a .env file (see README) with YOUTUBE_API_KEY=your_key_here."
        )

    comments = []
    page_token = None

    while len(comments) < max_comments:
        params = {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": min(100, max_comments - len(comments)),
            "order": "relevance",
            "textFormat": "plainText",
            "key": api_key,
        }
        if page_token:
            params["pageToken"] = page_token

        response = requests.get(YOUTUBE_API_URL, params=params, timeout=10)

        if response.status_code == 403:
            raise RuntimeError(
                "YouTube API returned 403 Forbidden. Common causes: comments are "
                "disabled on this video, your API key is invalid, or your daily "
                "quota is exhausted."
            )
        if response.status_code == 404:
            raise RuntimeError("Video not found. Check the URL/ID is correct.")
        response.raise_for_status()

        data = response.json()
        for item in data.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "author": snippet.get("authorDisplayName", "Unknown"),
                "text": snippet.get("textDisplay", ""),
                "like_count": snippet.get("likeCount", 0),
                "published_at": snippet.get("publishedAt", ""),
            })

        page_token = data.get("nextPageToken")
        if not page_token:
            break

    return comments[:max_comments]
