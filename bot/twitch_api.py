"""
Twitch Helix API helper for game detection.

Uses urllib.request to avoid adding a requests dependency.
"""

import json
import logging
import urllib.request
import urllib.error

from game_profiles.profiles import PROFILES

logger = logging.getLogger(__name__)


def get_channel_game(
    channel: str, client_id: str, oauth_token: str
) -> dict | None:
    """Detect what game a channel is currently playing.

    Calls GET https://api.twitch.tv/helix/streams?user_login={channel}

    Returns {"game_id": "512953", "game_name": "Elden Ring"} or None if
    the channel is offline or the request fails.
    """
    url = (
        f"https://api.twitch.tv/helix/streams?user_login="
        f"{urllib.request.quote(channel.strip().lower())}"
    )
    # Strip "oauth:" prefix if present
    token = oauth_token.strip()
    if token.lower().startswith("oauth:"):
        token = token[6:]

    req = urllib.request.Request(url, headers={
        "Client-Id": client_id.strip(),
        "Authorization": f"Bearer {token}",
    })

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
        logger.warning("Twitch Helix API error for %s: %s", channel, exc)
        return None

    streams = data.get("data", [])
    if not streams:
        return None

    stream = streams[0]
    return {
        "game_id": stream.get("game_id", ""),
        "game_name": stream.get("game_name", ""),
    }


def match_profile_by_game_id(game_id: str) -> str | None:
    """Find a profile whose twitch_game_id matches *game_id*.

    Returns the profile name, or None if no match is found.
    """
    if not game_id:
        return None

    for name, profile in PROFILES.items():
        if profile.twitch_game_id and profile.twitch_game_id == game_id:
            return name

    return None
