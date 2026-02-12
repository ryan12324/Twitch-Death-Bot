"""
Game profile definitions for death screen detection.

Each profile describes visual characteristics of a game's death screen:
- dominant_colors: BGR color ranges that dominate the death screen
- screen_regions: where on screen to look (normalized 0-1 coordinates)
- template_dir: subdirectory under game_profiles/templates/ for reference images
- cooldown_override: optional per-game cooldown override in seconds

Profiles are loaded from individual JSON files in the game_profiles/ directory.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_WEIGHTS: dict[str, float] = {
    "template": 5.0,
    "text": 4.0,
    "embedding": 4.0,
    "color": 3.0,
    "brightness": 2.0,
    "fade": 3.0,
    "scene_change": 1.0,
}


@dataclass
class ScreenRegion:
    """A normalized region of the screen (0.0 to 1.0)."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass
class ColorRange:
    """A BGR color range for detection."""
    lower: tuple  # (B, G, R) lower bound
    upper: tuple  # (B, G, R) upper bound


@dataclass
class GameProfile:
    name: str
    display_name: str
    # Where on screen the death indicator typically appears
    screen_regions: list[ScreenRegion] = field(default_factory=list)
    # Dominant color ranges on the death screen (BGR)
    dominant_colors: list[ColorRange] = field(default_factory=list)
    # Directory name for template images
    template_dir: str = ""
    # Minimum seconds between death detections
    cooldown_override: int | None = None
    # Whether the screen fades to black/red on death
    fade_to_color: ColorRange | None = None
    # Brightness threshold - death screens are often dark
    max_brightness: int | None = None
    # Minimum brightness - some death screens flash bright
    min_brightness: int | None = None
    # Text strings to look for via OCR (e.g. "YOU DIED")
    text_indicators: list[str] = field(default_factory=list)
    # Per-signal weights for confidence calculation
    weights: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# JSON <-> dataclass conversion
# ---------------------------------------------------------------------------

_PROFILES_DIR = Path(__file__).parent


def _parse_color_range(data: dict) -> ColorRange:
    return ColorRange(lower=tuple(data["lower"]), upper=tuple(data["upper"]))


def _parse_profile(data: dict) -> GameProfile:
    return GameProfile(
        name=data["name"],
        display_name=data["display_name"],
        template_dir=data.get("template_dir", ""),
        screen_regions=[
            ScreenRegion(r["x_min"], r["y_min"], r["x_max"], r["y_max"])
            for r in data.get("screen_regions", [])
        ],
        dominant_colors=[
            _parse_color_range(c) for c in data.get("dominant_colors", [])
        ],
        fade_to_color=(
            _parse_color_range(data["fade_to_color"])
            if data.get("fade_to_color")
            else None
        ),
        max_brightness=data.get("max_brightness"),
        min_brightness=data.get("min_brightness"),
        cooldown_override=data.get("cooldown_override"),
        text_indicators=data.get("text_indicators", []),
        weights=data.get("weights", {}),
    )


def profile_to_dict(profile: GameProfile) -> dict:
    """Serialize a GameProfile to a JSON-compatible dict."""
    def _cr(cr: ColorRange) -> dict:
        return {"lower": list(cr.lower), "upper": list(cr.upper)}

    return {
        "name": profile.name,
        "display_name": profile.display_name,
        "template_dir": profile.template_dir,
        "screen_regions": [
            {"x_min": r.x_min, "y_min": r.y_min, "x_max": r.x_max, "y_max": r.y_max}
            for r in profile.screen_regions
        ],
        "dominant_colors": [_cr(c) for c in profile.dominant_colors],
        "fade_to_color": _cr(profile.fade_to_color) if profile.fade_to_color else None,
        "max_brightness": profile.max_brightness,
        "min_brightness": profile.min_brightness,
        "cooldown_override": profile.cooldown_override,
        "text_indicators": profile.text_indicators,
        "weights": profile.weights,
    }


def _load_profiles() -> dict[str, GameProfile]:
    """Scan game_profiles/*.json and return a dict of name -> GameProfile."""
    profiles: dict[str, GameProfile] = {}
    for path in sorted(_PROFILES_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            profile = _parse_profile(data)
            profiles[profile.name] = profile
        except Exception as e:
            print(f"Warning: failed to load profile {path}: {e}")
    return profiles


def reload_profiles() -> None:
    """Reload all profiles from disk (clears and repopulates PROFILES in-place)."""
    PROFILES.clear()
    PROFILES.update(_load_profiles())


def save_profile(data: dict) -> GameProfile:
    """Validate, write a profile JSON file, and reload."""
    name = data.get("name", "").strip()
    if not name:
        raise ValueError("Profile name is required")
    # Ensure we can parse it
    profile = _parse_profile(data)
    path = _PROFILES_DIR / f"{name}.json"
    path.write_text(json.dumps(profile_to_dict(profile), indent=2) + "\n")
    reload_profiles()
    return profile


def delete_profile(name: str) -> None:
    """Delete a profile's JSON file and reload."""
    if name == "generic":
        raise ValueError("Cannot delete the generic profile")
    path = _PROFILES_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Profile '{name}' not found")
    path.unlink()
    reload_profiles()


# ---------------------------------------------------------------------------
# Module-level registry (populated at import time)
# ---------------------------------------------------------------------------

PROFILES: dict[str, GameProfile] = _load_profiles()


def get_profile(name: str) -> GameProfile:
    """Get a game profile by name."""
    if name not in PROFILES:
        available = ", ".join(PROFILES.keys())
        raise ValueError(f"Unknown game profile '{name}'. Available: {available}")
    return PROFILES[name]


def list_profiles() -> list[str]:
    """List all available profile names."""
    return list(PROFILES.keys())
