"""
Game profile definitions for death screen detection.

Each profile describes visual characteristics of a game's death screen:
- text_indicators: text strings that appear on death (used with OCR-free template matching)
- dominant_colors: BGR color ranges that dominate the death screen
- screen_regions: where on screen to look (normalized 0-1 coordinates)
- template_dir: subdirectory under game_profiles/templates/ for reference images
- cooldown_override: optional per-game cooldown override in seconds
"""

from dataclasses import dataclass, field


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


# --- Game Profile Definitions ---

ELDEN_RING = GameProfile(
    name="elden_ring",
    display_name="Elden Ring",
    screen_regions=[
        # "YOU DIED" text appears center screen
        ScreenRegion(0.25, 0.35, 0.75, 0.65),
    ],
    dominant_colors=[
        # Dark/black background with red text
        ColorRange((0, 0, 100), (80, 80, 255)),   # Red text
        ColorRange((0, 0, 0), (50, 50, 50)),       # Dark background
    ],
    template_dir="elden_ring",
    fade_to_color=ColorRange((0, 0, 0), (40, 40, 40)),
    max_brightness=60,
)

DARK_SOULS_2 = GameProfile(
    name="dark_souls_2",
    display_name="Dark Souls II",
    screen_regions=[
        # "YOU DIED" text appears center screen
        ScreenRegion(0.25, 0.35, 0.75, 0.65),
    ],
    dominant_colors=[
        ColorRange((0, 0, 100), (70, 70, 255)),   # Red/orange text
        ColorRange((0, 0, 0), (45, 45, 45)),       # Dark background
    ],
    template_dir="dark_souls_2",
    fade_to_color=ColorRange((0, 0, 0), (40, 40, 40)),
    max_brightness=55,
)

DARK_SOULS_3 = GameProfile(
    name="dark_souls_3",
    display_name="Dark Souls III",
    screen_regions=[
        # "YOU DIED" center screen, similar to Elden Ring
        ScreenRegion(0.25, 0.35, 0.75, 0.65),
    ],
    dominant_colors=[
        ColorRange((0, 0, 120), (60, 60, 255)),   # Red text
        ColorRange((0, 0, 0), (40, 40, 40)),       # Black background
    ],
    template_dir="dark_souls_3",
    fade_to_color=ColorRange((0, 0, 0), (40, 40, 40)),
    max_brightness=55,
)

SEKIRO = GameProfile(
    name="sekiro",
    display_name="Sekiro: Shadows Die Twice",
    screen_regions=[
        # Large kanji/text appears center screen on death
        ScreenRegion(0.20, 0.25, 0.80, 0.75),
    ],
    dominant_colors=[
        ColorRange((0, 0, 150), (50, 50, 255)),   # Red kanji
        ColorRange((0, 0, 0), (30, 30, 30)),       # Black background
    ],
    template_dir="sekiro",
    fade_to_color=ColorRange((0, 0, 0), (35, 35, 35)),
    max_brightness=50,
)

HOLLOW_KNIGHT = GameProfile(
    name="hollow_knight",
    display_name="Hollow Knight",
    screen_regions=[
        # Death shade appears, screen goes dark
        ScreenRegion(0.10, 0.10, 0.90, 0.90),
    ],
    dominant_colors=[
        ColorRange((0, 0, 0), (30, 30, 30)),       # Very dark screen
    ],
    template_dir="hollow_knight",
    fade_to_color=ColorRange((0, 0, 0), (25, 25, 25)),
    max_brightness=35,
    cooldown_override=10,
)

CELESTE = GameProfile(
    name="celeste",
    display_name="Celeste",
    screen_regions=[
        # Death burst happens at player position (anywhere on screen)
        ScreenRegion(0.0, 0.0, 1.0, 1.0),
    ],
    dominant_colors=[
        # Screen flashes/freezes briefly
        ColorRange((200, 200, 200), (255, 255, 255)),  # White flash
    ],
    template_dir="celeste",
    cooldown_override=3,  # Celeste deaths are fast
    min_brightness=200,
)

GENERIC = GameProfile(
    name="generic",
    display_name="Generic (Any Game)",
    screen_regions=[
        # Check the whole center area
        ScreenRegion(0.15, 0.20, 0.85, 0.80),
    ],
    dominant_colors=[
        # Dark/black fade is the most common death indicator
        ColorRange((0, 0, 0), (45, 45, 45)),
        # Red tint also common
        ColorRange((0, 0, 80), (60, 60, 220)),
    ],
    template_dir="generic",
    fade_to_color=ColorRange((0, 0, 0), (45, 45, 45)),
    max_brightness=50,
)

# Registry of all profiles
PROFILES: dict[str, GameProfile] = {
    "elden_ring": ELDEN_RING,
    "dark_souls_2": DARK_SOULS_2,
    "dark_souls_3": DARK_SOULS_3,
    "sekiro": SEKIRO,
    "hollow_knight": HOLLOW_KNIGHT,
    "celeste": CELESTE,
    "generic": GENERIC,
}


def get_profile(name: str) -> GameProfile:
    """Get a game profile by name."""
    if name not in PROFILES:
        available = ", ".join(PROFILES.keys())
        raise ValueError(f"Unknown game profile '{name}'. Available: {available}")
    return PROFILES[name]


def list_profiles() -> list[str]:
    """List all available profile names."""
    return list(PROFILES.keys())
