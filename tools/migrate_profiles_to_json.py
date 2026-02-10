#!/usr/bin/env python3
"""One-time migration: export hardcoded GameProfile instances to individual JSON files."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from game_profiles.profiles import PROFILES, GameProfile, ColorRange, ScreenRegion


def _color_range_to_dict(cr: ColorRange) -> dict:
    return {"lower": list(cr.lower), "upper": list(cr.upper)}


def profile_to_dict(p: GameProfile) -> dict:
    return {
        "name": p.name,
        "display_name": p.display_name,
        "template_dir": p.template_dir,
        "screen_regions": [
            {"x_min": r.x_min, "y_min": r.y_min, "x_max": r.x_max, "y_max": r.y_max}
            for r in p.screen_regions
        ],
        "dominant_colors": [_color_range_to_dict(c) for c in p.dominant_colors],
        "fade_to_color": _color_range_to_dict(p.fade_to_color) if p.fade_to_color else None,
        "max_brightness": p.max_brightness,
        "min_brightness": p.min_brightness,
        "cooldown_override": p.cooldown_override,
    }


def main():
    out_dir = Path(__file__).parent.parent / "game_profiles"
    for name, profile in PROFILES.items():
        data = profile_to_dict(profile)
        path = out_dir / f"{name}.json"
        path.write_text(json.dumps(data, indent=2) + "\n")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
