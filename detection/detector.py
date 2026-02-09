"""
Death screen detection engine.

Uses multiple strategies to detect death screens:
1. Template matching - slide a reference image across the frame to find it
2. Color analysis - check if the frame matches death screen color profiles
3. Brightness analysis - death screens are often very dark or very bright
4. Fade detection - detect full-screen fade to black/red
5. Scene change detection - detect sudden transitions between frames

Templates are sub-images (e.g. a cropped "YOU DIED" text). The detector
scans the frame at multiple scales to find them, like ctrl+F for images.
"""

import logging
import time
from pathlib import Path

import cv2
import numpy as np

from game_profiles.profiles import GameProfile, ScreenRegion

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent.parent / "game_profiles" / "templates"

# Scales to try when searching for a template in the frame.
# Covers cases where the stream resolution doesn't match the screenshot.
SEARCH_SCALES = [1.0, 0.75, 0.5, 1.25, 1.5]


class DeathDetector:
    def __init__(
        self,
        profile: GameProfile,
        threshold: float = 0.80,
        cooldown: int = 15,
    ):
        self.profile = profile
        self.threshold = threshold
        self.cooldown = profile.cooldown_override or cooldown
        self.templates: list[np.ndarray] = []  # grayscale templates
        self.last_detection_time: float = 0.0
        self.previous_frame: np.ndarray | None = None
        self.death_frame_count: int = 0
        self.required_consecutive: int = 2

        self._load_templates()

    def _load_templates(self) -> None:
        """Load reference template images (sub-images to search for)."""
        template_path = TEMPLATES_DIR / self.profile.template_dir
        if not template_path.exists():
            logger.warning(
                "No template directory at %s — template matching disabled.",
                template_path,
            )
            return

        for img_file in sorted(template_path.iterdir()):
            if img_file.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
                img = cv2.imread(str(img_file))
                if img is not None:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    self.templates.append(gray)
                    logger.info(
                        "Loaded template: %s (%dx%d)",
                        img_file.name,
                        gray.shape[1],
                        gray.shape[0],
                    )

        logger.info(
            "Loaded %d template(s) for %s",
            len(self.templates),
            self.profile.display_name,
        )

    def _extract_region(
        self, frame: np.ndarray, region: ScreenRegion
    ) -> np.ndarray:
        """Extract a sub-region from a frame using normalized coordinates."""
        h, w = frame.shape[:2]
        x1 = int(region.x_min * w)
        y1 = int(region.y_min * h)
        x2 = int(region.x_max * w)
        y2 = int(region.y_max * h)
        return frame[y1:y2, x1:x2]

    def _template_match_score(self, frame: np.ndarray) -> float:
        """
        Slide each template across the frame at multiple scales.
        Returns the best match score (0-1). This is a true sub-image
        search — the template can be found anywhere on screen.
        """
        if not self.templates:
            return 0.0

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_h, frame_w = frame_gray.shape[:2]

        best_score = 0.0

        for template in self.templates:
            tmpl_h, tmpl_w = template.shape[:2]

            for scale in SEARCH_SCALES:
                # Resize template to this scale
                new_w = int(tmpl_w * scale)
                new_h = int(tmpl_h * scale)

                # Template must be smaller than the frame
                if new_w >= frame_w or new_h >= frame_h:
                    continue
                if new_w < 10 or new_h < 10:
                    continue

                scaled = cv2.resize(template, (new_w, new_h))

                # If we have screen regions, search within each region
                # Otherwise search the whole frame
                search_areas = []
                if self.profile.screen_regions:
                    for region in self.profile.screen_regions:
                        area = self._extract_region(frame_gray, region)
                        if area.shape[0] > new_h and area.shape[1] > new_w:
                            search_areas.append(area)

                if not search_areas:
                    search_areas = [frame_gray]

                for area in search_areas:
                    result = cv2.matchTemplate(
                        area, scaled, cv2.TM_CCOEFF_NORMED
                    )
                    _, max_val, _, _ = cv2.minMaxLoc(result)
                    best_score = max(best_score, max_val)

                    # Early exit if we already found a strong match
                    if best_score > 0.85:
                        return best_score

        return best_score

    def _color_analysis_score(self, frame: np.ndarray) -> float:
        """Score based on how much of the frame matches death screen colors."""
        if not self.profile.dominant_colors:
            return 0.0

        total_pixels = frame.shape[0] * frame.shape[1]
        max_ratio = 0.0

        for color_range in self.profile.dominant_colors:
            lower = np.array(color_range.lower, dtype=np.uint8)
            upper = np.array(color_range.upper, dtype=np.uint8)
            mask = cv2.inRange(frame, lower, upper)
            ratio = np.count_nonzero(mask) / total_pixels
            max_ratio = max(max_ratio, ratio)

        return max_ratio

    def _brightness_score(self, frame: np.ndarray) -> float:
        """Score based on overall brightness matching death screen profile."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)

        score = 0.0

        if self.profile.max_brightness is not None:
            if mean_brightness <= self.profile.max_brightness:
                score = max(
                    score,
                    1.0 - (mean_brightness / self.profile.max_brightness),
                )

        if self.profile.min_brightness is not None:
            if mean_brightness >= self.profile.min_brightness:
                score = max(
                    score,
                    min(1.0, mean_brightness / 255.0),
                )

        return score

    def _fade_detection_score(self, frame: np.ndarray) -> float:
        """Detect if the screen has faded to the death color."""
        if self.profile.fade_to_color is None:
            return 0.0

        lower = np.array(self.profile.fade_to_color.lower, dtype=np.uint8)
        upper = np.array(self.profile.fade_to_color.upper, dtype=np.uint8)
        mask = cv2.inRange(frame, lower, upper)

        total_pixels = frame.shape[0] * frame.shape[1]
        fade_ratio = np.count_nonzero(mask) / total_pixels

        if fade_ratio > 0.85:
            return fade_ratio
        return fade_ratio * 0.5

    def _scene_change_score(self, frame: np.ndarray) -> float:
        """Detect sudden scene changes that may indicate death."""
        if self.previous_frame is None:
            return 0.0

        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        previous_gray = cv2.cvtColor(self.previous_frame, cv2.COLOR_BGR2GRAY)

        size = (320, 240)
        current_small = cv2.resize(current_gray, size)
        previous_small = cv2.resize(previous_gray, size)

        diff = cv2.absdiff(current_small, previous_small)
        mean_diff = np.mean(diff)

        score = min(1.0, mean_diff / 80.0)
        return score

    def analyze_frame(self, frame: np.ndarray) -> tuple[bool, float]:
        """
        Analyze a frame for death screen indicators.

        Returns:
            (is_death, confidence) - whether death was detected and confidence 0-1
        """
        now = time.time()

        # Check cooldown
        if now - self.last_detection_time < self.cooldown:
            self.previous_frame = frame.copy()
            return False, 0.0

        # Compute individual scores
        template_score = self._template_match_score(frame)
        color_score = self._color_analysis_score(frame)
        brightness_score = self._brightness_score(frame)
        fade_score = self._fade_detection_score(frame)
        scene_change = self._scene_change_score(frame)

        # If template matching finds a strong match, trust it heavily
        if self.templates:
            if template_score > 0.75:
                # Strong template match — weight it very high
                confidence = (
                    template_score * 0.60
                    + color_score * 0.10
                    + brightness_score * 0.10
                    + fade_score * 0.10
                    + scene_change * 0.10
                )
            else:
                confidence = (
                    template_score * 0.35
                    + color_score * 0.15
                    + brightness_score * 0.15
                    + fade_score * 0.20
                    + scene_change * 0.15
                )
        else:
            confidence = (
                color_score * 0.30
                + brightness_score * 0.25
                + fade_score * 0.25
                + scene_change * 0.20
            )

        self.previous_frame = frame.copy()

        # Require consecutive frames to reduce false positives
        if confidence >= self.threshold:
            self.death_frame_count += 1
        else:
            self.death_frame_count = 0

        is_death = self.death_frame_count >= self.required_consecutive

        if is_death:
            self.last_detection_time = now
            self.death_frame_count = 0
            logger.info(
                "DEATH DETECTED! Confidence: %.2f (template=%.2f, color=%.2f, "
                "brightness=%.2f, fade=%.2f, scene_change=%.2f)",
                confidence,
                template_score,
                color_score,
                brightness_score,
                fade_score,
                scene_change,
            )

        return is_death, confidence
