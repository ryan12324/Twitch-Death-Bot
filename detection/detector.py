"""
Death screen detection engine.

Uses multiple strategies to detect death screens:
1. Template matching - compare against known death screen reference images
2. Color analysis - check if the frame matches death screen color profiles
3. Structural similarity - detect sudden scene changes to dark/death screens

All strategies are combined with configurable weights for final confidence.
"""

import logging
import os
import time
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from game_profiles.profiles import GameProfile, ScreenRegion

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent.parent / "game_profiles" / "templates"


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
        self.templates: list[np.ndarray] = []
        self.last_detection_time: float = 0.0
        self.previous_frame: np.ndarray | None = None
        self.death_frame_count: int = 0  # consecutive frames that look like death
        self.required_consecutive: int = 2  # need N consecutive frames to confirm

        self._load_templates()

    def _load_templates(self) -> None:
        """Load reference death screen template images for the current game."""
        template_path = TEMPLATES_DIR / self.profile.template_dir
        if not template_path.exists():
            logger.warning(
                "No template directory found at %s. "
                "Template matching will be skipped. "
                "Detection will rely on color/brightness analysis.",
                template_path,
            )
            return

        for img_file in sorted(template_path.iterdir()):
            if img_file.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
                template = cv2.imread(str(img_file))
                if template is not None:
                    self.templates.append(template)
                    logger.info("Loaded template: %s", img_file.name)

        logger.info(
            "Loaded %d templates for %s",
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
        """Score how well the frame matches any loaded death screen template."""
        if not self.templates:
            return 0.0

        best_score = 0.0
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for template in self.templates:
            template_resized = cv2.resize(
                template, (frame.shape[1], frame.shape[0])
            )
            template_gray = cv2.cvtColor(template_resized, cv2.COLOR_BGR2GRAY)

            # Structural similarity
            score, _ = ssim(frame_gray, template_gray, full=True)
            best_score = max(best_score, score)

            # Also try normalized cross-correlation on regions
            for region in self.profile.screen_regions:
                region_frame = self._extract_region(frame_gray, region)
                region_template = self._extract_region(template_gray, region)

                if region_frame.size == 0 or region_template.size == 0:
                    continue

                # Resize template region to match frame region
                region_template = cv2.resize(
                    region_template,
                    (region_frame.shape[1], region_frame.shape[0]),
                )

                result = cv2.matchTemplate(
                    region_frame, region_template, cv2.TM_CCOEFF_NORMED
                )
                _, max_val, _, _ = cv2.minMaxLoc(result)
                best_score = max(best_score, max_val)

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
                # The darker it is (relative to threshold), the higher the score
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

        # Need a very high ratio of pixels to be in the fade color range
        if fade_ratio > 0.85:
            return fade_ratio
        return fade_ratio * 0.5

    def _scene_change_score(self, frame: np.ndarray) -> float:
        """Detect sudden scene changes that may indicate death."""
        if self.previous_frame is None:
            return 0.0

        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        previous_gray = cv2.cvtColor(self.previous_frame, cv2.COLOR_BGR2GRAY)

        # Resize to common size for comparison
        size = (320, 240)
        current_small = cv2.resize(current_gray, size)
        previous_small = cv2.resize(previous_gray, size)

        # Calculate absolute difference
        diff = cv2.absdiff(current_small, previous_small)
        mean_diff = np.mean(diff)

        # Normalize: a mean diff of 80+ is a major scene change
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

        # Weighted combination depends on available data
        if self.templates:
            # Template matching is most reliable when templates exist
            confidence = (
                template_score * 0.40
                + color_score * 0.15
                + brightness_score * 0.15
                + fade_score * 0.15
                + scene_change * 0.15
            )
        else:
            # Without templates, rely more on color/brightness
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
