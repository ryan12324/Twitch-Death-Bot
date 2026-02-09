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
logger.info("TEMPLATES_DIR resolved to: %s (exists=%s)", TEMPLATES_DIR, TEMPLATES_DIR.exists())

# Scales to try when searching for a template in the frame.
# Wide range covers high-res templates matched against lower-res frames.
SEARCH_SCALES = [1.0, 0.75, 0.5, 0.35, 0.25, 0.15, 1.25, 1.5]

# All frames are normalized to this resolution before analysis so that
# detection behaves identically regardless of input size.
WORKING_W, WORKING_H = 1280, 720


class DeathDetector:
    def __init__(
        self,
        profile: GameProfile,
        threshold: float = 0.80,
        cooldown: int = 15,
        required_consecutive: int = 2,
    ):
        self.profile = profile
        self.threshold = threshold
        self.cooldown = profile.cooldown_override or cooldown
        self.templates: list[np.ndarray] = []  # grayscale templates
        self.last_detection_time: float = 0.0
        self.previous_frame: np.ndarray | None = None
        self.death_frame_count: int = 0
        self.required_consecutive: int = required_consecutive

        # Last computed individual scores (populated by analyze_frame)
        self.last_scores: dict[str, float] = {}

        logger.info(
            "DeathDetector init: profile=%s threshold=%.2f cooldown=%d "
            "required_consecutive=%d",
            profile.display_name, threshold, self.cooldown, required_consecutive,
        )

        self._load_templates()

    def _load_templates(self) -> None:
        """Load reference template images (sub-images to search for)."""
        template_path = TEMPLATES_DIR / self.profile.template_dir
        logger.info(
            "Looking for templates in: %s (exists=%s)",
            template_path, template_path.exists(),
        )
        if not template_path.exists():
            logger.warning(
                "No template directory at %s — template matching disabled.",
                template_path,
            )
            return

        files_found = list(sorted(template_path.iterdir()))
        logger.info(
            "Files in template dir: %s",
            [f.name for f in files_found],
        )

        for img_file in files_found:
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
                else:
                    logger.error(
                        "FAILED to load template (cv2.imread returned None): %s",
                        img_file,
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

    def _compute_scales(
        self, tmpl_w: int, tmpl_h: int, frame_w: int, frame_h: int
    ) -> list[float]:
        """Build the list of scales to try, including auto-fit for oversized templates."""
        scales = list(SEARCH_SCALES)

        if tmpl_w > frame_w * 0.5 or tmpl_h > frame_h * 0.5:
            # Tight fit: template nearly fills the frame — best match when
            # the uploaded image IS (or is very close to) the template itself
            tight = min((frame_w - 2) / tmpl_w, (frame_h - 2) / tmpl_h)
            if 0.05 < tight < 3.0:
                scales.append(round(tight, 4))

            # 80% / 60% / 40% fits for more breathing room
            fit = min((frame_w * 0.8) / tmpl_w, (frame_h * 0.8) / tmpl_h)
            for s in [fit, fit * 0.75, fit * 0.5]:
                if 0.05 < s < 3.0 and round(s, 4) not in scales:
                    scales.append(round(s, 4))

        logger.info(
            "scales: template %dx%d in frame %dx%d -> %s",
            tmpl_w, tmpl_h, frame_w, frame_h, scales,
        )
        return scales

    def _template_match_score(self, frame: np.ndarray) -> float:
        """
        Slide each template across the frame at multiple scales.
        Returns the best match score (0-1). This is a true sub-image
        search — the template can be found anywhere on screen.

        Uses both pixel-based AND edge-based matching. Edge matching
        is critical for games like Dark Souls where the death overlay
        is semi-transparent — the text edges stay the same but the
        background pixels change depending on where you die.
        """
        if not self.templates:
            logger.info("template_match: no templates loaded, returning 0.0")
            return 0.0

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_h, frame_w = frame_gray.shape[:2]
        frame_edges = cv2.Canny(frame_gray, 50, 150)

        logger.info("template_match: frame=%dx%d, %d template(s)", frame_w, frame_h, len(self.templates))

        best_score = 0.0

        for t_idx, template in enumerate(self.templates):
            tmpl_h, tmpl_w = template.shape[:2]
            scales = self._compute_scales(tmpl_w, tmpl_h, frame_w, frame_h)
            logger.info(
                "template_match: template[%d] %dx%d, trying %d scales: %s",
                t_idx, tmpl_w, tmpl_h, len(scales), scales,
            )

            for scale in scales:
                new_w = int(tmpl_w * scale)
                new_h = int(tmpl_h * scale)

                if new_w >= frame_w or new_h >= frame_h:
                    logger.debug(
                        "  scale=%.4f -> %dx%d SKIPPED (too large for %dx%d frame)",
                        scale, new_w, new_h, frame_w, frame_h,
                    )
                    continue
                if new_w < 10 or new_h < 10:
                    logger.debug(
                        "  scale=%.4f -> %dx%d SKIPPED (too small)",
                        scale, new_w, new_h,
                    )
                    continue

                scaled = cv2.resize(template, (new_w, new_h))
                scaled_edges = cv2.Canny(scaled, 50, 150)

                # Build search areas from screen regions, fall back to full frame
                search_areas_gray = []
                search_areas_edge = []
                if self.profile.screen_regions:
                    for region in self.profile.screen_regions:
                        area_g = self._extract_region(frame_gray, region)
                        area_e = self._extract_region(frame_edges, region)
                        if area_g.shape[0] > new_h and area_g.shape[1] > new_w:
                            search_areas_gray.append(area_g)
                            search_areas_edge.append(area_e)
                        else:
                            logger.debug(
                                "  scale=%.4f region too small (%dx%d) for template %dx%d",
                                scale, area_g.shape[1], area_g.shape[0], new_w, new_h,
                            )

                using_regions = len(search_areas_gray) > 0
                if not search_areas_gray:
                    search_areas_gray = [frame_gray]
                    search_areas_edge = [frame_edges]

                for area_g, area_e in zip(search_areas_gray, search_areas_edge):
                    # Pixel-based match
                    result = cv2.matchTemplate(
                        area_g, scaled, cv2.TM_CCOEFF_NORMED
                    )
                    _, max_val, _, max_loc = cv2.minMaxLoc(result)

                    # Edge-based match — robust to background changes
                    result_e = cv2.matchTemplate(
                        area_e, scaled_edges, cv2.TM_CCOEFF_NORMED
                    )
                    _, max_val_e, _, max_loc_e = cv2.minMaxLoc(result_e)

                    logger.info(
                        "  scale=%.4f -> %dx%d | area=%dx%d region=%s | "
                        "pixel=%.4f@(%d,%d) edge=%.4f@(%d,%d) | best=%.4f",
                        scale, new_w, new_h,
                        area_g.shape[1], area_g.shape[0],
                        using_regions,
                        max_val, max_loc[0], max_loc[1],
                        max_val_e, max_loc_e[0], max_loc_e[1],
                        best_score,
                    )

                    best_score = max(best_score, max_val, max_val_e)

                    if best_score > 0.85:
                        logger.info("  EARLY EXIT: best_score=%.4f > 0.85", best_score)
                        return best_score

        logger.info("template_match: final best_score=%.4f", best_score)
        return best_score

    def _color_analysis_score(self, frame: np.ndarray) -> float:
        """Score based on how much of the frame matches death screen colors."""
        if not self.profile.dominant_colors:
            logger.info("color_analysis: no dominant_colors in profile, returning 0.0")
            return 0.0

        total_pixels = frame.shape[0] * frame.shape[1]
        max_ratio = 0.0

        for i, color_range in enumerate(self.profile.dominant_colors):
            lower = np.array(color_range.lower, dtype=np.uint8)
            upper = np.array(color_range.upper, dtype=np.uint8)
            mask = cv2.inRange(frame, lower, upper)
            ratio = np.count_nonzero(mask) / total_pixels
            logger.info(
                "color_analysis: range[%d] BGR(%s)-(%s) -> ratio=%.4f",
                i, color_range.lower, color_range.upper, ratio,
            )
            max_ratio = max(max_ratio, ratio)

        logger.info("color_analysis: final score=%.4f", max_ratio)
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
                logger.info(
                    "brightness: mean=%.1f <= max_thresh=%d -> score=%.4f",
                    mean_brightness, self.profile.max_brightness, score,
                )
            else:
                logger.info(
                    "brightness: mean=%.1f > max_thresh=%d -> no match",
                    mean_brightness, self.profile.max_brightness,
                )

        if self.profile.min_brightness is not None:
            if mean_brightness >= self.profile.min_brightness:
                new_score = min(1.0, mean_brightness / 255.0)
                score = max(score, new_score)
                logger.info(
                    "brightness: mean=%.1f >= min_thresh=%d -> score=%.4f",
                    mean_brightness, self.profile.min_brightness, score,
                )
            else:
                logger.info(
                    "brightness: mean=%.1f < min_thresh=%d -> no match",
                    mean_brightness, self.profile.min_brightness,
                )

        if self.profile.max_brightness is None and self.profile.min_brightness is None:
            logger.info("brightness: no thresholds in profile, returning 0.0")

        logger.info("brightness: final score=%.4f (mean=%.1f)", score, mean_brightness)
        return score

    def _fade_detection_score(self, frame: np.ndarray) -> float:
        """Detect if the screen has faded to the death color."""
        if self.profile.fade_to_color is None:
            logger.info("fade: no fade_to_color in profile, returning 0.0")
            return 0.0

        lower = np.array(self.profile.fade_to_color.lower, dtype=np.uint8)
        upper = np.array(self.profile.fade_to_color.upper, dtype=np.uint8)
        mask = cv2.inRange(frame, lower, upper)

        total_pixels = frame.shape[0] * frame.shape[1]
        fade_ratio = np.count_nonzero(mask) / total_pixels

        if fade_ratio > 0.85:
            score = fade_ratio
        else:
            score = fade_ratio * 0.5

        logger.info(
            "fade: BGR(%s)-(%s) -> ratio=%.4f, score=%.4f",
            self.profile.fade_to_color.lower,
            self.profile.fade_to_color.upper,
            fade_ratio, score,
        )
        return score

    def _scene_change_score(self, frame: np.ndarray) -> float:
        """Detect sudden scene changes that may indicate death."""
        if self.previous_frame is None:
            logger.info("scene_change: no previous frame, returning 0.0")
            return 0.0

        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        previous_gray = cv2.cvtColor(self.previous_frame, cv2.COLOR_BGR2GRAY)

        size = (320, 240)
        current_small = cv2.resize(current_gray, size)
        previous_small = cv2.resize(previous_gray, size)

        diff = cv2.absdiff(current_small, previous_small)
        mean_diff = np.mean(diff)

        score = min(1.0, mean_diff / 80.0)
        logger.info("scene_change: mean_diff=%.1f -> score=%.4f", mean_diff, score)
        return score

    @staticmethod
    def _normalize_frame(frame: np.ndarray) -> np.ndarray:
        """Scale frame to fit within working resolution, preserving aspect ratio.

        Never stretches — only scales down proportionally if the frame is
        larger than WORKING_W x WORKING_H. This keeps template matching
        accurate because the pixel proportions stay intact.
        """
        h, w = frame.shape[:2]
        if w <= WORKING_W and h <= WORKING_H:
            logger.debug("normalize: %dx%d fits, no resize needed", w, h)
            return frame

        scale = min(WORKING_W / w, WORKING_H / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        logger.info(
            "normalize: %dx%d -> %dx%d (scale=%.4f, aspect preserved)",
            w, h, new_w, new_h, scale,
        )
        return cv2.resize(frame, (new_w, new_h))

    def analyze_frame(self, frame: np.ndarray) -> tuple[bool, float]:
        """
        Analyze a frame for death screen indicators.

        Returns:
            (is_death, confidence) - whether death was detected and confidence 0-1
        """
        logger.info(
            "=== analyze_frame START === input=%dx%d profile=%s",
            frame.shape[1], frame.shape[0], self.profile.name,
        )
        frame = self._normalize_frame(frame)
        now = time.time()

        # Check cooldown
        if now - self.last_detection_time < self.cooldown:
            logger.info("analyze_frame: in cooldown, skipping")
            self.previous_frame = frame.copy()
            return False, 0.0

        # Compute individual scores
        template_score = self._template_match_score(frame)
        color_score = self._color_analysis_score(frame)
        brightness_score = self._brightness_score(frame)
        fade_score = self._fade_detection_score(frame)
        scene_change = self._scene_change_score(frame)

        # Compute base confidence from persistent signals (things that stay
        # true for every frame of a death screen, not just the transition).
        if self.templates and template_score > 0.75:
            weight_tier = "STRONG_TEMPLATE"
            confidence = (
                template_score * 0.65
                + color_score * 0.10
                + brightness_score * 0.10
                + fade_score * 0.15
            )
        elif self.templates and template_score > 0.40:
            weight_tier = "MODERATE_TEMPLATE"
            confidence = (
                template_score * 0.40
                + color_score * 0.20
                + brightness_score * 0.15
                + fade_score * 0.25
            )
        else:
            weight_tier = "VISUAL_ONLY"
            confidence = (
                color_score * 0.35
                + brightness_score * 0.30
                + fade_score * 0.35
            )

        logger.info(
            "weighting: tier=%s templates_loaded=%d template_score=%.4f",
            weight_tier, len(self.templates), template_score,
        )

        # Scene change is a bonus for transitions, never a penalty.
        # Death screens persist across multiple frames, so low scene change
        # on the second/third frame should not drag confidence down.
        if self.previous_frame is not None and scene_change > 0.3:
            old_conf = confidence
            confidence = min(1.0, confidence + scene_change * 0.05)
            logger.info(
                "scene bonus: %.4f -> %.4f (scene=%.4f)",
                old_conf, confidence, scene_change,
            )

        # Store individual scores for the web GUI
        self.last_scores = {
            "template": round(template_score, 4),
            "color": round(color_score, 4),
            "brightness": round(brightness_score, 4),
            "fade": round(fade_score, 4),
            "scene_change": round(scene_change, 4),
            "confidence": round(confidence, 4),
        }

        self.previous_frame = frame.copy()

        # Require consecutive frames to reduce false positives
        if confidence >= self.threshold:
            self.death_frame_count += 1
        else:
            self.death_frame_count = 0

        is_death = self.death_frame_count >= self.required_consecutive

        logger.info(
            "=== analyze_frame END === confidence=%.4f threshold=%.2f "
            "consecutive=%d/%d is_death=%s | "
            "template=%.4f color=%.4f brightness=%.4f fade=%.4f scene=%.4f",
            confidence, self.threshold,
            self.death_frame_count, self.required_consecutive, is_death,
            template_score, color_score, brightness_score, fade_score, scene_change,
        )

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
