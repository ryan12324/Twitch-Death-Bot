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

from game_profiles.profiles import DEFAULT_WEIGHTS, GameProfile, ScreenRegion

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
        """Load reference template images and pre-compute scaled versions."""
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

        # Pre-compute scaled templates + edge maps for all search scales
        self._scaled_cache: dict[tuple[int, float], tuple[np.ndarray, np.ndarray]] = {}
        for t_idx, template in enumerate(self.templates):
            tmpl_h, tmpl_w = template.shape[:2]
            for scale in SEARCH_SCALES:
                new_w = int(tmpl_w * scale)
                new_h = int(tmpl_h * scale)
                if new_w < 10 or new_h < 10:
                    continue
                scaled = cv2.resize(template, (new_w, new_h))
                scaled_edges = cv2.Canny(scaled, 50, 150)
                self._scaled_cache[(t_idx, scale)] = (scaled, scaled_edges)
        logger.info("Pre-cached %d scaled template variants", len(self._scaled_cache))

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

        return scales

    def _template_match_score(self, frame: np.ndarray, frame_gray: np.ndarray | None) -> float:
        """
        Slide each template across the frame at multiple scales.
        Returns the best match score (0-1). Uses pre-computed scaled
        template cache to avoid per-frame resize/Canny overhead.

        Uses both pixel-based AND edge-based matching. Edge matching
        is critical for games like Dark Souls where the death overlay
        is semi-transparent — the text edges stay the same but the
        background pixels change depending on where you die.
        """
        if not self.templates:
            self._last_match_loc = None
            return 0.0

        frame_h, frame_w = frame.shape[:2]

        # When frame_gray is None (lazy grayscale mode), convert per-region
        # or fall back to full-frame conversion if no regions.
        if frame_gray is None and not self.profile.screen_regions:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Pre-extract search regions once (reused across all scales).
        # Canny is computed per-region crop instead of full-frame — much cheaper
        # when screen_regions are defined (e.g. 200x100 crop vs 1280x720).
        search_regions: list[tuple[np.ndarray, np.ndarray, int, int]] = []
        full_frame_edges: np.ndarray | None = None  # lazily computed if no regions
        if self.profile.screen_regions:
            for region in self.profile.screen_regions:
                roi_bgr = self._extract_region(frame, region)
                roi_g = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY) if frame_gray is None else self._extract_region(frame_gray, region)
                roi_e = cv2.Canny(roi_g, 50, 150)
                ox = int(region.x_min * frame_w)
                oy = int(region.y_min * frame_h)
                search_regions.append((roi_g, roi_e, ox, oy))

        best_score = 0.0
        best_match_loc = None  # ((x, y), (w, h), score)

        for t_idx, template in enumerate(self.templates):
            tmpl_h, tmpl_w = template.shape[:2]
            scales = self._compute_scales(tmpl_w, tmpl_h, frame_w, frame_h)

            for scale in scales:
                # Use pre-computed cache when available, else compute on the fly
                cache_key = (t_idx, scale)
                if cache_key in self._scaled_cache:
                    scaled, scaled_edges = self._scaled_cache[cache_key]
                    new_w, new_h = scaled.shape[1], scaled.shape[0]
                else:
                    new_w = int(tmpl_w * scale)
                    new_h = int(tmpl_h * scale)
                    if new_w < 10 or new_h < 10:
                        continue
                    scaled = cv2.resize(template, (new_w, new_h))
                    scaled_edges = cv2.Canny(scaled, 50, 150)

                if new_w >= frame_w or new_h >= frame_h:
                    continue

                # Build search areas from pre-extracted regions
                search_areas: list[tuple[np.ndarray, np.ndarray, int, int]] = []
                if search_regions:
                    for roi_g, roi_e, ox, oy in search_regions:
                        if roi_g.shape[0] > new_h and roi_g.shape[1] > new_w:
                            search_areas.append((roi_g, roi_e, ox, oy))

                if not search_areas:
                    # No regions big enough (or none defined) — fall back to full frame
                    if frame_gray is None:
                        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if full_frame_edges is None:
                        full_frame_edges = cv2.Canny(frame_gray, 50, 150)
                    search_areas = [(frame_gray, full_frame_edges, 0, 0)]

                for area_g, area_e, ox, oy in search_areas:
                    # Guard: template must fit within search area
                    if area_g.shape[0] < new_h or area_g.shape[1] < new_w:
                        continue
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

                    # Track best match location for overlay
                    if max_val > best_score:
                        best_score = max_val
                        best_match_loc = (
                            (max_loc[0] + ox, max_loc[1] + oy),
                            (new_w, new_h), max_val,
                        )
                    if max_val_e > best_score:
                        best_score = max_val_e
                        best_match_loc = (
                            (max_loc_e[0] + ox, max_loc_e[1] + oy),
                            (new_w, new_h), max_val_e,
                        )

                    if best_score > 0.85:
                        self._last_match_loc = best_match_loc
                        return best_score

        self._last_match_loc = best_match_loc
        logger.debug("template_match: final best_score=%.4f", best_score)
        return best_score

    def _color_analysis_score(self, frame: np.ndarray) -> float:
        """Score based on how much of the frame matches death screen colors.

        When screen_regions are defined, color analysis is restricted to those
        regions so that HUD elements and other non-death-screen areas don't
        pollute the score.
        """
        if not self.profile.dominant_colors:
            return 0.0

        # Build list of areas to analyze
        if self.profile.screen_regions:
            areas = [self._extract_region(frame, r) for r in self.profile.screen_regions]
        else:
            areas = [frame]

        max_ratio = 0.0

        for color_range in self.profile.dominant_colors:
            lower = np.array(color_range.lower, dtype=np.uint8)
            upper = np.array(color_range.upper, dtype=np.uint8)
            for area in areas:
                total_pixels = area.shape[0] * area.shape[1]
                mask = cv2.inRange(area, lower, upper)
                ratio = np.count_nonzero(mask) / total_pixels
                max_ratio = max(max_ratio, ratio)

        logger.debug("color_analysis: final score=%.4f", max_ratio)
        return max_ratio

    def _brightness_score(self, frame: np.ndarray, frame_gray: np.ndarray | None) -> float:
        """Score based on overall brightness matching death screen profile.

        When screen_regions are defined, computes the mean brightness across
        all region crops (averaged) rather than the full frame.
        """
        if self.profile.screen_regions:
            brightness_values = []
            for region in self.profile.screen_regions:
                if frame_gray is not None:
                    crop_gray = self._extract_region(frame_gray, region)
                else:
                    crop_bgr = self._extract_region(frame, region)
                    crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
                brightness_values.append(np.mean(crop_gray))
            mean_brightness = float(np.mean(brightness_values))
        else:
            if frame_gray is None:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_brightness = float(np.mean(frame_gray))

        score = 0.0

        if self.profile.max_brightness is not None:
            if mean_brightness <= self.profile.max_brightness:
                score = max(
                    score,
                    1.0 - (mean_brightness / self.profile.max_brightness),
                )

        if self.profile.min_brightness is not None:
            if mean_brightness >= self.profile.min_brightness:
                new_score = min(1.0, mean_brightness / 255.0)
                score = max(score, new_score)

        logger.debug("brightness: final score=%.4f (mean=%.1f)", score, mean_brightness)
        return score

    def _fade_detection_score(self, frame: np.ndarray) -> float:
        """Detect if the screen has faded to the death color."""
        if self.profile.fade_to_color is None:
            return 0.0

        lower = np.array(self.profile.fade_to_color.lower, dtype=np.uint8)
        upper = np.array(self.profile.fade_to_color.upper, dtype=np.uint8)

        if self.profile.screen_regions:
            areas = [self._extract_region(frame, r) for r in self.profile.screen_regions]
        else:
            areas = [frame]

        total_pixels = sum(a.shape[0] * a.shape[1] for a in areas)
        fade_pixels = sum(np.count_nonzero(cv2.inRange(a, lower, upper)) for a in areas)
        fade_ratio = fade_pixels / total_pixels

        if fade_ratio > 0.85:
            score = fade_ratio
        else:
            score = fade_ratio * 0.5

        logger.debug("fade: ratio=%.4f, score=%.4f", fade_ratio, score)
        return score

    def _scene_change_score(self, frame_gray: np.ndarray) -> float:
        """Detect sudden scene changes that may indicate death.

        Expects frame_gray (current) and self.previous_frame (grayscale).
        """
        if self.previous_frame is None:
            return 0.0

        size = (320, 240)
        current_small = cv2.resize(frame_gray, size)
        previous_small = cv2.resize(self.previous_frame, size)

        diff = cv2.absdiff(current_small, previous_small)
        mean_diff = np.mean(diff)

        score = min(1.0, mean_diff / 80.0)
        logger.debug("scene_change: mean_diff=%.1f -> score=%.4f", mean_diff, score)
        return score

    def _text_detection_score(self, frame: np.ndarray) -> float:
        """Score based on OCR text matching against text_indicators.

        Uses EasyOCR to detect text in the frame (cropped to screen_regions
        when defined). Returns the max OCR confidence for any indicator match.
        """
        if not self.profile.text_indicators:
            self._last_text_boxes = []
            return 0.0

        # Lazy-init PaddleOCR reader
        if not hasattr(self, '_ocr_reader') or self._ocr_reader is None:
            try:
                from paddleocr import PaddleOCR
                self._ocr_reader = PaddleOCR(
                    ocr_version="PP-OCRv4",
                    lang="en",
                    use_angle_cls=False,
                    use_gpu=False,
                    show_log=False,
                )
                logger.info("text_detection: PaddleOCR reader initialized")
            except ImportError:
                logger.warning("text_detection: paddleocr not installed")
                self._last_text_boxes = []
                return 0.0

        h, w = frame.shape[:2]

        # Build list of areas to scan with offsets
        search_areas = []
        if self.profile.screen_regions:
            for region in self.profile.screen_regions:
                area = self._extract_region(frame, region)
                ox = int(region.x_min * w)
                oy = int(region.y_min * h)
                search_areas.append((area, ox, oy))
        else:
            search_areas.append((frame, 0, 0))

        indicators_lower = [t.lower() for t in self.profile.text_indicators]
        best_score = 0.0
        text_boxes = []

        for area, ox, oy in search_areas:
            try:
                raw = self._ocr_reader.ocr(area, cls=False)
                results = raw[0] if raw and raw[0] else []
            except Exception as e:
                logger.warning("text_detection: OCR failed: %s", e)
                continue

            for bbox, (text, confidence) in results:
                text_lower = text.lower()
                matched = any(ind in text_lower for ind in indicators_lower)
                # Offset bbox to frame coordinates
                offset_bbox = [[pt[0] + ox, pt[1] + oy] for pt in bbox]
                text_boxes.append((offset_bbox, text, confidence, matched))
                if matched:
                    best_score = max(best_score, confidence)

        self._last_text_boxes = text_boxes
        logger.debug("text_detection: final score=%.4f", best_score)
        return best_score

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
        frame = self._normalize_frame(frame)
        now = time.time()

        # When screen_regions are defined, skip the expensive full-frame
        # grayscale conversion.  Individual methods convert their own region
        # crops to gray internally.  We only need a small 320x240 grayscale
        # for scene_change_score (which resizes to that size anyway).
        if self.profile.screen_regions:
            frame_gray = None  # lazy — methods convert per-region as needed
            frame_gray_small = cv2.cvtColor(
                cv2.resize(frame, (320, 240)), cv2.COLOR_BGR2GRAY
            )
        else:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_gray_small = frame_gray  # _scene_change_score will resize

        # Check cooldown
        if now - self.last_detection_time < self.cooldown:
            self.previous_frame = frame_gray_small
            return False, 0.0

        # --- Cheap signals first (fast: color mask, brightness mean, fade mask, scene diff) ---
        color_score = self._color_analysis_score(frame)
        brightness_score = self._brightness_score(frame, frame_gray)
        fade_score = self._fade_detection_score(frame)
        scene_change = self._scene_change_score(frame_gray_small)

        # --- Medium cost: template matching (fast with pre-computed cache) ---
        template_score = self._template_match_score(frame, frame_gray)

        # --- Expensive: OCR (~1-2s). Gate behind cheaper signals. ---
        # Only run OCR when cheaper signals strongly suggest a death,
        # unless text is the only configured signal for this profile.
        has_cheap_signals = bool(
            self.templates or self.profile.dominant_colors
            or self.profile.max_brightness is not None
            or self.profile.min_brightness is not None
            or self.profile.fade_to_color is not None
        )
        gate_score = max(template_score, color_score, brightness_score, fade_score)
        if self.profile.text_indicators and (not has_cheap_signals or gate_score > 0.5):
            text_score = self._text_detection_score(frame)
        else:
            text_score = 0.0
            self._last_text_boxes = []

        # Build map of configured signals (only signals the profile has data for)
        configured: dict[str, float] = {}
        if self.templates:
            configured["template"] = template_score
        if self.profile.text_indicators:
            configured["text"] = text_score
        if self.profile.dominant_colors:
            configured["color"] = color_score
        if self.profile.max_brightness is not None or self.profile.min_brightness is not None:
            configured["brightness"] = brightness_score
        if self.profile.fade_to_color is not None:
            configured["fade"] = fade_score
        if self.previous_frame is not None:
            configured["scene_change"] = scene_change

        # Collect weights for configured signals only
        profile_weights = self.profile.weights or {}
        active: dict[str, float] = {}
        for signal in configured:
            active[signal] = profile_weights.get(signal, DEFAULT_WEIGHTS.get(signal, 1.0))

        # Normalize and compute weighted average
        total_weight = sum(active.values())
        if total_weight > 0:
            confidence = sum(
                configured[signal] * (active[signal] / total_weight)
                for signal in active
            )
        else:
            confidence = 0.0

        # Store individual scores for the web GUI
        self.last_scores = {
            "template": round(template_score, 4),
            "text": round(text_score, 4),
            "color": round(color_score, 4),
            "brightness": round(brightness_score, 4),
            "fade": round(fade_score, 4),
            "scene_change": round(scene_change, 4),
            "confidence": round(confidence, 4),
            "configured": list(configured.keys()),
        }

        # Store small grayscale for scene_change_score — saves memory
        self.previous_frame = frame_gray_small

        # Require consecutive frames to reduce false positives
        if confidence >= self.threshold:
            self.death_frame_count += 1
        else:
            self.death_frame_count = 0

        is_death = self.death_frame_count >= self.required_consecutive

        logger.debug(
            "analyze_frame: confidence=%.4f threshold=%.2f "
            "consecutive=%d/%d is_death=%s",
            confidence, self.threshold,
            self.death_frame_count, self.required_consecutive, is_death,
        )

        if is_death:
            self.last_detection_time = now
            self.death_frame_count = 0
            logger.info(
                "DEATH DETECTED! Confidence: %.2f (template=%.2f, text=%.2f, color=%.2f, "
                "brightness=%.2f, fade=%.2f, scene_change=%.2f)",
                confidence,
                template_score,
                text_score,
                color_score,
                brightness_score,
                fade_score,
                scene_change,
            )

        return is_death, confidence

    # ------------------------------------------------------------------
    # Debug visualizations (called only from image-upload endpoint)
    # ------------------------------------------------------------------

    def analyze_frame_debug(
        self, frame: np.ndarray
    ) -> tuple[bool, float, dict[str, np.ndarray]]:
        """Run detection and generate debug visualization images.

        Returns (is_death, confidence, debug_images) where debug_images maps
        tab names to BGR images suitable for JPEG encoding.
        """
        # Normalize once; analyze_frame will see it's already within bounds
        # and skip the redundant resize.
        frame = self._normalize_frame(frame)

        is_death, confidence = self.analyze_frame(frame)

        debug_images: dict[str, np.ndarray] = {}
        debug_images["original"] = frame.copy()

        tmpl_img, edge_img = self._debug_template_match(frame)
        debug_images["template_match"] = tmpl_img
        debug_images["edge_detection"] = edge_img

        debug_images["color_mask"] = self._debug_color_mask(frame)
        debug_images["brightness"] = self._debug_brightness(frame)

        fade_img = self._debug_fade_mask(frame)
        if fade_img is not None:
            debug_images["fade_mask"] = fade_img

        debug_images["text_detection"] = self._debug_text_detection(frame)

        return is_death, confidence, debug_images

    def _debug_template_match(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Visualize template matching: best match box + edge view."""
        vis = frame.copy()
        h, w = frame.shape[:2]

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_edges = cv2.Canny(frame_gray, 50, 150)

        if not self.templates:
            cv2.putText(
                vis, "No templates loaded", (w // 2 - 140, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 200), 2,
            )
            edge_bgr = cv2.cvtColor(frame_edges, cv2.COLOR_GRAY2BGR)
            return vis, edge_bgr

        # Draw search regions
        if self.profile.screen_regions:
            for region in self.profile.screen_regions:
                x1 = int(region.x_min * w)
                y1 = int(region.y_min * h)
                x2 = int(region.x_max * w)
                y2 = int(region.y_max * h)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (200, 200, 0), 2)

        # Find best match across all templates and scales
        best_score = 0.0
        best_loc = (0, 0)
        best_size = (0, 0)
        best_heatmap = None

        for template in self.templates:
            tmpl_h, tmpl_w = template.shape[:2]
            scales = self._compute_scales(tmpl_w, tmpl_h, w, h)

            for scale in scales:
                new_w = int(tmpl_w * scale)
                new_h = int(tmpl_h * scale)
                if new_w >= w or new_h >= h or new_w < 10 or new_h < 10:
                    continue

                scaled = cv2.resize(template, (new_w, new_h))

                # Determine search area
                search_gray = frame_gray
                offset_x, offset_y = 0, 0
                if self.profile.screen_regions:
                    for region in self.profile.screen_regions:
                        area = self._extract_region(frame_gray, region)
                        if area.shape[0] > new_h and area.shape[1] > new_w:
                            search_gray = area
                            offset_x = int(region.x_min * w)
                            offset_y = int(region.y_min * h)
                            break

                result = cv2.matchTemplate(
                    search_gray, scaled, cv2.TM_CCOEFF_NORMED
                )
                _, max_val, _, max_loc = cv2.minMaxLoc(result)

                if max_val > best_score:
                    best_score = max_val
                    best_loc = (max_loc[0] + offset_x, max_loc[1] + offset_y)
                    best_size = (new_w, new_h)
                    best_heatmap = result

        # Draw best match rectangle
        if best_size[0] > 0:
            color = (0, 255, 0) if best_score > 0.75 else (0, 165, 255)
            pt1 = best_loc
            pt2 = (best_loc[0] + best_size[0], best_loc[1] + best_size[1])
            cv2.rectangle(vis, pt1, pt2, color, 3)
            label = f"Match: {best_score:.3f}"
            cv2.putText(
                vis, label, (pt1[0], pt1[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
            )

        # Heatmap thumbnail in top-right corner
        if best_heatmap is not None:
            hm_norm = cv2.normalize(
                best_heatmap, None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8)
            hm_color = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)
            thumb_w, thumb_h = 160, 90
            hm_thumb = cv2.resize(hm_color, (thumb_w, thumb_h))
            vis[8 : 8 + thumb_h, w - thumb_w - 8 : w - 8] = hm_thumb

        edge_bgr = cv2.cvtColor(frame_edges, cv2.COLOR_GRAY2BGR)
        return vis, edge_bgr

    def _debug_color_mask(self, frame: np.ndarray) -> np.ndarray:
        """Visualize dominant color mask overlay.

        When screen_regions are defined, only highlights color matches within
        those regions and draws region outlines on the visualization.
        """
        h, w = frame.shape[:2]

        if not self.profile.dominant_colors:
            vis = frame.copy()
            cv2.putText(
                vis, "No dominant_colors in profile", (w // 2 - 180, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 200), 2,
            )
            return vis

        # Build combined mask restricted to screen regions
        combined_mask = np.zeros((h, w), dtype=np.uint8)

        if self.profile.screen_regions:
            for region in self.profile.screen_regions:
                x1 = int(region.x_min * w)
                y1 = int(region.y_min * h)
                x2 = int(region.x_max * w)
                y2 = int(region.y_max * h)
                crop = frame[y1:y2, x1:x2]
                for color_range in self.profile.dominant_colors:
                    lower = np.array(color_range.lower, dtype=np.uint8)
                    upper = np.array(color_range.upper, dtype=np.uint8)
                    region_mask = cv2.inRange(crop, lower, upper)
                    combined_mask[y1:y2, x1:x2] = cv2.bitwise_or(
                        combined_mask[y1:y2, x1:x2], region_mask
                    )
        else:
            for color_range in self.profile.dominant_colors:
                lower = np.array(color_range.lower, dtype=np.uint8)
                upper = np.array(color_range.upper, dtype=np.uint8)
                mask = cv2.inRange(frame, lower, upper)
                combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Dimmed base at 30%
        vis = (frame * 0.3).astype(np.uint8)
        # Highlight matching pixels in green
        green_tint = np.zeros_like(frame)
        green_tint[:, :, 1] = 180  # green channel
        mask_3ch = cv2.merge([combined_mask, combined_mask, combined_mask])
        vis = np.where(mask_3ch > 0, cv2.addWeighted(frame, 0.6, green_tint, 0.4, 0), vis)

        # Draw region outlines
        if self.profile.screen_regions:
            for region in self.profile.screen_regions:
                x1 = int(region.x_min * w)
                y1 = int(region.y_min * h)
                x2 = int(region.x_max * w)
                y2 = int(region.y_max * h)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (200, 200, 0), 2)

        # Show match percentage (against region area if regions defined)
        if self.profile.screen_regions:
            region_pixels = sum(
                int((r.x_max - r.x_min) * w) * int((r.y_max - r.y_min) * h)
                for r in self.profile.screen_regions
            )
            total_pixels = max(region_pixels, 1)
        else:
            total_pixels = h * w
        match_pct = np.count_nonzero(combined_mask) / total_pixels * 100
        label = f"Color match: {match_pct:.1f}%"
        if self.profile.screen_regions:
            label += " (within regions)"
        cv2.putText(
            vis, label, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
        )
        return vis

    def _debug_brightness(self, frame: np.ndarray) -> np.ndarray:
        """Visualize brightness with threshold lines."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        h, w = vis.shape[:2]

        mean_val = np.mean(gray)

        # Mean brightness line (green)
        mean_y = int((1.0 - mean_val / 255.0) * h)
        cv2.line(vis, (0, mean_y), (w, mean_y), (0, 255, 0), 2)
        cv2.putText(
            vis, f"Mean: {mean_val:.0f}", (10, mean_y - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )

        # max_brightness threshold (yellow)
        if self.profile.max_brightness is not None:
            thr_y = int((1.0 - self.profile.max_brightness / 255.0) * h)
            cv2.line(vis, (0, thr_y), (w, thr_y), (0, 255, 255), 2)
            cv2.putText(
                vis, f"max_brightness: {self.profile.max_brightness}",
                (10, thr_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
            )

        # min_brightness threshold (orange)
        if self.profile.min_brightness is not None:
            thr_y = int((1.0 - self.profile.min_brightness / 255.0) * h)
            cv2.line(vis, (0, thr_y), (w, thr_y), (0, 165, 255), 2)
            cv2.putText(
                vis, f"min_brightness: {self.profile.min_brightness}",
                (10, thr_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2,
            )

        return vis

    def _debug_fade_mask(self, frame: np.ndarray) -> np.ndarray | None:
        """Visualize fade-to-color mask. Returns None if profile has no fade_to_color."""
        if self.profile.fade_to_color is None:
            return None

        h, w = frame.shape[:2]
        lower = np.array(self.profile.fade_to_color.lower, dtype=np.uint8)
        upper = np.array(self.profile.fade_to_color.upper, dtype=np.uint8)
        mask = cv2.inRange(frame, lower, upper)

        total_pixels = h * w
        fade_ratio = np.count_nonzero(mask) / total_pixels

        # Dimmed base at 30%
        vis = (frame * 0.3).astype(np.uint8)
        # Purple tint for matching pixels
        purple_tint = np.zeros_like(frame)
        purple_tint[:, :, 0] = 180  # blue
        purple_tint[:, :, 2] = 140  # red
        mask_3ch = cv2.merge([mask, mask, mask])
        vis = np.where(mask_3ch > 0, cv2.addWeighted(frame, 0.5, purple_tint, 0.5, 0), vis)

        # Show fade ratio and threshold status
        pct = fade_ratio * 100
        exceeds = fade_ratio > 0.85
        label = f"Fade: {pct:.1f}%"
        if exceeds:
            label += " (STRONG FADE)"
        color = (0, 0, 255) if exceeds else (180, 100, 220)
        cv2.putText(
            vis, label, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
        )
        cv2.putText(
            vis, "Threshold: 85%", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 100, 220), 1,
        )
        return vis

    def _debug_text_detection(self, frame: np.ndarray) -> np.ndarray:
        """Visualize OCR text detection with bounding boxes."""
        vis = frame.copy()
        h, w = frame.shape[:2]

        if not self.profile.text_indicators:
            cv2.putText(
                vis, "No text_indicators in profile", (w // 2 - 180, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 200), 2,
            )
            return vis

        if not hasattr(self, '_ocr_reader') or self._ocr_reader is None:
            try:
                from paddleocr import PaddleOCR
                self._ocr_reader = PaddleOCR(
                    ocr_version="PP-OCRv4",
                    lang="en",
                    use_angle_cls=False,
                    use_gpu=False,
                    show_log=False,
                )
            except ImportError:
                cv2.putText(
                    vis, "paddleocr not installed", (w // 2 - 160, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 200), 2,
                )
                return vis

        indicators_lower = [t.lower() for t in self.profile.text_indicators]

        # Build search areas with their offsets
        search_areas = []
        if self.profile.screen_regions:
            for region in self.profile.screen_regions:
                x1 = int(region.x_min * w)
                y1 = int(region.y_min * h)
                x2 = int(region.x_max * w)
                y2 = int(region.y_max * h)
                crop = frame[y1:y2, x1:x2]
                search_areas.append((crop, x1, y1))
                # Draw region outline
                cv2.rectangle(vis, (x1, y1), (x2, y2), (200, 200, 0), 2)
        else:
            search_areas.append((frame, 0, 0))

        for area, offset_x, offset_y in search_areas:
            try:
                raw = self._ocr_reader.ocr(area, cls=False)
                results = raw[0] if raw and raw[0] else []
            except Exception:
                continue

            for bbox, (text, confidence) in results:
                text_lower = text.lower()
                matched = any(ind in text_lower for ind in indicators_lower)

                # Convert bbox points to frame coordinates
                pts = np.array(bbox, dtype=np.int32)
                pts[:, 0] += offset_x
                pts[:, 1] += offset_y

                color = (0, 255, 0) if matched else (128, 128, 128)
                thickness = 2 if matched else 1
                cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=thickness)

                # Label
                label = f"{text} ({confidence:.2f})"
                label_pos = (pts[0][0], pts[0][1] - 8)
                cv2.putText(
                    vis, label, label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                )

        return vis
