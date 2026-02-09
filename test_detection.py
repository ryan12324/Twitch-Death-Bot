#!/usr/bin/env python3
"""Quick test: run the detector against the DS2 death screen image."""

import sys
sys.path.insert(0, "/home/user/Twitch-Death-Bot")

import cv2
import numpy as np
from detection.detector import DeathDetector
from game_profiles.profiles import get_profile

TEMPLATE = "/home/user/Twitch-Death-Bot/game_profiles/templates/dark_souls_2/dark-souls-death.png"

# Load the image as if it were a captured frame (resize to 1280x720)
frame = cv2.imread(TEMPLATE)
print(f"Original image: {frame.shape[1]}x{frame.shape[0]}")

frame_720 = cv2.resize(frame, (1280, 720))
print(f"Resized to stream res: {frame_720.shape[1]}x{frame_720.shape[0]}")

profile = get_profile("dark_souls_2")
detector = DeathDetector(profile=profile, threshold=0.80, cooldown=0)

print(f"\nTemplates loaded: {len(detector.templates)}")
for i, t in enumerate(detector.templates):
    print(f"  [{i}] {t.shape[1]}x{t.shape[0]}")

# Analyze the frame
print("\n--- Analyzing frame ---")
scores = {}
scores["template"] = detector._template_match_score(frame_720)
scores["color"] = detector._color_analysis_score(frame_720)
scores["brightness"] = detector._brightness_score(frame_720)
scores["fade"] = detector._fade_detection_score(frame_720)
scores["scene_change"] = detector._scene_change_score(frame_720)

for k, v in scores.items():
    print(f"  {k:15s}: {v:.4f}")

# Run full analyze (need 2 consecutive)
detector.required_consecutive = 1  # override for single-frame test
is_death, confidence = detector.analyze_frame(frame_720)
print(f"\n  confidence:      {confidence:.4f}")
print(f"  threshold:       {detector.threshold}")
print(f"  DETECTED:        {is_death}")
