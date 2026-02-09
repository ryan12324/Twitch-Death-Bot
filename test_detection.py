#!/usr/bin/env python3
"""Quick test: run the detector against the DS2 death screen image."""

import sys
sys.path.insert(0, "/home/user/Twitch-Death-Bot")

import cv2
from detection.detector import DeathDetector
from game_profiles.profiles import get_profile

TEMPLATE = "/home/user/Twitch-Death-Bot/game_profiles/templates/dark_souls_2/dark-souls-death.png"

# Load the image as if it were a captured frame (resize to 1280x720)
frame = cv2.imread(TEMPLATE)
print(f"Original image: {frame.shape[1]}x{frame.shape[0]}")

frame_720 = cv2.resize(frame, (1280, 720))
print(f"Resized to stream res: {frame_720.shape[1]}x{frame_720.shape[0]}")

profile = get_profile("dark_souls_2")
detector = DeathDetector(profile=profile, threshold=0.80, cooldown=0, required_consecutive=1)

print(f"\nTemplates loaded: {len(detector.templates)}")
for i, t in enumerate(detector.templates):
    print(f"  [{i}] {t.shape[1]}x{t.shape[0]}")

# Single call through analyze_frame — same code path as production
print("\n--- Analyzing frame ---")
is_death, confidence = detector.analyze_frame(frame_720)

for k, v in detector.last_scores.items():
    print(f"  {k:15s}: {v:.4f}")

print(f"\n  threshold:       {detector.threshold}")
print(f"  DETECTED:        {is_death}")
