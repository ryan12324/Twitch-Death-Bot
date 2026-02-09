#!/usr/bin/env python3
"""
Utility to create death screen template images from a Twitch VOD or local video.

Usage:
    python tools/create_templates.py --source video.mp4 --game elden_ring
    python tools/create_templates.py --source https://twitch.tv/videos/12345 --game dark_souls_3

This tool helps you build the template image library that powers detection.
It plays through a video and lets you press 'S' to save the current frame
as a death screen reference template.

Controls:
    S     - Save current frame as a template
    SPACE - Pause/resume playback
    Q     - Quit
    LEFT  - Skip back 5 seconds
    RIGHT - Skip forward 5 seconds
"""

import argparse
import os
import sys
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(
        description="Create death screen template images from video"
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to a video file or Twitch VOD URL",
    )
    parser.add_argument(
        "--game",
        required=True,
        help="Game profile name (e.g. elden_ring, dark_souls_3)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: game_profiles/templates/<game>)",
    )
    args = parser.parse_args()

    # Resolve output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = (
            Path(__file__).parent.parent
            / "game_profiles"
            / "templates"
            / args.game
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Count existing templates to set filename numbering
    existing = list(out_dir.glob("death_*.png"))
    next_idx = len(existing) + 1

    source = args.source

    # If it's a Twitch URL, try resolving with streamlink
    if "twitch.tv" in source:
        import subprocess

        result = subprocess.run(
            ["streamlink", "--stream-url", source, "best"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Failed to resolve Twitch URL: {result.stderr}")
            sys.exit(1)
        source = result.stdout.strip()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source: {args.source}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    paused = False

    print("=" * 60)
    print("Death Screen Template Creator")
    print("=" * 60)
    print(f"Game:   {args.game}")
    print(f"Output: {out_dir}")
    print()
    print("Controls:")
    print("  S     - Save current frame as death template")
    print("  SPACE - Pause/resume")
    print("  Q     - Quit")
    print("  LEFT  - Skip back 5s")
    print("  RIGHT - Skip forward 5s")
    print("=" * 60)

    cv2.namedWindow("Template Creator", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Template Creator", 1280, 720)

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video")
                break

        display = frame.copy()

        # Show position info
        pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        pos_s = int(pos_ms / 1000)
        minutes, seconds = divmod(pos_s, 60)
        status = "PAUSED" if paused else "PLAYING"
        cv2.putText(
            display,
            f"[{status}] {minutes:02d}:{seconds:02d} | "
            f"Templates saved: {next_idx - 1} | Press S to save, Q to quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Template Creator", display)

        key = cv2.waitKey(1 if not paused else 50) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("s"):
            # Save frame as template
            filename = f"death_{next_idx:03d}.png"
            filepath = out_dir / filename
            cv2.imwrite(str(filepath), frame)
            print(f"Saved: {filepath}")
            next_idx += 1
        elif key == ord(" "):
            paused = not paused
        elif key == 81:  # LEFT arrow
            new_pos = max(0, pos_ms - 5000)
            cap.set(cv2.CAP_PROP_POS_MSEC, new_pos)
        elif key == 83:  # RIGHT arrow
            new_pos = pos_ms + 5000
            cap.set(cv2.CAP_PROP_POS_MSEC, new_pos)

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone! {next_idx - 1} templates saved to {out_dir}")


if __name__ == "__main__":
    main()
