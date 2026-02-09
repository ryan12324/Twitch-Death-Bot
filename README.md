# Twitch Death Counter Bot

A Python bot that watches a Twitch stream, detects death screens from popular games using image analysis, and tracks death counts in chat.

## Features

- **Image-based death detection** using OpenCV template matching, color analysis, brightness detection, and scene change detection
- **Per-session and all-time tracking** with persistent JSON storage
- **Twitch chat integration** with commands for viewers to check death counts
- **Pre-built game profiles** for Elden Ring, Dark Souls II, Dark Souls III, Sekiro, Hollow Knight, and Celeste
- **Generic profile** for any game with dark/fade-to-black death screens
- **Death clip recording** saves a short video clip around every death for easy compilation
- **Web testing GUI** to test detection against live streams with real-time score visualization
- **Docker + Coolify ready** for one-click self-hosted deployment
- **Template creator tool** to build reference images from VODs or local video

## Requirements

- Python 3.10+
- [FFmpeg](https://ffmpeg.org/) installed and on your PATH
- [Streamlink](https://streamlink.github.io/) (installed via pip)
- A Twitch bot account with an OAuth token

## Setup

1. **Clone and install dependencies:**

```bash
git clone https://github.com/your-repo/Twitch-Death-Bot.git
cd Twitch-Death-Bot
pip install -r requirements.txt
```

2. **Configure the bot:**

```bash
cp config.example.env .env
```

Edit `.env` with your Twitch credentials and settings:

```
TWITCH_TOKEN=oauth:your_token_here
TWITCH_CHANNEL=target_channel_name
GAME_PROFILE=elden_ring
```

Get a Twitch OAuth token at https://twitchtokengenerator.com

3. **Create death screen templates (recommended):**

For best detection accuracy, capture reference screenshots of the death screen from the game you're tracking. Use the built-in tool:

```bash
# From a local video file
python tools/create_templates.py --source gameplay.mp4 --game elden_ring

# From a Twitch VOD
python tools/create_templates.py --source https://twitch.tv/videos/12345 --game dark_souls_3
```

Press `S` when you see a death screen to save it as a template. The more templates you provide, the better the detection.

Templates are saved to `game_profiles/templates/<game>/`.

4. **Run the bot:**

```bash
python main.py
```

## Chat Commands

| Command | Description |
|---------|-------------|
| `!deaths` | Deaths this stream session |
| `!totaldeaths` | All-time death count |
| `!deathstats` | Full statistics breakdown |
| `!game` | Show which game is being tracked |
| `!clips` | Show how many death clips have been saved |

## Supported Games

| Profile | Game | Detection Method |
|---------|------|-----------------|
| `elden_ring` | Elden Ring | Dark screen + red "YOU DIED" text |
| `dark_souls_2` | Dark Souls II | Dark screen + "YOU DIED" text |
| `dark_souls_3` | Dark Souls III | Dark screen + red "YOU DIED" text |
| `sekiro` | Sekiro: Shadows Die Twice | Dark screen + red kanji |
| `hollow_knight` | Hollow Knight | Screen fade to black |
| `celeste` | Celeste | White screen flash |
| `generic` | Any game | Dark fade + red tint |

## How Detection Works

The bot combines multiple detection strategies with weighted scoring:

1. **Template Matching** (40% weight when templates exist): Compares the current frame against saved reference death screen images using structural similarity (SSIM) and normalized cross-correlation.

2. **Color Analysis** (15-30%): Checks what percentage of pixels fall within the game's known death screen color ranges (e.g., dark background + red text for Souls games).

3. **Brightness Analysis** (15-25%): Measures overall frame brightness. Death screens in most games are significantly darker or brighter than normal gameplay.

4. **Fade Detection** (15-25%): Checks if the screen has faded to the death color (usually black).

5. **Scene Change Detection** (15-20%): Detects sudden changes between frames that may indicate a death transition.

A death is only confirmed after **2 consecutive frames** exceed the confidence threshold, reducing false positives.

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `TWITCH_TOKEN` | (required) | Bot OAuth token |
| `TWITCH_CHANNEL` | (required) | Channel to monitor |
| `GAME_PROFILE` | `generic` | Game profile for detection |
| `CAPTURE_INTERVAL` | `2.0` | Seconds between frame captures |
| `DETECTION_THRESHOLD` | `0.80` | Confidence threshold (0.0 - 1.0) |
| `DEATH_COOLDOWN` | `15` | Seconds between death detections |
| `STREAM_QUALITY` | `720p` | Stream quality for capture |
| `CLIP_ENABLED` | `true` | Save a video clip around each death |
| `CLIP_PRE_DEATH_SECONDS` | `3.0` | Seconds of footage before the death |
| `CLIP_POST_DEATH_SECONDS` | `2.0` | Seconds of footage after the death |
| `CLIP_OUTPUT_FPS` | `10.0` | Playback FPS for saved clips |
| `CLIP_OUTPUT_DIR` | `clips/` | Directory for saved death clips |
| `WEB_PORT` | `4444` | Port for the web testing GUI |
| `WEB_DEBUG` | `false` | Flask debug mode (auto-reload) |

## Adding a New Game

1. Add a `GameProfile` in `game_profiles/profiles.py` with the game's death screen characteristics
2. Register it in the `PROFILES` dict
3. Create a template directory: `mkdir game_profiles/templates/your_game`
4. Use the template creator to capture reference death screens
5. Set `GAME_PROFILE=your_game` in `.env`

## Project Structure

```
Twitch-Death-Bot/
├── main.py                          # Entry point
├── requirements.txt
├── config.example.env
├── bot/
│   └── twitch_bot.py                # Twitch chat bot and commands
├── detection/
│   ├── detector.py                  # Death screen detection engine
│   ├── stream_capture.py            # Twitch stream frame capture
│   ├── clip_recorder.py             # Death clip recorder for compilations
│   └── counter.py                   # Death counter with persistence
├── game_profiles/
│   ├── profiles.py                  # Game-specific detection profiles
│   └── templates/                   # Reference death screen images
│       ├── elden_ring/
│       ├── dark_souls_2/
│       ├── dark_souls_3/
│       ├── sekiro/
│       ├── hollow_knight/
│       └── celeste/
├── web/
│   ├── app.py                       # Flask web testing GUI
│   └── templates/index.html         # Dashboard frontend
├── tools/
│   └── create_templates.py          # Template image creator utility
├── Dockerfile                       # Docker build for Coolify
├── docker-compose.yml               # Bot + web GUI services
├── coolify.json                     # Coolify service template
├── clips/                           # Saved death clips (gitignored)
└── data/
    └── deaths.json                  # Persistent death count storage
```

## Death Compilation

When `CLIP_ENABLED=true` (the default), the bot keeps a rolling buffer of recent frames. On every death, it saves a short `.mp4` clip to the `clips/` directory containing:

- **3 seconds before** the death (configurable via `CLIP_PRE_DEATH_SECONDS`)
- **2 seconds after** the death (configurable via `CLIP_POST_DEATH_SECONDS`)

Each clip has a timestamp overlay showing time relative to the death (`-2.0s`, `-1.0s`, `+0.0s`, `+1.0s`, etc.).

To combine all clips into a single compilation video using ffmpeg:

```bash
# Create a file list
ls clips/death_*.mp4 | sort | sed 's/^/file /' > clips/list.txt

# Concatenate into one video
ffmpeg -f concat -safe 0 -i clips/list.txt -c copy death_compilation.mp4
```

Set `CLIP_ENABLED=false` in `.env` to disable clip recording and save disk space.

## Testing GUI

A browser-based dashboard for testing detection against any live stream. Lets you tune settings in real time without touching the bot's Twitch chat connection.

### Run locally

```bash
python web/app.py
```

Open `http://localhost:4444`. From the dashboard you can:

- **Connect to any Twitch stream** by channel name
- **Watch live detection** with per-strategy score bars overlaid on the video
- **Adjust the confidence threshold** with a slider and see the effect immediately
- **Upload a screenshot** to test detection on a single image
- **View the death log** with timestamps and confidence values

### Run with Docker

```bash
docker compose up web
```

The GUI is available at `http://localhost:4444`.

## Deploying to Coolify

This project is ready for [Coolify](https://coolify.io/) self-hosted deployment.

### Option 1: Docker Compose (recommended)

1. In Coolify, create a new service and select **Docker Compose**
2. Point it at this repository
3. Add your `.env` variables in Coolify's environment settings:
   - `TWITCH_TOKEN`
   - `TWITCH_CHANNEL`
   - `GAME_PROFILE`
   - Any other overrides from the config reference
4. Deploy — Coolify will build and start both the `bot` and `web` services
5. Map a domain to the `web` service (port 4444) for the testing GUI

### Option 2: Single service (bot only or web only)

1. In Coolify, create a new service and select **Dockerfile**
2. Point it at this repo
3. Set the `MODE` environment variable:
   - `MODE=bot` — runs the Twitch chat bot with detection (default)
   - `MODE=web` — runs only the web testing GUI
4. For the web GUI, expose port `4444`
5. Add the rest of your `.env` variables

### Persistent storage

Map Docker volumes so death data and clips survive redeployments:

- `/app/data` — death counter JSON (session history, all-time stats)
- `/app/clips` — saved death clip videos
- `/app/game_profiles/templates` — your reference template images
