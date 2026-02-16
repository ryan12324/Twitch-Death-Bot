# Twitch Death Bot - Setup Guide

The Death Counter Bot is a **web application** that monitors Twitch streams, detects death screens using computer vision, and announces deaths in chat. You configure and run detection jobs from the web UI — pick a channel, choose a game profile, and hit Start.

---

## Prerequisites

- **Python 3.10+**
- **FFmpeg** installed and on your PATH
- A **Twitch account** for the bot to send chat messages from

### Install FFmpeg

**Linux (Debian/Ubuntu):**
```bash
sudo apt update && sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
Download from https://ffmpeg.org/download.html and add to your PATH.

Verify: `ffmpeg -version`

---

## Step 1: Get a Twitch OAuth Token

The bot needs an OAuth token to read and send messages in Twitch chat.

1. Go to **https://twitchtokengenerator.com**
2. Log in with the Twitch account the bot should post as
3. Select **Bot Chat Token** (needs `chat:read` and `chat:edit` scopes)
4. Click **Generate Token**
5. Copy the token — it looks like `oauth:abc123def456...`

You'll paste this into the web UI when starting a detection job.

### Optional: Twitch Client ID (for auto game detection)

If you want the "Detect Game" button in the web UI to work (auto-detects what game a channel is playing):

1. Go to **https://dev.twitch.tv/console/apps**
2. Click **Register Your Application**
3. Fill in: Name = `DeathCounterBot`, Redirect URL = `http://localhost`, Category = Chat Bot
4. Click **Create** and copy your **Client ID**

---

## Step 2: Install Dependencies

```bash
cd Twitch-Death-Bot
pip install -r requirements.txt
```

---

## Step 3: Configure Defaults

```bash
cp config.example.env .env
```

Edit `.env` — these values **pre-fill the web UI forms** so you don't have to type them every time:

```env
# Your OAuth token (pre-fills the token field in the web UI)
TWITCH_TOKEN=oauth:your_token_here

# Default channel (pre-fills the channel field)
TWITCH_CHANNEL=your_channel_name

# Optional: Client ID for auto game detection
TWITCH_CLIENT_ID=your_client_id_here
```

The `.env` is for convenience. The actual values used for each detection job are whatever you enter in the web UI when you click Start.

---

## Step 4: Start the Web App

```bash
python web/app.py
```

Open **http://localhost:4444** in your browser.

---

## Step 5: Start a Detection Job

1. Go to **http://localhost:4444/bot**
2. Fill in:
   - **Channel** — the Twitch channel to monitor (the streamer's username)
   - **Game Profile** — which game's death screen to detect (e.g. `elden_ring`, `sekiro`, `generic`)
   - **Twitch Token** — your OAuth token from Step 1
3. Click **Start**

The bot will:
- Connect to the channel's live stream and start capturing frames
- Run death detection against each frame using the selected game profile
- Join the channel's Twitch chat as a bot
- Announce each detected death in chat
- Record short video clips around each death
- Show live detection scores and video in the web UI

### What each page does

| Page | URL | Purpose |
|------|-----|---------|
| **Home** | `/` | Overview dashboard |
| **Bot Control** | `/bot` | Start/stop detection jobs, view live video + scores, manage chat bot |
| **Test Detection** | `/test` | Test detection against a live stream without the chat bot |
| **Profiles** | `/profiles` | Create/edit game profiles, upload template images |

---

## Step 6: Test the Chat Bot

Once a job is running, go to the Twitch channel's chat and type:

| Command | Description |
|---------|-------------|
| `!deaths` | Deaths this stream session |
| `!totaldeaths` | All-time death count |
| `!deathstats` | Full stats: session, game total, all-time, sessions played |
| `!game` | Which game is being tracked |
| `!clips` | How many death clips have been saved |
| `!help` | List all commands |

### Mod / Broadcaster Commands

| Command | Description |
|---------|-------------|
| `!reset` | Reset session death count to 0 |
| `!adddeath` | Manually add a death (for missed detections) |
| `!removedeath` | Remove the last death (undo false positives) |
| `!setgame <profile>` | Switch game profile mid-stream (e.g. `!setgame sekiro`) |
| `!toggleannounce` | Turn automatic death announcements on/off |

---

## Game Profiles

| Profile | Game | What it detects |
|---------|------|----------------|
| `elden_ring` | Elden Ring | Dark screen + red "YOU DIED" text |
| `dark_souls_2` | Dark Souls II | Dark screen + "YOU DIED" text |
| `dark_souls_3` | Dark Souls III | Dark screen + red "YOU DIED" text |
| `sekiro` | Sekiro | Dark screen + red kanji death text |
| `hollow_knight` | Hollow Knight | Screen fade to black |
| `celeste` | Celeste | White screen flash on death |
| `generic` | Any game | Dark fade + general death indicators |

You can create custom profiles from the **/profiles** page in the web UI, or by adding a JSON file to `game_profiles/`.

---

## Running with Docker

### Docker Compose (recommended)

```bash
cp config.example.env .env
# Edit .env with your defaults

docker compose up -d
```

This starts:
- **web** service — the web UI on port 4444 (this is the main thing you interact with)
- **bot** service — optional CLI mode for headless operation

Open **http://localhost:4444/bot** and start a detection job from the UI.

### Single Container

```bash
docker build -t death-bot .

# Run the web UI (primary mode)
docker run -d --env-file .env -e MODE=web -p 4444:4444 \
  -v death-data:/app/data -v death-clips:/app/clips death-bot
```

---

## CLI Mode (Alternative)

If you prefer running without the web UI, `main.py` provides a headless CLI mode that reads everything from `.env`:

```bash
# Set TWITCH_TOKEN, TWITCH_CHANNEL, GAME_PROFILE in .env, then:
python main.py
```

This connects directly to one channel with one profile and runs until you Ctrl+C. The web UI is the recommended way to run the bot since it gives you live video, score visualization, and the ability to start/stop/reconfigure without restarting.

---

## Troubleshooting

### "Login authentication failed"
Your OAuth token is invalid or expired. Get a new one from https://twitchtokengenerator.com. The token must start with `oauth:`.

### "streamlink not found"
```bash
pip install streamlink
```

### "Could not connect to channel. Is the stream live?"
The channel must be **live** for stream capture to work. If the stream is offline, the bot can still join chat — detection starts when the stream goes live.

### Bot doesn't respond to commands
- Token needs `chat:read` and `chat:edit` scopes
- Make sure the channel name matches where you're chatting
- Check the web UI terminal output for errors

### Too many false positives
- Increase the **threshold** slider in the web UI (try 0.85-0.90)
- Use `!removedeath` in chat to undo false detections
- Upload more template images on the **/profiles** page for better accuracy

### Bot sends too many messages
- Toggle announcements off from the web UI or with `!toggleannounce`
- Set `ANNOUNCE_DEATHS=false` in `.env` to disable by default

---

## Configuration Reference

All values can be set in `.env` and serve as defaults for the web UI.

| Variable | Default | Description |
|----------|---------|-------------|
| `TWITCH_TOKEN` | — | Bot OAuth token (pre-fills web UI) |
| `TWITCH_CHANNEL` | — | Default channel (pre-fills web UI) |
| `TWITCH_CLIENT_ID` | — | App Client ID for auto game detection |
| `GAME_PROFILE` | `generic` | Default game profile |
| `TARGET_FPS` | `15` | Frame capture rate (5-30) |
| `DETECTION_THRESHOLD` | `0.80` | Confidence threshold (0.0-1.0) |
| `DEATH_COOLDOWN` | `15` | Seconds between detections |
| `STREAM_QUALITY` | `720p` | Stream capture quality |
| `COMMAND_PREFIX` | `!` | Chat command prefix |
| `ANNOUNCE_DEATHS` | `true` | Auto-announce deaths in chat |
| `CLIP_ENABLED` | `true` | Record death clips |
| `CLIP_PRE_DEATH_SECONDS` | `3.0` | Pre-death clip footage |
| `CLIP_POST_DEATH_SECONDS` | `2.0` | Post-death clip footage |
| `CLIP_OUTPUT_FPS` | `10.0` | Clip playback FPS |
| `WEB_PORT` | `4444` | Web UI port |
| `WEB_DEBUG` | `false` | Flask debug mode |
