# Twitch Death Bot - Setup Guide

Step-by-step instructions to get the Twitch Death Counter Bot running in your channel.

---

## Prerequisites

- **Python 3.10+** installed
- **FFmpeg** installed and on your PATH
- A **Twitch account** for the bot (can be your own account or a separate bot account)

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

Verify it works:
```bash
ffmpeg -version
```

---

## Step 1: Get a Twitch OAuth Token

The bot needs an OAuth token to read and write messages in your Twitch chat.

### Option A: Quick Token (twitchtokengenerator.com)

1. Go to https://twitchtokengenerator.com
2. Log in with the Twitch account you want the bot to use
3. Select **Bot Chat Token** (needs `chat:read` and `chat:edit` scopes)
4. Click **Generate Token**
5. Copy the token — it looks like `oauth:abc123def456...`

### Option B: Twitch Developer Console (more control)

1. Go to https://dev.twitch.tv/console/apps
2. Click **Register Your Application**
3. Fill in:
   - **Name:** `DeathCounterBot` (or anything unique)
   - **OAuth Redirect URLs:** `http://localhost`
   - **Category:** Chat Bot
4. Click **Create**
5. Copy your **Client ID** (you'll need this for auto-game detection)
6. Generate an OAuth token using the Twitch OAuth flow or a tool like https://twitchtokengenerator.com

**Important:** The token must have at minimum:
- `chat:read` — read messages in the channel
- `chat:edit` — send messages to the channel

---

## Step 2: Install Dependencies

```bash
cd Twitch-Death-Bot
pip install -r requirements.txt
```

This installs everything including the computer vision libraries. If you only need the chat bot without death detection (for testing), the key packages are:
- `twitchio` — Twitch chat library
- `python-dotenv` — Environment variable loading

---

## Step 3: Configure the Bot

Copy the example config and edit it:

```bash
cp config.example.env .env
```

Open `.env` in your editor and set these required values:

```env
# REQUIRED: Your OAuth token from Step 1
TWITCH_TOKEN=oauth:your_token_here

# REQUIRED: The channel name to join (your channel, without the #)
TWITCH_CHANNEL=your_channel_name

# Which game to detect deaths for
GAME_PROFILE=elden_ring
```

### Available Game Profiles

| Profile | Game | What it detects |
|---------|------|----------------|
| `elden_ring` | Elden Ring | Dark screen + red "YOU DIED" text |
| `dark_souls_2` | Dark Souls II | Dark screen + "YOU DIED" text |
| `dark_souls_3` | Dark Souls III | Dark screen + red "YOU DIED" text |
| `sekiro` | Sekiro | Dark screen + red kanji death text |
| `hollow_knight` | Hollow Knight | Screen fade to black |
| `celeste` | Celeste | White screen flash on death |
| `generic` | Any game | Dark fade + general death indicators |
| `auto` | Auto-detect | Queries Twitch API (needs Client ID) |

---

## Step 4: Run the Bot

```bash
python main.py
```

You should see output like:
```
2024-01-15 14:30:00 [INFO] death_bot: Game profile: Elden Ring
2024-01-15 14:30:01 [INFO] death_bot: Bot starting... Press Ctrl+C to stop.
2024-01-15 14:30:02 [INFO] bot.twitch_bot: Bot connected as your_bot_name
2024-01-15 14:30:02 [INFO] bot.twitch_bot: Monitoring channel: your_channel
```

The bot will:
1. Connect to your Twitch channel's chat
2. Send a welcome message: "Death counter bot is now online!"
3. Start capturing your stream and detecting deaths (if you're live)
4. Respond to chat commands

If the stream is offline, the bot runs in **chat-only mode** and will start detection when the stream goes live.

Press `Ctrl+C` to stop the bot gracefully.

---

## Step 5: Test It

Once the bot is running, go to your Twitch chat and type:

- `!deaths` — Should respond "No deaths yet this stream! PogChamp"
- `!help` — Lists all available commands
- `!game` — Shows which game is being tracked
- `!adddeath` — (Mod only) Manually adds a death to test the counter

---

## Chat Commands

### Everyone

| Command | Description |
|---------|-------------|
| `!deaths` | Deaths this stream session |
| `!totaldeaths` | All-time death count |
| `!deathstats` | Full stats: session, game total, all-time, sessions played |
| `!game` | Which game is being tracked |
| `!clips` | How many death clips have been recorded |
| `!help` | List all commands |

### Mod / Broadcaster Only

| Command | Description |
|---------|-------------|
| `!reset` | Reset the session death count to 0 |
| `!adddeath` | Manually add a death (for missed detections) |
| `!removedeath` | Remove the last death (undo false positives) |
| `!setgame <profile>` | Switch game profile mid-stream (e.g. `!setgame sekiro`) |
| `!toggleannounce` | Turn automatic death announcements on/off |

---

## Auto-Game Detection (Optional)

If you play multiple games, you can set `GAME_PROFILE=auto` so the bot automatically picks the right detection profile based on what game your Twitch channel is set to.

### Setup

1. Go to https://dev.twitch.tv/console/apps and register an application (or use an existing one)
2. Copy your **Client ID**
3. Add to your `.env`:

```env
TWITCH_CLIENT_ID=your_client_id_here
GAME_PROFILE=auto
```

The bot will query the Twitch Helix API on startup to check what game you're playing and pick the matching profile. If no specific profile matches, it falls back to `generic`.

---

## Running with Docker

### Docker Compose (recommended)

```bash
# Create your .env file first (see Step 3)
docker compose up -d
```

This starts two services:
- **bot** — The death detection bot + chat commands
- **web** — The testing GUI on port 4444

### Docker (single container)

```bash
# Build
docker build -t death-bot .

# Run the bot
docker run -d --env-file .env -v death-data:/app/data -v death-clips:/app/clips death-bot

# Or run the web GUI
docker run -d --env-file .env -e MODE=web -p 4444:4444 -v death-data:/app/data death-bot
```

---

## Troubleshooting

### "TWITCH_TOKEN and TWITCH_CHANNEL must be set"

Your `.env` file is missing or the values are empty. Make sure:
1. You copied `config.example.env` to `.env`
2. You replaced the placeholder values with real credentials
3. The `.env` file is in the project root (same directory as `main.py`)

### "Login authentication failed"

Your OAuth token is invalid or expired. Get a new one from https://twitchtokengenerator.com. Make sure the token starts with `oauth:`.

### "streamlink not found"

Streamlink isn't installed or not on your PATH:
```bash
pip install streamlink
```

### "Failed to start stream capture. Is [channel] live?"

This is normal when the channel is offline. The bot will still work in chat-only mode and start detection when the stream goes live. If the channel IS live, check:
- The channel name is spelled correctly (case-insensitive, no `#` prefix)
- streamlink can reach the stream: `streamlink https://www.twitch.tv/channel_name`

### "ffmpeg failed to decode stream"

FFmpeg isn't installed or can't be found:
```bash
ffmpeg -version
```

### Bot doesn't respond to commands

1. Check that the bot has `chat:read` and `chat:edit` scopes on the OAuth token
2. Make sure `TWITCH_CHANNEL` matches the channel you're chatting in
3. Commands must start with the prefix (default `!`). If you changed `COMMAND_PREFIX`, use that prefix
4. Check the terminal output for error messages

### Too many false positive deaths

- Increase `DETECTION_THRESHOLD` (e.g., `0.90` instead of `0.80`)
- Increase `DEATH_COOLDOWN` (e.g., `30` seconds)
- Use `!removedeath` in chat to undo false positives
- Add more template images for better detection accuracy

### Bot sends too many messages

- Use `!toggleannounce` to turn off automatic death announcements
- Set `ANNOUNCE_DEATHS=false` in `.env` to disable them by default
- Death counting still works — viewers can check with `!deaths`

---

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `TWITCH_TOKEN` | (required) | Bot OAuth token (`oauth:...`) |
| `TWITCH_CHANNEL` | (required) | Channel to join |
| `TWITCH_CLIENT_ID` | (optional) | App Client ID for auto game detection |
| `GAME_PROFILE` | `generic` | Game profile or `auto` |
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
| `CLIP_OUTPUT_DIR` | `clips/` | Clip save directory |
| `WEB_PORT` | `4444` | Web GUI port |
| `WEB_DEBUG` | `false` | Flask debug mode |
