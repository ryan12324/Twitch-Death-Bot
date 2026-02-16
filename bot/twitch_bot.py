"""
Twitch chat bot for death counter commands.

Provides chat commands:
  !deaths      - Show deaths this stream
  !totaldeaths - Show all-time death count
  !deathstats  - Full stats breakdown
  !game        - Show current game being tracked
  !clips       - Show how many death clips have been saved
  !help        - List available commands
  !reset       - (Mod) Reset session death count
  !setgame     - (Mod) Change game profile mid-stream
  !adddeath    - (Mod) Manually add a death
  !removedeath - (Mod) Remove the last death
"""

import logging
import random

from twitchio.ext import commands

from detection.clip_recorder import ClipRecorder
from detection.counter import DeathCounter
from game_profiles.profiles import get_profile, list_profiles

logger = logging.getLogger(__name__)

# Rotating death messages to keep chat interesting
DEATH_MESSAGES = [
    "DEATH #{session} this stream! ({total} all-time)",
    "RIP! That's death #{session}! ({total} total deaths)",
    "Another one bites the dust... death #{session} ({total} all-time)",
    "F in chat. Death #{session} this stream. ({total} total)",
    "Death #{session}! The counter keeps climbing... ({total} all-time)",
    "And they're down again! Death #{session} ({total} total)",
    "Death #{session}... they just keep coming. ({total} all-time)",
    "That's #{session} for today! ({total} deaths all-time)",
]

# Milestone messages for big death counts
MILESTONE_MESSAGES = {
    10: "10 deaths this stream! Double digits!",
    25: "25 deaths! A quarter century of pain!",
    50: "50 deaths this stream! Halfway to 100!",
    69: "69 deaths... nice.",
    100: "100 DEATHS THIS STREAM! Triple digits!",
    150: "150 deaths... is this a record?",
    200: "200 DEATHS! Unbelievable!",
}


def _is_mod_or_broadcaster(ctx: commands.Context) -> bool:
    """Check if the user is a moderator or the broadcaster."""
    return ctx.author.is_mod or ctx.author.is_broadcaster


class DeathBot(commands.Bot):
    def __init__(
        self,
        token: str,
        prefix: str,
        channel: str,
        counter: DeathCounter,
        game: str,
        clip_recorder: ClipRecorder | None = None,
        announce_enabled: bool = True,
    ):
        super().__init__(
            token=token,
            prefix=prefix,
            initial_channels=[channel],
        )
        self.counter = counter
        self.game = game
        self.channel_name = channel
        self.clip_recorder = clip_recorder
        self.announce_enabled = announce_enabled
        self._prefix = prefix

    async def event_ready(self) -> None:
        logger.info("Bot connected as %s", self.nick)
        logger.info("Monitoring channel: %s", self.channel_name)
        channel = self.get_channel(self.channel_name)
        if channel:
            await channel.send(
                f"Death counter bot is now online! "
                f"Tracking: {self.game}. "
                f"Type {self._prefix}deaths to check the count."
            )

    async def event_message(self, message) -> None:
        if message.echo:
            return
        await self.handle_commands(message)

    # ------------------------------------------------------------------
    # Public commands (anyone can use)
    # ------------------------------------------------------------------

    @commands.command(name="deaths")
    async def cmd_deaths(self, ctx: commands.Context) -> None:
        """Show the death count for this stream session."""
        count = self.counter.session_deaths
        if count == 0:
            await ctx.send("No deaths yet this stream! PogChamp")
        elif count == 1:
            await ctx.send("1 death this stream.")
        else:
            await ctx.send(f"{count} deaths this stream.")

    @commands.command(name="totaldeaths")
    async def cmd_total_deaths(self, ctx: commands.Context) -> None:
        """Show the all-time death count."""
        total = self.counter.total_deaths
        await ctx.send(f"All-time deaths: {total}")

    @commands.command(name="deathstats")
    async def cmd_death_stats(self, ctx: commands.Context) -> None:
        """Show full death statistics."""
        stats = self.counter.get_stats(self.game)
        await ctx.send(
            f"[{stats['game']}] "
            f"This stream: {stats['session_deaths']} | "
            f"This game total: {stats['game_deaths']} | "
            f"All-time: {stats['total_deaths']} | "
            f"Sessions: {stats['sessions_played']}"
        )

    @commands.command(name="game")
    async def cmd_game(self, ctx: commands.Context) -> None:
        """Show the current game being tracked."""
        await ctx.send(f"Currently tracking deaths for: {self.game}")

    @commands.command(name="clips")
    async def cmd_clips(self, ctx: commands.Context) -> None:
        """Show how many death clips have been recorded."""
        if self.clip_recorder and self.clip_recorder.enabled:
            count = self.clip_recorder.get_clip_count()
            await ctx.send(
                f"{count} death clip(s) saved this session for the compilation!"
            )
        else:
            await ctx.send("Clip recording is not enabled.")

    @commands.command(name="help", aliases=["commands"])
    async def cmd_help(self, ctx: commands.Context) -> None:
        """List available bot commands."""
        p = self._prefix
        msg = (
            f"Commands: "
            f"{p}deaths - Session deaths | "
            f"{p}totaldeaths - All-time | "
            f"{p}deathstats - Full stats | "
            f"{p}game - Current game | "
            f"{p}clips - Clip count"
        )
        if _is_mod_or_broadcaster(ctx):
            msg += (
                f" | Mod: {p}reset, {p}adddeath, {p}removedeath, {p}setgame"
            )
        await ctx.send(msg)

    # ------------------------------------------------------------------
    # Mod / broadcaster commands
    # ------------------------------------------------------------------

    @commands.command(name="reset")
    async def cmd_reset(self, ctx: commands.Context) -> None:
        """(Mod) Reset the session death count to zero."""
        if not _is_mod_or_broadcaster(ctx):
            await ctx.send("Only mods and the broadcaster can reset the counter.")
            return

        old_count = self.counter.session_deaths
        self.counter.session_deaths = 0
        if self.counter.data.get("current_session"):
            self.counter.data["current_session"]["deaths"] = 0
        self.counter._save()

        logger.info(
            "Session deaths reset by %s (was %d)", ctx.author.name, old_count
        )
        await ctx.send(f"Session death count reset to 0 (was {old_count}).")

    @commands.command(name="adddeath")
    async def cmd_add_death(self, ctx: commands.Context) -> None:
        """(Mod) Manually add a death to the counter."""
        if not _is_mod_or_broadcaster(ctx):
            await ctx.send("Only mods and the broadcaster can add deaths.")
            return

        session_count = self.counter.record_death(self.game, confidence=1.0)
        total_count = self.counter.total_deaths
        logger.info("Manual death added by %s", ctx.author.name)
        await ctx.send(
            f"Death added manually! "
            f"Session: {session_count} | Total: {total_count}"
        )

    @commands.command(name="removedeath")
    async def cmd_remove_death(self, ctx: commands.Context) -> None:
        """(Mod) Remove the last death (undo a false positive)."""
        if not _is_mod_or_broadcaster(ctx):
            await ctx.send("Only mods and the broadcaster can remove deaths.")
            return

        if self.counter.session_deaths <= 0:
            await ctx.send("No deaths to remove.")
            return

        self.counter.session_deaths -= 1
        self.counter.data["total_deaths"] = max(
            0, self.counter.data.get("total_deaths", 1) - 1
        )
        game_deaths = self.counter.data.get("games", {}).get(self.game, 1)
        self.counter.data.setdefault("games", {})[self.game] = max(
            0, game_deaths - 1
        )
        if self.counter.data.get("current_session"):
            self.counter.data["current_session"]["deaths"] = self.counter.session_deaths
        self.counter._save()

        logger.info("Death removed by %s", ctx.author.name)
        await ctx.send(
            f"Last death removed. "
            f"Session: {self.counter.session_deaths} | "
            f"Total: {self.counter.total_deaths}"
        )

    @commands.command(name="setgame")
    async def cmd_set_game(self, ctx: commands.Context) -> None:
        """(Mod) Change the game profile. Usage: !setgame elden_ring"""
        if not _is_mod_or_broadcaster(ctx):
            await ctx.send("Only mods and the broadcaster can change the game.")
            return

        parts = ctx.message.content.split(maxsplit=1)
        if len(parts) < 2:
            profiles = ", ".join(list_profiles())
            await ctx.send(f"Usage: {self._prefix}setgame <profile>. Available: {profiles}")
            return

        new_game = parts[1].strip().lower()
        available = list_profiles()
        if new_game not in available:
            await ctx.send(
                f"Unknown profile '{new_game}'. "
                f"Available: {', '.join(available)}"
            )
            return

        profile = get_profile(new_game)
        self.game = profile.display_name
        logger.info("Game changed to %s by %s", self.game, ctx.author.name)
        await ctx.send(f"Now tracking deaths for: {self.game}")

    @commands.command(name="toggleannounce")
    async def cmd_toggle_announce(self, ctx: commands.Context) -> None:
        """(Mod) Toggle automatic death announcements on/off."""
        if not _is_mod_or_broadcaster(ctx):
            await ctx.send("Only mods and the broadcaster can toggle announcements.")
            return

        self.announce_enabled = not self.announce_enabled
        state = "ON" if self.announce_enabled else "OFF"
        logger.info(
            "Death announcements toggled %s by %s", state, ctx.author.name
        )
        await ctx.send(f"Automatic death announcements: {state}")

    # ------------------------------------------------------------------
    # Death announcements (called from detection thread)
    # ------------------------------------------------------------------

    async def announce_death(self, session_count: int, total_count: int) -> None:
        """Send a death announcement to chat with varied messages."""
        if not self.announce_enabled:
            return

        channel = self.get_channel(self.channel_name)
        if not channel:
            return

        # Check for milestone messages first
        milestone_msg = MILESTONE_MESSAGES.get(session_count)
        if milestone_msg:
            await channel.send(milestone_msg)
            return

        # Pick a random death message
        template = random.choice(DEATH_MESSAGES)
        msg = template.format(session=session_count, total=total_count)
        await channel.send(msg)
