"""
Twitch chat bot for death counter commands.

Provides chat commands:
  !deaths      - Show deaths this stream
  !totaldeaths - Show all-time death count
  !deathstats  - Full stats breakdown
  !game        - Show current game being tracked
  !clips       - Show how many death clips have been saved
"""

import logging

import twitchio
from twitchio import eventsub
from twitchio.ext import commands

from detection.clip_recorder import ClipRecorder
from detection.counter import DeathCounter

logger = logging.getLogger(__name__)


class DeathBotCommands(commands.Component):
    """Chat commands for the death counter bot."""

    def __init__(self, bot: "DeathBot") -> None:
        self.bot: DeathBot = bot

    @commands.command(name="deaths")
    async def cmd_deaths(self, ctx: commands.Context) -> None:
        """Show the death count for this stream session."""
        count = self.bot.counter.session_deaths
        if count == 0:
            await ctx.send("No deaths yet this stream! PogChamp")
        elif count == 1:
            await ctx.send("1 death this stream.")
        else:
            await ctx.send(f"{count} deaths this stream.")

    @commands.command(name="totaldeaths")
    async def cmd_total_deaths(self, ctx: commands.Context) -> None:
        """Show the all-time death count."""
        total = self.bot.counter.total_deaths
        await ctx.send(f"All-time deaths: {total}")

    @commands.command(name="deathstats")
    async def cmd_death_stats(self, ctx: commands.Context) -> None:
        """Show full death statistics."""
        stats = self.bot.counter.get_stats(self.bot.game)
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
        await ctx.send(f"Currently tracking deaths for: {self.bot.game}")

    @commands.command(name="clips")
    async def cmd_clips(self, ctx: commands.Context) -> None:
        """Show how many death clips have been recorded."""
        if self.bot.clip_recorder and self.bot.clip_recorder.enabled:
            count = self.bot.clip_recorder.get_clip_count()
            await ctx.send(
                f"{count} death clip(s) saved this session for the compilation!"
            )
        else:
            await ctx.send("Clip recording is not enabled.")


class DeathBot(commands.Bot):
    def __init__(
        self,
        *,
        client_id: str,
        client_secret: str,
        bot_id: str,
        prefix: str,
        channel: str,
        counter: DeathCounter,
        game: str,
        user_token: str = "",
        refresh_token: str = "",
        clip_recorder: ClipRecorder | None = None,
    ):
        super().__init__(
            client_id=client_id,
            client_secret=client_secret,
            bot_id=bot_id,
            prefix=prefix,
        )
        self.counter = counter
        self.game = game
        self.channel_name = channel
        self._channel_id: str = ""
        self.clip_recorder = clip_recorder
        self._user_token = user_token
        self._refresh_token = refresh_token

    async def load_tokens(self, path: str | None = None) -> None:
        """Load pre-configured user token if available."""
        if self._user_token:
            await self.add_token(self._user_token, self._refresh_token)

    async def setup_hook(self) -> None:
        # Look up channel's numeric user ID from login name
        users = await self.fetch_users(logins=[self.channel_name])
        if not users:
            logger.error("Could not find Twitch user: %s", self.channel_name)
            return
        self._channel_id = str(users[0].id)
        logger.info(
            "Resolved channel %s -> id %s", self.channel_name, self._channel_id
        )

        # Subscribe to chat messages for the target channel
        sub = eventsub.ChatMessageSubscription(
            broadcaster_user_id=self._channel_id,
            user_id=self.bot_id,
        )
        await self.subscribe_websocket(payload=sub)

        # Register command component
        await self.add_component(DeathBotCommands(self))

    async def event_ready(self) -> None:
        logger.info("Bot connected (bot_id=%s)", self.bot_id)
        logger.info("Monitoring channel: %s", self.channel_name)

    async def announce_death(self, session_count: int, total_count: int) -> None:
        """Send a death announcement to chat."""
        if not self._channel_id:
            logger.warning("Cannot announce death: channel ID not resolved")
            return
        broadcaster = self.create_partialuser(self._channel_id)
        await broadcaster.send_message(
            f"DEATH #{session_count} this stream! ({total_count} all-time)",
            sender=self.bot_id,
        )
