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

from twitchio.ext import commands

from detection.clip_recorder import ClipRecorder
from detection.counter import DeathCounter

logger = logging.getLogger(__name__)


class DeathBot(commands.Bot):
    def __init__(
        self,
        token: str,
        prefix: str,
        channel: str,
        counter: DeathCounter,
        game: str,
        clip_recorder: ClipRecorder | None = None,
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

    async def event_ready(self) -> None:
        logger.info("Bot connected as %s", self.nick)
        logger.info("Monitoring channel: %s", self.channel_name)

    async def event_message(self, message) -> None:
        if message.echo:
            return
        await self.handle_commands(message)

    @commands.command(name="deaths")
    async def cmd_deaths(self, ctx: commands.Context) -> None:
        """Show the death count for this stream session."""
        count = self.counter.session_deaths
        if count == 0:
            await ctx.send("No deaths yet this stream! PogChamp")
        elif count == 1:
            await ctx.send(f"1 death this stream.")
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

    async def announce_death(self, session_count: int, total_count: int) -> None:
        """Send a death announcement to chat."""
        channel = self.get_channel(self.channel_name)
        if channel:
            await channel.send(
                f"DEATH #{session_count} this stream! "
                f"({total_count} all-time)"
            )
