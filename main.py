"""
Chat Summarizer Bot (AuctionWorld)

What this file does:
1. Registers slash commands for Discord.
2. Collects messages from the current channel in day-sized chunks.
3. Sends one async OpenRouter summary request per day.
4. Combines all summaries into one summary.txt and appends total price at the end.

Mini tutorial: how to add a new slash command
1. Create small helper functions first (validation, business logic, output builder).
2. Add a command function under "Command Handlers" section.
3. Keep command functions thin and call helpers.

Example:
@bot.tree.command(name='ping', description='Health check')
async def ping(interaction: discord.Interaction):
    await interaction.response.send_message('pong')
"""

import asyncio
import io
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import discord
import requests
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv


# ============================================================================
# Configuration
# ============================================================================

load_dotenv()

TOKEN = os.getenv('DISCORD_TOKEN')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_SITE_URL = os.getenv('OPENROUTER_SITE_URL', '')
OPENROUTER_SITE_NAME = os.getenv('OPENROUTER_SITE_NAME', '')
DEFAULT_MODEL = 'openai/gpt-5.5'

GUILD_ID_RAW = os.getenv('DISCORD_GUILD_ID', '').strip()
TARGET_GUILD = discord.Object(id=int(GUILD_ID_RAW)) if GUILD_ID_RAW.isdigit() else None
PERMISSIONS_FILE = 'command_permissions.json'

PERIOD_TO_DAYS = {
    '1d': 1,
    '3d': 3,
    '7d': 7,
    '30d': 30,
}

MODEL_BY_QUALITY = {
    'low': 'openai/gpt-5-nano',
    'medium': 'openai/gpt-4.1-mini',
    'high': 'openai/gpt-5.5',
}

MODEL_PRICING_PER_MILLION = {
    'openai/gpt-5-nano': {'input': 0.05, 'output': 0.40},
    'openai/gpt-4.1-mini': {'input': 0.40, 'output': 1.60},
    'openai/gpt-5.5': {'input': 2.00, 'output': 8.00},
}

# Token-saving guardrail for per-day logs.
MAX_DAY_LOG_CHARS = 18000


SYSTEM_PROMPT = """You are a concise analysis bot for AuctionWorld chat logs.

Your task is to extract only the most important and meaningful information from chat logs that contribute to understanding the value, benefits, improvements, feedback, or issues related to AuctionWorld.

Input Parameters:
- startDate - beginning of the time range
- endDate - end of the time range

The channel is automatically inferred from where the command is executed.

Scope Rules:
- Only analyze messages from the current channel
- Only include messages within the given date range
- Ignore all messages outside this range
- Ignore irrelevant, casual, or off-topic conversation
- Focus ONLY on content related to AuctionWorld system value, features, feedback, issues, improvements, or decisions

Summarization Rules:
- Keep only high-signal content: bugs, features, improvements, user feedback, business impact, decisions.
- Remove small talk, repetition, jokes, and off-topic chatter.
- Compress duplicates into one point.
- Do not infer, assume, or fill gaps.
- Keep dates, names, numbers, quantities, and decisions exactly as written in logs.
- If a detail is uncertain or contradictory, state that clearly instead of resolving it.
- Do not add owners, dates, or outcomes unless explicitly present in the logs.

Output Format (must follow exactly):
1. Overview
- 2-4 bullets of key outcomes.

2. Topic Breakdown
Use separate topic blocks. Do not merge unrelated topics.

Topic: <name>
- Summary:
    - <1-2 bullets>
- What Happened:
    - <bullets>
- Action Items:
    - <bullets or "Insufficient context in this section">
- Decisions / Risks / Blockers:
    - <bullets or "Insufficient context in this section">

3. Cross-Topic Priorities
- Bullets of highest-priority actions.

Formatting rule:
- Do NOT use markdown heading symbols like #, ##, or ### anywhere in the output.

Behavior Rules:
- Do NOT invent messages or assume missing context
- Only use provided chat logs
- If unclear, respond exactly with "Insufficient context in this section"
- Be accurate and structured
- Prioritize correctness over creativity
- Preserve facts exactly; do not alter meaning"""


# ============================================================================
# Bot Setup
# ============================================================================

log_handler = logging.FileHandler(filename='discord.log', encoding='utf-8', mode='w')
intents = discord.Intents.default()
intents.message_content = True
intents.messages = True

bot = commands.Bot(command_prefix='!', intents=intents)
commands_synced = False


# ============================================================================
# Types
# ============================================================================

@dataclass(frozen=True)
class DayPayload:
    start_dt: datetime
    end_dt: datetime
    logs: str
    message_count: int


@dataclass(frozen=True)
class DaySummary:
    start_dt: datetime
    end_dt: datetime
    message_count: int
    text: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


# ============================================================================
# Permissions / Validation
# ============================================================================

def load_command_permissions() -> dict[str, Any]:
    """Load role/admin command permissions from command_permissions.json."""
    default_permissions = {
        'allowed_role_ids': [],
        'allow_administrator': True,
    }

    if not os.path.exists(PERMISSIONS_FILE):
        return default_permissions

    try:
        with open(PERMISSIONS_FILE, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
    except (OSError, json.JSONDecodeError) as err:
        print(f'[permissions] Failed to read {PERMISSIONS_FILE}: {err}')
        return default_permissions

    raw_ids = data.get('allowed_role_ids', [])
    allowed_role_ids: list[int] = []
    for role_id in raw_ids:
        try:
            allowed_role_ids.append(int(role_id))
        except (TypeError, ValueError):
            continue

    return {
        'allowed_role_ids': allowed_role_ids,
        'allow_administrator': bool(data.get('allow_administrator', True)),
    }


def member_can_run_summarize(member: discord.Member) -> bool:
    """Check whether a guild member can run /summarize."""
    permissions = load_command_permissions()
    allowed_role_ids = set(permissions['allowed_role_ids'])

    if permissions['allow_administrator'] and member.guild_permissions.administrator:
        return True

    if not allowed_role_ids:
        return False

    return any(role.id in allowed_role_ids for role in member.roles)


def validate_summarize_context(interaction: discord.Interaction) -> str | None:
    """Return an error message when interaction cannot be summarized."""
    if not OPENROUTER_API_KEY:
        return 'OPENROUTER_API_KEY is not set in environment variables.'

    if interaction.channel is None or not hasattr(interaction.channel, 'history'):
        return 'This command can only be used in a text channel.'

    if not isinstance(interaction.user, discord.Member):
        return 'This command can only be used inside a server.'

    if not member_can_run_summarize(interaction.user):
        return f'You do not have permission to use this command. Configure allowed roles in {PERMISSIONS_FILE}.'

    return None


# ============================================================================
# OpenRouter Request Helpers
# ============================================================================

def request_openrouter_summary(model: str, chat_logs: str) -> dict[str, Any]:
    """Send a chat completion request and return summary, usage, and cost metadata."""
    headers = {
        'Authorization': f'Bearer {OPENROUTER_API_KEY}',
    }
    if OPENROUTER_SITE_URL:
        headers['HTTP-Referer'] = OPENROUTER_SITE_URL
    if OPENROUTER_SITE_NAME:
        headers['X-Title'] = OPENROUTER_SITE_NAME

    payload = {
        'model': model,
        'messages': [
            {
                'role': 'system',
                'content': SYSTEM_PROMPT,
            },
            {
                'role': 'user',
                'content': chat_logs,
            },
        ],
    }

    response = requests.post(
        url='https://openrouter.ai/api/v1/chat/completions',
        headers=headers,
        data=json.dumps(payload),
        timeout=120,
    )
    response.raise_for_status()

    data = response.json()
    usage = data.get('usage', {}) if isinstance(data.get('usage', {}), dict) else {}
    choices = data.get('choices', [])
    if not choices:
        return {
            'summary': 'No summary returned by model.',
            'usage': usage,
            'cost_usd': extract_cost_usd(data, response.headers),
        }

    message = choices[0].get('message', {})
    return {
        'summary': message.get('content', 'No summary text returned.'),
        'usage': usage,
        'cost_usd': extract_cost_usd(data, response.headers),
    }


def extract_cost_usd(data: dict[str, Any], headers: requests.structures.CaseInsensitiveDict[str]) -> float | None:
    """Try several API fields and headers to get request cost in USD."""
    usage = data.get('usage', {}) if isinstance(data.get('usage', {}), dict) else {}
    for key in ('cost', 'total_cost', 'cost_usd', 'total_cost_usd'):
        value = usage.get(key) or data.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue

    header_value = headers.get('x-openrouter-cost') or headers.get('x-cost-usd')
    if header_value is None:
        return None

    try:
        return float(header_value)
    except (TypeError, ValueError):
        return None


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Fallback cost estimate used when API does not return explicit cost."""
    pricing = MODEL_PRICING_PER_MILLION.get(model)
    if not pricing:
        return 0.0

    input_cost = (prompt_tokens / 1_000_000) * pricing['input']
    output_cost = (completion_tokens / 1_000_000) * pricing['output']
    return input_cost + output_cost


# ============================================================================
# Message Collection + Text Processing
# ============================================================================

def compact_line(value: str) -> str:
    """Normalize whitespace so prompt tokens are used on signal, not formatting noise."""
    return re.sub(r'\s+', ' ', value).strip()


def strip_markdown_headers(value: str) -> str:
    """Remove markdown heading symbols if model accidentally emits them."""
    return re.sub(r'(?m)^\s*#{1,6}\s*', '', value).strip()


def build_daily_logs(messages: list[discord.Message]) -> str:
    """Convert a day's messages into compact lines for the model prompt."""
    lines: list[str] = []
    total_chars = 0

    for msg in messages:
        text = compact_line(msg.clean_content)
        if not text:
            continue

        timestamp = msg.created_at.astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
        author = compact_line(msg.author.display_name)
        line = f'[{timestamp}] {author}: {text}'

        next_len = total_chars + len(line) + 1
        if next_len > MAX_DAY_LOG_CHARS:
            lines.append('[truncated to save tokens]')
            break

        lines.append(line)
        total_chars = next_len

    return '\n'.join(lines)


def build_daily_ranges(start_dt: datetime, end_dt: datetime) -> list[tuple[datetime, datetime]]:
    """Split a large window into day-sized windows for parallel requests."""
    ranges: list[tuple[datetime, datetime]] = []
    current = start_dt
    while current < end_dt:
        day_end = min(current + timedelta(days=1), end_dt)
        ranges.append((current, day_end))
        current = day_end
    return ranges


def resolve_period_window(period_value: str) -> tuple[datetime, datetime]:
    """Translate slash command period value into UTC start/end datetimes."""
    days = PERIOD_TO_DAYS.get(period_value, 7)
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days)
    return start_dt, end_dt


async def collect_day_payloads(channel: discord.abc.Messageable, start_dt: datetime, end_dt: datetime) -> list[DayPayload]:
    """Collect and package per-day messages ready for OpenRouter calls."""
    day_payloads: list[DayPayload] = []

    for day_start, day_end in build_daily_ranges(start_dt, end_dt):
        day_messages: list[discord.Message] = []
        async for msg in channel.history(limit=None, after=day_start, before=day_end, oldest_first=True):
            if msg.author.bot:
                continue
            if not msg.clean_content.strip():
                continue
            day_messages.append(msg)

        day_logs = build_daily_logs(day_messages)
        if not day_logs:
            continue

        day_payloads.append(
            DayPayload(
                start_dt=day_start,
                end_dt=day_end,
                logs=day_logs,
                message_count=len(day_messages),
            )
        )

    return day_payloads


# ============================================================================
# Summarization Pipeline
# ============================================================================

async def summarize_day_payload(model: str, payload: DayPayload) -> DaySummary:
    """Run one day summary request in a worker thread."""
    result = await asyncio.to_thread(request_openrouter_summary, model, payload.logs)
    usage = result.get('usage', {}) if isinstance(result.get('usage', {}), dict) else {}
    prompt_tokens = int(usage.get('prompt_tokens', 0) or 0)
    completion_tokens = int(usage.get('completion_tokens', 0) or 0)

    request_cost = result.get('cost_usd')
    if request_cost is None:
        request_cost = estimate_cost_usd(model, prompt_tokens, completion_tokens)

    return DaySummary(
        start_dt=payload.start_dt,
        end_dt=payload.end_dt,
        message_count=payload.message_count,
        text=strip_markdown_headers(str(result.get('summary', 'No summary text returned.'))),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=float(request_cost),
    )


async def run_day_summaries(model: str, day_payloads: list[DayPayload]) -> list[DaySummary]:
    """Run all day summaries concurrently and return them in chronological order."""
    tasks = [
        asyncio.create_task(summarize_day_payload(model, payload))
        for payload in day_payloads
    ]

    results: list[DaySummary] = []
    for task in asyncio.as_completed(tasks):
        results.append(await task)

    results.sort(key=lambda row: row.start_dt)
    return results


def build_summary_file(day_summaries: list[DaySummary]) -> str:
    """Create final summary.txt content.

    Current output contract:
    - Start immediately with summary sections.
    - End with a single total price line.
    """
    if not day_summaries:
        return 'Insufficient context in this section\n\nTotal price (USD): 0.000000'

    sections = [summary.text for summary in day_summaries if summary.text]
    if not sections:
        sections = ['Insufficient context in this section']

    total_cost = sum(summary.cost_usd for summary in day_summaries)
    sections.append(f'Total price (USD): {total_cost:.6f}')
    return '\n\n'.join(sections)


# ============================================================================
# Discord Events + Command Handlers
# ============================================================================

@bot.event
async def on_ready() -> None:
    """Sync command tree once after bot login."""
    global commands_synced

    if commands_synced:
        print(f'Logged in as {bot.user.name} - {bot.user.id}')
        return

    # Clear stale global registrations, then re-register local command before guild sync.
    bot.tree.clear_commands(guild=None)
    bot.tree.add_command(summarize, override=True)
    await bot.tree.sync()

    if TARGET_GUILD:
        bot.tree.copy_global_to(guild=TARGET_GUILD)
        synced = await bot.tree.sync(guild=TARGET_GUILD)
        print(f'Synced {len(synced)} slash command(s) to guild {TARGET_GUILD.id}')
    elif bot.guilds:
        for guild in bot.guilds:
            bot.tree.copy_global_to(guild=guild)
            synced = await bot.tree.sync(guild=guild)
            print(f'Synced {len(synced)} slash command(s) to guild {guild.name} ({guild.id})')
        print('Commands are synced per-guild for immediate availability.')

    commands_synced = True
    print(f'Logged in as {bot.user.name} - {bot.user.id}')


@bot.tree.command(name='summarize', description='Summarize this channel using a preset time range')
@app_commands.describe(
    period='Pick a time range',
    quality='Summary quality and model tier',
)
@app_commands.choices(
    period=[
        app_commands.Choice(name='Last 24 hours', value='1d'),
        app_commands.Choice(name='Last 3 days', value='3d'),
        app_commands.Choice(name='Last 7 days', value='7d'),
        app_commands.Choice(name='Last 30 days', value='30d'),
    ],
    quality=[
        app_commands.Choice(name='Low', value='low'),
        app_commands.Choice(name='Medium', value='medium'),
        app_commands.Choice(name='High', value='high'),
    ],
)
async def summarize(
    interaction: discord.Interaction,
    period: app_commands.Choice[str],
    quality: app_commands.Choice[str],
) -> None:
    """Summarize current channel, one request per day, then combine into summary.txt."""
    error_message = validate_summarize_context(interaction)
    if error_message:
        await interaction.response.send_message(error_message, ephemeral=True)
        return

    await interaction.response.defer(thinking=True)

    start_dt, end_dt = resolve_period_window(period.value)
    if end_dt <= start_dt:
        await interaction.followup.send('endDate must be later than startDate.', ephemeral=True)
        return

    model = MODEL_BY_QUALITY.get(quality.value, DEFAULT_MODEL)
    print(
        f'[summarize] Collecting messages from channel={interaction.channel_id} '
        f'between {start_dt.isoformat()} and {end_dt.isoformat()}'
    )

    day_payloads = await collect_day_payloads(interaction.channel, start_dt, end_dt)
    if not day_payloads:
        await interaction.followup.send('No messages found in this channel for that time range.')
        return

    try:
        day_summaries = await run_day_summaries(model, day_payloads)
    except requests.HTTPError as err:
        detail = err.response.text if err.response is not None else str(err)
        await interaction.followup.send(f'OpenRouter request failed: {detail[:1500]}')
        return
    except Exception as err:
        await interaction.followup.send(f'Unexpected error during summarization: {err}')
        return

    combined_summary = build_summary_file(day_summaries)
    file_obj = discord.File(io.BytesIO(combined_summary.encode('utf-8')), filename='summary.txt')
    await interaction.followup.send(file=file_obj)


if __name__ == '__main__':
    if not TOKEN:
        raise RuntimeError('DISCORD_TOKEN is not set in environment variables.')

    bot.run(TOKEN, log_handler=log_handler, log_level=logging.DEBUG)