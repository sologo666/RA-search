# Usage:
#   export TELEGRAM_BOT_TOKEN="<your_bot_token>"
#   export RA_API_BASE_URL="https://api.example.com/ra"  # TODO: Replace with your RA API base URL
#   export RA_API_TOKEN="<your_ra_api_token>"            # TODO: Replace with your RA API token
#   python bot.py

from __future__ import annotations
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, List, Optional, Tuple

import httpx
from telegram import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    Update,
)
from telegram.constants import ParseMode
from telegram.ext import (
    AIORateLimiter,
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ConversationHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# Conversation states
CITY, COUNTRY, DATE_RANGE, FILTERS, CONFIRM_SEARCH = range(5)

# Pagination
PAGE_SIZE = 5

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class Event:
    """Resident Advisor event model."""

    title: str
    date: str
    time: Optional[str]
    venue: str
    city: str
    country: str
    genres: List[str]
    url: str


def _ensure_env(name: str) -> str:
    """Return environment variable value or raise a clear error."""

    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


async def fetch_events(
    city: str,
    country_code: Optional[str],
    start_date: date,
    end_date: Optional[date],
    genre: Optional[str],
    event_type: Optional[str],
    weekend_only: bool = False,
) -> List[Event]:
    """Fetch events from Resident Advisor via an HTTP API.

    This function assumes the existence of an HTTP JSON API (for example, an Apify
    Resident Advisor scraper). Adjust the URL, payload, and response parsing to
    match the real service.
    """

    base_url = _ensure_env("RA_API_BASE_URL")
    token = os.getenv("RA_API_TOKEN")
    url = f"{base_url.rstrip('/')}/events"

    payload: dict[str, Any] = {
        "city": city,
        "country_code": country_code,
        "start_date": start_date.isoformat(),
        "end_date": (end_date or start_date).isoformat(),
        "genre": genre,
        "event_type": event_type,
        "weekend_only": weekend_only,
    }

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    logger.info("Fetching events for city=%s start=%s end=%s", city, start_date, end_date)
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.exception("Error calling RA API: %s", exc)
            raise

    data = response.json()

    events: List[Event] = []
    for item in data.get("events", []):  # TODO: align with real API fields
        events.append(
            Event(
                title=item.get("title", "Untitled"),
                date=item.get("date", "Unknown date"),
                time=item.get("time"),
                venue=item.get("venue", "Unknown venue"),
                city=item.get("city", city),
                country=item.get("country", country_code or ""),
                genres=item.get("genres", []) or [],
                url=item.get("url", "https://ra.co/events"),
            )
        )

    return events


def parse_natural_date(text: str) -> date:
    """Parse common natural language date keywords."""

    lowered = text.strip().lower()
    today = date.today()

    if lowered in {"today", "tonight"}:
        return today
    if lowered == "tomorrow":
        return today + timedelta(days=1)

    # YYYY-MM-DD
    try:
        return datetime.strptime(text.strip(), "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError("Unsupported date format") from exc


def parse_date_range(text: str) -> Tuple[date, Optional[date], bool]:
    """Parse date or date range strings.

    Accepts formats like:
    - 2025-12-31
    - 2025-12-31 to 2026-01-02
    - today / tomorrow / tonight
    - this weekend (interpreted as upcoming Friday-Sunday)
    """

    lowered = text.strip().lower()
    today = date.today()

    if lowered == "this weekend":
        # Upcoming Friday to Sunday
        days_ahead = (4 - today.weekday()) % 7  # Friday = 4
        start = today + timedelta(days=days_ahead)
        end = start + timedelta(days=2)
        return start, end, True

    if " to " in lowered:
        start_text, end_text = lowered.split(" to ", 1)
        start = parse_natural_date(start_text)
        end = parse_natural_date(end_text)
        if end < start:
            raise ValueError("End date must be after start date")
        return start, end, False

    single_date = parse_natural_date(text)
    return single_date, None, False


def format_event(event: Event) -> str:
    """Format an event for Telegram output."""

    time_part = f" {event.time}" if event.time else ""
    genres = ", ".join(event.genres) if event.genres else "N/A"
    return (
        f"🎶 {event.title}\n"
        f"📍 {event.venue}, {event.city}, {event.country}\n"
        f"📅 {event.date}{time_part}\n"
        f"🎛 Genres: {genres}\n"
        f"🔗 {event.url}"
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""

    message = (
        "Welcome to the Resident Advisor Event Finder!\n"
        "Search for electronic music events by city and date.\n\n"
        "Examples:\n"
        "• /search\n"
        "• Berlin, tomorrow\n"
        "• Amsterdam, 2025-12-31 to 2026-01-02\n"
        "Use /help for detailed instructions."
    )
    await update.message.reply_text(message)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command."""

    message = (
        "Commands:\n"
        "/start - Welcome message\n"
        "/help - Show this help\n"
        "/search - Guided search flow\n"
        "/cancel - Cancel current search\n\n"
        "You can also send messages like 'Berlin, tomorrow' or 'Amsterdam, 2025-12-31 to 2026-01-02'."
    )
    await update.message.reply_text(message)


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancel the conversation."""

    context.user_data.clear()
    await update.message.reply_text("Search cancelled. Use /search to start again.")
    return ConversationHandler.END


async def search_entry(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Start the search conversation."""

    context.user_data.clear()
    context.user_data["search"] = {}
    await update.message.reply_text("Which city are you interested in?")
    return CITY


async def ask_country(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle city input and ask for country code."""

    city = update.message.text.strip()
    context.user_data.setdefault("search", {})["city"] = city
    await update.message.reply_text(
        "Country code? (e.g., de, us). Leave empty for default (Berlin -> de, otherwise us)."
    )
    return COUNTRY


async def ask_date_range(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle country input and ask for date range."""

    country_code = update.message.text.strip()
    city = context.user_data.get("search", {}).get("city", "").lower()
    if not country_code:
        country_code = "de" if city == "berlin" else "us"

    context.user_data.setdefault("search", {})["country_code"] = country_code.lower()
    await update.message.reply_text(
        "Enter date or range (e.g., 2025-12-31 or 2025-12-31 to 2026-01-02). Keywords: today, tomorrow, this weekend."
    )
    return DATE_RANGE


async def ask_filters(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle date input and begin filter prompts."""

    text = update.message.text.strip()
    try:
        start, end, weekend = parse_date_range(text)
    except ValueError:
        await update.message.reply_text(
            "Could not parse the date. Please use formats like 2025-12-31 or 2025-12-31 to 2026-01-02, or say 'this weekend'."
        )
        return DATE_RANGE

    search = context.user_data.setdefault("search", {})
    search["start_date"] = start
    search["end_date"] = end
    search["weekend_only"] = weekend
    context.user_data["filter_stage"] = "genre"

    await update.message.reply_text(
        "Any genre filter? (comma-separated like 'techno, house'). Send 'skip' to ignore."
    )
    return FILTERS


async def handle_filters(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle filter inputs in stages (genre -> event type -> weekend flag)."""

    text = update.message.text.strip()
    stage = context.user_data.get("filter_stage", "genre")
    search = context.user_data.setdefault("search", {})

    if stage == "genre":
        if text.lower() != "skip" and text:
            search["genre"] = text
        else:
            search["genre"] = None
        context.user_data["filter_stage"] = "event_type"
        await update.message.reply_text(
            "Event type? (club / festival / day party). Send 'skip' to ignore."
        )
        return FILTERS

    if stage == "event_type":
        normalized = text.strip().lower()
        if normalized in {"club", "festival", "day party"}:
            search["event_type"] = normalized
        elif normalized == "skip" or not normalized:
            search["event_type"] = None
        else:
            await update.message.reply_text(
                "Invalid event type. Please choose from club, festival, day party, or send 'skip'."
            )
            return FILTERS

        context.user_data["filter_stage"] = "weekend"
        await update.message.reply_text("Only this weekend? (yes/no).")
        return FILTERS

    if stage == "weekend":
        normalized = text.strip().lower()
        if normalized in {"yes", "y"}:
            search["weekend_only"] = True
        elif normalized in {"no", "n", "skip", ""}:
            search.setdefault("weekend_only", False)
        else:
            await update.message.reply_text("Please answer with yes or no.")
            return FILTERS

        context.user_data.pop("filter_stage", None)
        return await confirm_search(update, context)

    context.user_data.pop("filter_stage", None)
    return await confirm_search(update, context)


async def confirm_search(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Summarize query and ask for confirmation."""

    search = context.user_data.get("search", {})
    start_date = search.get("start_date")
    end_date = search.get("end_date")
    date_text = start_date.isoformat() if isinstance(start_date, date) else ""
    if end_date:
        date_text = f"{date_text} to {end_date.isoformat()}"

    summary = (
        "Searching with:\n"
        f"City: {search.get('city')}\n"
        f"Country: {search.get('country_code')}\n"
        f"Date: {date_text}\n"
        f"Genre: {search.get('genre') or 'Any'}\n"
        f"Event type: {search.get('event_type') or 'Any'}\n"
        f"Weekend only: {search.get('weekend_only', False)}\n\n"
        "Proceed? (yes/no)"
    )
    await update.message.reply_text(summary)
    return CONFIRM_SEARCH


async def execute_search(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Execute the search after confirmation."""

    text = update.message.text.strip().lower()
    if text not in {"yes", "y", "no", "n"}:
        await update.message.reply_text("Please respond with yes or no.")
        return CONFIRM_SEARCH

    if text in {"no", "n"}:
        await update.message.reply_text("Search cancelled. Use /search to start again.")
        return ConversationHandler.END

    search = context.user_data.get("search", {})
    try:
        events = await fetch_events(
            city=search.get("city", ""),
            country_code=search.get("country_code"),
            start_date=search.get("start_date"),
            end_date=search.get("end_date"),
            genre=search.get("genre"),
            event_type=search.get("event_type"),
            weekend_only=bool(search.get("weekend_only")),
        )
    except Exception:
        await update.message.reply_text(
            "Sorry, something went wrong while fetching events. Please try again later."
        )
        return ConversationHandler.END

    context.user_data["last_results"] = events
    context.user_data["page"] = 0
    await send_results(update.message, events, page=0)
    return ConversationHandler.END


def build_pagination_keyboard(page: int, total: int) -> InlineKeyboardMarkup:
    """Build inline keyboard for pagination."""

    buttons: list[list[InlineKeyboardButton]] = []
    nav: list[InlineKeyboardButton] = []
    if page > 0:
        nav.append(InlineKeyboardButton("◀️ Prev", callback_data=f"PAGE:{page-1}"))
    if (page + 1) * PAGE_SIZE < total:
        nav.append(InlineKeyboardButton("Next ▶️", callback_data=f"PAGE:{page+1}"))
    if nav:
        buttons.append(nav)
    buttons.append([InlineKeyboardButton("🔁 New search", callback_data="NEW_SEARCH")])
    return InlineKeyboardMarkup(buttons)


async def send_results(message: Message, events: List[Event], page: int) -> None:
    """Send paginated event results."""

    if not events:
        await message.reply_text("No RA events found for your search. Try another date or city.")
        return

    start = page * PAGE_SIZE
    end = start + PAGE_SIZE
    slice_events = events[start:end]
    formatted = "\n\n".join(format_event(evt) for evt in slice_events)
    keyboard = build_pagination_keyboard(page, len(events))
    await message.reply_text(formatted, reply_markup=keyboard, parse_mode=ParseMode.HTML)


async def handle_pagination(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle pagination callback queries."""

    query: CallbackQuery = update.callback_query
    await query.answer()

    if query.data == "NEW_SEARCH":
        await query.message.reply_text("Starting a new search. Which city?")
        context.user_data.clear()
        context.user_data["search"] = {}
        return

    if not query.data or not query.data.startswith("PAGE:"):
        return

    try:
        page = int(query.data.split(":", 1)[1])
    except ValueError:
        return

    events = context.user_data.get("last_results", [])
    if not isinstance(events, list) or not events:
        await query.message.reply_text("No results to paginate. Start a new search with /search.")
        return

    context.user_data["page"] = page
    await query.message.edit_text(
        "\n\n".join(
            format_event(evt) for evt in events[page * PAGE_SIZE : (page + 1) * PAGE_SIZE]
        ),
        reply_markup=build_pagination_keyboard(page, len(events)),
        parse_mode=ParseMode.HTML,
    )


def parse_free_text(message: str) -> Optional[Tuple[str, str]]:
    """Parse free-text search messages in the format 'city, date'."""

    if "," not in message:
        return None
    parts = [part.strip() for part in message.split(",", 1)]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None
    return parts[0], parts[1]


async def free_text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle free-text search queries outside the conversation."""

    message = update.message.text
    parsed = parse_free_text(message)
    if not parsed:
        await update.message.reply_text(
            "I couldn't understand that. Try 'City, date' or use /search for guidance."
        )
        return

    city, date_text = parsed
    try:
        start, end, weekend = parse_date_range(date_text)
    except ValueError:
        await update.message.reply_text(
            "Could not parse the date. Try formats like 2025-12-31 or 'this weekend'."
        )
        return

    country_code = "de" if city.strip().lower() == "berlin" else "us"

    try:
        events = await fetch_events(
            city=city,
            country_code=country_code,
            start_date=start,
            end_date=end,
            genre=None,
            event_type=None,
            weekend_only=weekend,
        )
    except Exception:
        await update.message.reply_text(
            "Sorry, something went wrong while fetching events. Please try again later."
        )
        return

    context.user_data["last_results"] = events
    context.user_data["page"] = 0
    await send_results(update.message, events, page=0)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log errors raised by the dispatcher."""

    logger.exception("Update %s caused error %s", update, context.error)


def build_conversation_handler() -> ConversationHandler:
    """Build the ConversationHandler for /search."""

    return ConversationHandler(
        entry_points=[CommandHandler("search", search_entry)],
        states={
            CITY: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_country)],
            COUNTRY: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_date_range)],
            DATE_RANGE: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_filters)],
            FILTERS: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_filters)],
            CONFIRM_SEARCH: [MessageHandler(filters.TEXT & ~filters.COMMAND, execute_search)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
        name="search_conversation",
        persistent=False,
    )


def main() -> None:
    """Run the bot."""

    token = _ensure_env("TELEGRAM_BOT_TOKEN")
    application = (
        Application.builder()
        .token(token)
        .rate_limiter(AIORateLimiter())
        .build()
    )

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("cancel", cancel))
    application.add_handler(build_conversation_handler())
    application.add_handler(CallbackQueryHandler(handle_pagination))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, free_text_handler))
    application.add_error_handler(error_handler)

    application.run_polling()


if __name__ == "__main__":
    main()
