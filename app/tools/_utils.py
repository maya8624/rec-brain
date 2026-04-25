from datetime import datetime, timezone
from zoneinfo import ZoneInfo

_SYDNEY = ZoneInfo("Australia/Sydney")
_FMT = "%a %d %b %Y at %I:%M %p %Z"


def fmt_dt_sydney(dt: datetime | str | None) -> str:
    """Format a UTC datetime (or ISO string) as Sydney local time."""
    if not dt:
        return ""
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
        except ValueError:
            return dt
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(_SYDNEY).strftime(_FMT)
