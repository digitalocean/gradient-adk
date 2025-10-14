import asyncio
from typing import Optional


DEFAULT_RETRY_STATUSES = {429, 500, 502, 503, 504}


async def async_backoff_sleep(
    attempt: int,
    base: float = 0.5,
    cap: float = 8.0,
    jitter: float = 0.2,
    retry_after: Optional[float] = None,
):
    """
    Exponential backoff with optional Retry-After handling.
    attempt starts at 1.
    """
    if retry_after is not None:
        await asyncio.sleep(min(retry_after, cap))
        return
    delay = min(cap, base * (2 ** (attempt - 1)))
    # light jitter: +/- jitter * delay
    jitter_amt = delay * jitter
    await asyncio.sleep(delay + (jitter_amt if attempt % 2 else -jitter_amt))
