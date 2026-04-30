import os, sys, time, logging
from contextlib import contextmanager
from pathlib import Path

DEBUG_ON = os.environ.get("UPSIDER_DEBUG", "0") == "1"
LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"

# --- logger setup ---
logger = logging.getLogger("upsider")
if not logger.handlers:
    logger.setLevel(logging.DEBUG if DEBUG_ON else logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    ch.setLevel(logging.DEBUG if DEBUG_ON else logging.INFO)
    fh = logging.FileHandler(LOG_FILE, mode="a")
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(ch)
    logger.addHandler(fh)

def set_debug(on: bool):
    """Toggle debug mode on/off at runtime."""
    global DEBUG_ON
    DEBUG_ON = bool(on)
    lvl = logging.DEBUG if on else logging.INFO
    for h in logger.handlers:
        h.setLevel(lvl)
    logger.setLevel(lvl)
    logger.info(f"Debug mode set to {on}")

def df_info(name, df, head=3):
    """Log shape, NA count, and preview of a DataFrame."""
    try:
        rows = len(df)
        cols = len(df.columns) if hasattr(df, "columns") else "n/a"
        na = int(df.isna().sum().sum()) if hasattr(df, "isna") else 0
        idx_name = getattr(df.index, "names", None)
        logger.info(f"[DF] {name}: shape=({rows},{cols}), total_NA={na}, index_names={idx_name}")
        if DEBUG_ON:
            sample = df.head(head).to_string()
            logger.debug(f"[DF] {name} head:\n{sample}")
    except Exception as e:
        logger.warning(f"df_info failed for {name}: {e}")

@contextmanager
def timer(msg: str):
    """Context manager for timing code blocks."""
    t0 = time.time()
    logger.info(f"[TIMER] start: {msg}")
    try:
        yield
    finally:
        dt = time.time() - t0
        logger.info(f"[TIMER] end: {msg} | {dt:.3f}s")

def trace(func):
    """Decorator to log entry/exit of functions when DEBUG_ON is true."""
    from functools import wraps
    @wraps(func)
    def wrapper(*args, **kwargs):
        if DEBUG_ON:
            logger.debug(f"[TRACE] enter {func.__name__}")
        result = func(*args, **kwargs)
        if DEBUG_ON:
            logger.debug(f"[TRACE] exit  {func.__name__}")
        return result
    return wrapper
