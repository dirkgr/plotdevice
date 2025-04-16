from typing import *
import time
import logging

def logging_tqdm(
    i,
    *,
    logger: Optional[logging.Logger] = None,
    desc: str = "Working",
    total: Optional[int] = None,
    seconds_between_updates: int = 5
):
    if logger is None:
        logger = logging.getLogger()

    if total is None:
        if hasattr(i, "__len__"):
            total = len(i)

    def elapsed_string(time_in_seconds: float) -> str:
        s = ""
        if time_in_seconds >= (60*60):
            hours = time_in_seconds // (60*60)
            s += f"{hours:.0f}:"
            time_in_seconds -= (60*60) * hours
        if time_in_seconds >= 60 or len(s) > 0:
            minutes = time_in_seconds // 60
            if len(s) <= 0:
                s = f"{minutes:.0f}:"
            else:
                s += f"{minutes:02.0f}:"
            time_in_seconds -= 60 * minutes
        if len(s) <= 0:
            s = f"{time_in_seconds:.0f}s"
        else:
            s += f"{time_in_seconds:02.0f}"
        return s

    start = None
    last_print_time = None
    done = 0
    for item in i:
        if start is None:
            start = time.time()
            last_print_time = start
        yield item
        done += 1
        now = time.time()
        if now - last_print_time > seconds_between_updates:
            time_since_start = now - start
            speed = done / time_since_start
            if total is not None:
                fraction = done / total
                logger.info(f"{desc}, {done}/{total} ({fraction*100:.2f}%), {speed:.2f} per second, ETA {elapsed_string((total - done) / speed)}")
            else:
                logger.info(f"{desc}, {done} in {elapsed_string(time_since_start)}, {speed:.2f} per second")

            last_print_time = now

    if start is not None:
        time_since_start = time.time() - start
        if time_since_start > 5:
            logger.info(f"{desc} finished, {done} in {elapsed_string(time_since_start)}")

def make_tqdm(logger: Optional[logging.Logger] = None):
    def tqdm(*args, **kwargs):
        return logging_tqdm(*args, **kwargs, logger=logger)
    return tqdm
