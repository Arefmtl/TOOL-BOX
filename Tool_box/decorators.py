"""
decorators.py — Structured timing and logging decorator for pipeline steps.

Provides the @step decorator used across all TOOL-BOX tools for:
- Logging function start/end
- Measuring execution time
- Catching and re-raising exceptions with context
"""

import time
from functools import wraps
from loguru import logger


def step(name: str):
    """Structured timing and logging decorator for pipeline steps.

    Args:
        name: Human-readable name for the step (e.g., 'Data Loading', 'Model Training')

    Usage:
        @step('Data Cleaning')
        def clean_data(self, data):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(f"START | step={name} | function={func.__name__}")
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                duration = time.perf_counter() - start
                logger.success(
                    f"END | step={name} | function={func.__name__} | duration={duration:.3f}s"
                )
                return result
            except Exception as e:
                duration = time.perf_counter() - start
                logger.error(
                    f"FAIL | step={name} | function={func.__name__} | duration={duration:.3f}s | error={e}"
                )
                raise
        return wrapper
    return decorator