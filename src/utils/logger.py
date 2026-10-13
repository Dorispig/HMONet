"""Documentation string."""

# ============================================================
# Implementation comment.
# ============================================================
import logging

# ============================================================
# Implementation comment.
# ============================================================
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

# ============================================================
# Implementation comment.
# ============================================================

# ============================================================
# Implementation comment.
# ============================================================


__all__ = [
    "get_logger",
]


def get_logger(name="train", level=logging.INFO):
    """Documentation string."""
    # Implementation comment.
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()  # Implementation comment.

    # Implementation comment.
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Implementation comment.
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    return logger


if __name__ == "__main__":
    logger = get_logger()
    logger.info("Logger initialized successfully.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.info("Logger test completed.")

