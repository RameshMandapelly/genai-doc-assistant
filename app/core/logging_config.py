import logging
import sys


def configure_logging(level: int = logging.INFO) -> None:
    """
    Sets up a single shared logging configuration for the whole app:
    - Console output (so you see it in the uvicorn terminal)
    - File output to app.log (so you have a record after the terminal scrolls away)

    Call this once, at startup, before anything else logs. Guards against being
    called twice (e.g. under uvicorn --reload) so you don't get duplicated log lines.
    """
    root = logging.getLogger()
    if root.handlers:
        return  # already configured

    root.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    file_handler = logging.FileHandler("app.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # Third-party libraries are noisy at INFO level (HTTP calls, model loading, etc.)
    # Quiet them down so your own log lines aren't buried.
    for noisy_logger in ("httpx", "chromadb", "sentence_transformers", "urllib3"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
