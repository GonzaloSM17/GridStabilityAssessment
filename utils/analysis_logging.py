"""Consistent file and console logging for published analysis workflows."""

import logging
from pathlib import Path


class _Utf8AppendHandler(logging.Handler):
    """Append one record at a time, avoiding persistent Windows file locks."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path

    def emit(self, record) -> None:
        try:
            message = self.format(record)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(f"{message}\n")
        except OSError:
            self.handleError(record)


def configure_analysis_logger(output_directory: str | Path, workflow: str) -> logging.Logger:
    """Return one UTF-8 logger writing ``analysis.log`` in an output root."""
    directory = Path(output_directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    logger_name = f"grid_stability_assessment.{workflow}.{directory}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = _Utf8AppendHandler(directory / "analysis.log")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
