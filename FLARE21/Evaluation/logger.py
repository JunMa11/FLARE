import os
import sys

from loguru import logger as loguru_logger

loguru_logger.configure(
    handlers=[dict(sink=sys.stderr, filter=lambda record: record["extra"]["console"], level="DEBUG",),],
)


logger = loguru_logger.bind(console=True)


def add_file_handler_to_logger(
    name: str, dir_path="./logs", level="DEBUG",
):
    loguru_logger.add(
        sink=os.path.join(dir_path, f"{name}-{level}.log"),
        level=level,
        filter=lambda record: "console" in record["extra"],
        enqueue=True,
    )
