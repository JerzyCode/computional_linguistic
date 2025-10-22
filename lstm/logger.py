import logging

log = logging.getLogger("lstm")
log.setLevel(logging.INFO)

formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# file_handler = logging.FileHandler("app.log", mode="a", encoding="utf-8")
# file_handler.setFormatter(formatter)

log.addHandler(console_handler)
# logger.addHandler(file_handler)
