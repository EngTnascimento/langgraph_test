import logging


def get_logger(name, level=logging.INFO, node=None):
    console_format = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s]  [%(message)s]"
    )
    logger = logging.getLogger(name)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    return logger
