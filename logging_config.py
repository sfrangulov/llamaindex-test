import logging
import colorlog

def setup_logging() ->None:
    """Logger"""
    handler = logging.StreamHandler()

    handler.setFormatter(colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(name)s:%(reset)s %(message)s",
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'bold_red',
        }
    ))
    logging.basicConfig(level=logging.DEBUG, handlers=[handler])
