"""darclight.__init__.py"""
import logging
import os
from datetime import datetime

LOG_DIR = './logs'
os.makedirs(LOG_DIR, exist_ok=True)

# generate filename
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
logfile = os.path.join(LOG_DIR, f"log_{timestamp}.log")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False

if not logger.hasHandlers():
    # file handler
    file_handler = logging.FileHandler(logfile)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(file_handler)
