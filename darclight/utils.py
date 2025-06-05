"""Module to provide utility tools for the reduction and analysis process."""
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

def enable_logging(file:bool=True, console:bool=False, filename:str='log'):
    """later"""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    if root_logger.hasHandlers():
        root_logger.info("Logging is already enabled.")
        return None

    if file:
        log_dir = './logs'
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        logfile = os.path.join(log_dir, f"{filename}_{timestamp}.log")

        file_handler = logging.FileHandler(logfile)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        root_logger.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
        root_logger.addHandler(console_handler)

    logger.info("Logging has been enabled.")
