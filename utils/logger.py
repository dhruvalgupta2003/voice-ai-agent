import logging
import inspect
import os
from datetime import datetime
from colorama import init, Fore, Style
from config.load_config import Config

# Load Config
config = Config()

# Initialize colorama
init(autoreset=True)

# Ensure the logs directory exists
LOG_DIR = config.LOGS_DIR
os.makedirs(LOG_DIR, exist_ok=True)

# Generate a log file name based on current date
log_file_path = os.path.join(LOG_DIR, f"log_{datetime.now().strftime('%Y-%m-%d')}.log")

class ColorFormatter(logging.Formatter):
    COLOR_MAP = {
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.DEBUG: Fore.CYAN
    }

    def format(self, record):
        # Capture filename, line number, and code context
        frame = inspect.currentframe()
        while frame:
            if frame.f_globals.get('__name__') == record.module:
                break
            frame = frame.f_back
        if frame:
            info = inspect.getframeinfo(frame.f_back)
            record.line = info.lineno
            record.file = os.path.basename(info.filename)
            record.code = info.code_context[0].strip() if info.code_context else 'N/A'
        else:
            record.line = record.lineno
            record.file = os.path.basename(record.pathname)
            record.code = 'N/A'

        # Add color for terminal
        color = self.COLOR_MAP.get(record.levelno, "")
        # Format message using f-string to avoid string formatting issues
        message = (
            f"{color}[{record.levelname}] {record.file}:{record.line} "
            f"-> {record.code}\n{Style.RESET_ALL}{record.getMessage()}"
        )
        return message

class FileFormatter(logging.Formatter):
    def format(self, record):
        # Simpler format for file
        return f"[{record.levelname}] {record.asctime} | {record.filename}:{record.lineno} -> {record.getMessage()}"

def get_logger(name="custom_logger"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers
    if not logger.handlers:
        # Console handler with color
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(ColorFormatter())

        # File handler
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            "[%(levelname)s] %(asctime)s | %(filename)s:%(lineno)d -> %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S'
        ))

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger