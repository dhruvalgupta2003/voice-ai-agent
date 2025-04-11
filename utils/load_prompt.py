import os
from config.load_config import Config
from utils.logger import get_logger

# load config
config = Config()

# setup logging
logger = get_logger()

def load_prompt(file_name):
    """Load prompt from a file."""
    dir_path = config.BASE_DIR
    prompt_path = os.path.join(dir_path, 'prompts', f'{file_name}.txt')

    try:
        with open(prompt_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except FileNotFoundError:
        logger.error(f"Could not find file: {prompt_path}")
        raise
