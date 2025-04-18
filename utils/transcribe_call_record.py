import time
import requests
from utils.logger import get_logger
from config.load_config import Config

config = Config()
logger = get_logger()

async def transcribe_user_call_record(filepath):
    try:
        start_time = time.time()
        # Open the written file for reading as binary
        with open(filepath, "rb") as audio_file:
            headers = {"Authorization": f"Bearer {config.GROQ_API_KEY}"}
            files = {"file": ("audio.wav", audio_file, "audio/wav")}

            response = requests.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers=headers,
                files=files,
                data={"model": "whisper-large-v3"},
                timeout=10
            )

        if response.status_code == 200:
            transcribed_text = response.json().get("text", "")
            duration = time.time() - start_time
            logger.info(f"TRANSCRIBED TEXT in {duration:.2f}s: {transcribed_text}")
            return transcribed_text.split(".")
        else:
            logger.error(f"Transcription API error: {response.status_code} {response.text}")
            return ""
    except requests.exceptions.Timeout:
        logger.error("Transcription request timed out")
        return ""
    except requests.exceptions.RequestException as e:
        logger.error(f"Transcription request error: {e}")
        return ""
    except Exception as e:
        logger.error(f"Transcription processing error: {e}")
        return ""

